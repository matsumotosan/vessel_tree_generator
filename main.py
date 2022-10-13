import os
import hydra
import random
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from copy import deepcopy
from tqdm import tqdm

from src.config import VesselConfig
from src.fwd_projection_functions import *
from src.generate_main_branch import generate_main_branch
from src.tube_functions import *
from src.utils import *


cs = ConfigStore.instance()
cs.store(name="vessel_config", node=VesselConfig)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: VesselConfig) -> None:
    print(f"Generating vessels with config:\n{OmegaConf.to_yaml(cfg)}")

    # Prepare directory to store generated vessels
    create_nested_dir(cfg.paths.save_dir, cfg.paths.dataset_name)
    
    # Initialize random number generator
    random.seed(cfg.flags.random_seed)
    rng = np.random.default_rng()

    # Calculate number of upsampled centerline points
    jj = cfg.geometry.centerline.supersampling
    num_centerline_points = cfg.geometry.centerline.num_centerline_points # number of interpolated centerline points to save
    supersampled_num_centerline_points = jj * num_centerline_points #use larger number of centerline points to create solid surface for projections, if necessary
    
    for spline_idx in tqdm(range(cfg.geometry.num_trees)):
        # Construct main branch centerline
        main_C, main_dC = generate_main_branch(
            vessel_type=cfg.geometry.vessel_type,
            min_length=cfg.geometry.main_branch.min_length,
            max_length=cfg.geometry.main_branch.max_length,
            n_points=supersampled_num_centerline_points,
            control_point_path=cfg.paths.control_point_path,
            rng=rng,
            shear=cfg.geometry.centerline.shear,
            warp=cfg.geometry.centerline.warp
        )

        # Construct side branches centerlines (add to main branch)
        tree, dtree, connections = branched_tree_generator(
            main_C,
            main_dC,
            cfg.geometry.num_branches,
            supersampled_num_centerline_points,
            cfg.geometry.side_branch,
            curve_type=cfg.geometry.vessel_type
        )

        # Save vessel specs to dict
        vessel_info = {
            'spline_index': int(spline_idx),
            'tree_type': cfg.geometry.vessel_type,
            'num_centerline_points': num_centerline_points,
            'theta_array': [],
            'phi_array': [],
            'main_vessel': deepcopy(cfg.geometry.main_branch)
            }

        for branch_index in range(cfg.geometry.num_branches):
            vessel_info[f"branch{branch_index + 1}"] = deepcopy(cfg.geometry.branches)

        # Generate radial/surface coordinates for centerline
        spline_array_list = []      # centerline + radial coordinate
        surface_coords = []         # (x,y,z) surface coordinates
        coords = np.empty((0, 3))
        skip = False
        
        # Generate surface points for each branch (main + side)
        for tree_idx, (C, dC) in enumerate(zip(tree, dtree)):
            if tree_idx == 0:   # main branch
                key = "main_vessel"
                is_main_branch = True
                
                rand_stenoses = np.random.randint(0, 3)
                max_radius = [random.uniform(0.004, cfg.geometry.main_branch.max_diameter) / 2]
            else:   # side branch
                key = f"branch{tree_idx}"
                is_main_branch = False
                
                rand_stenoses = np.random.randint(0, 2)
                max_radius = [
                    random.uniform(
                        cfg.geometry.side_branch[tree_idx].min_radius,
                        cfg.geometry.side_branch[tree_idx].max_radius
                    )
                ]

            percent_stenosis = None
            stenosis_pos = None
            num_stenosis_points = None

            # Generate surface for given centerline
            try:
                surface, R, percent_stenosis, stenosis_pos, num_stenosis_points = get_vessel_surface(
                    curve=C,
                    derivatives=dC,
                    branch_points=connections,
                    num_centerline_points=supersampled_num_centerline_points,
                    num_circle_points=cfg.geometry.num_theta,
                    radius=max_radius,
                    is_main_branch=is_main_branch,
                    num_stenoses=cfg.geometry.stenoses.n_stenoses,
                    constant_radius=cfg.geometry.stenoses.constant_radius,
                    stenosis_severity=cfg.geometry.stenoses.severity,
                    stenosis_position=cfg.geometry.stenoses.position,
                    stenosis_length=cfg.geometry.stenoses.length,
                    stenosis_type=cfg.geometry.stenoses.type,
                    return_surface=True
                    )
            except ValueError:
                print(f"Invalid sampling, skipping {tree_idx}.")
                skip = True
                continue

            # Append to list of centerline + radial coordinate
            spline_array = np.concatenate((C, np.expand_dims(R, axis=-1)), axis=1)[::jj,:]
            spline_array_list.append(spline_array)

            # Append to list of surface coordinates (x,y,z) by branch
            surface_coords.append(surface)
            
            # Append to array coordinates of entire vessel
            coords = np.concatenate((coords, surface.reshape(-1, 3)))

            # Update vessel_info dict for branch
            # vessel_info[key]['num_stenoses'] = int(rand_stenoses)
            # vessel_info[key]['max_radius'] = float(R[0]*1000)
            # vessel_info[key]['min_radius'] = float(R[-1]*1000)
            # if connections[tree_idx] is not None:
            #     vessel_info[key]['branch_point'] = int(connections[tree_idx]/jj)
            # if rand_stenoses > 0:
            #     vessel_info[key]['stenosis_severity'] = [float(i) for i in percent_stenosis]
            #     vessel_info[key]['stenosis_position'] = [int(i/jj) for i in stenosis_pos]
            #     vessel_info[key]['num_stenosis_points'] = [int(i/jj) for i in num_stenosis_points]

        if skip:
            continue

        # Plot 3D surface
        if cfg.flags.save_surface and spline_idx < 10:
            plot_surface(
                surface_coords,
                os.path.join(cfg.paths.save_dir, cfg.paths.dataset_name, f"{spline_idx:4d}_3Dsurface")
            )

        # Generate projections of vessel surface (optionally)
        if cfg.projections.generate_projections:
            vessel_info["ImagerPixelSpacing"] = cfg.projections.pixel_spacing
            vessel_info["SID"] = cfg.projections.sid

            # centering vessel at origin for cone-beam projections
            centered_coords = np.subtract(coords, np.mean(surface_coords[0].reshape(-1,3), axis=0))
            use_RCA_angles = cfg.geometry.vessel_type == "RCA"
            images, theta_array, phi_array = generate_projection_images(
                centered_coords,
                spline_idx,
                cfg.projections.num_projections,
                cfg.projections.image_dim,
                cfg.paths.save_dir,
                cfg.paths.dataset_name,
                cfg.projections.pixel_spacing,
                cfg.projections.sid,
                RCA=use_RCA_angles
            )
            vessel_info['theta_array'] = theta_array.tolist()
            vessel_info['phi_array'] = phi_array.tolist()

        # Save generated geometry (x, y, z, r)
        if not os.path.exists(os.path.join(cfg.paths.save_dir, cfg.paths.dataset_name, "labels", cfg.paths.dataset_name)):
            os.makedirs(os.path.join(cfg.paths.save_dir, cfg.paths.dataset_name, "labels", cfg.paths.dataset_name))
        if not os.path.exists(os.path.join(cfg.paths.save_dir, cfg.paths.dataset_name, "info")):
            os.makedirs(os.path.join(cfg.paths.save_dir, cfg.paths.dataset_name, "info"))

        tree_array = np.array(spline_array_list)
        np.save(os.path.join(cfg.paths.save_dir, cfg.paths.dataset_name, "labels", cfg.paths.dataset_name, "{:04d}".format(spline_idx)), tree_array)

        # Save geometry generation specs in json
        json_path = os.path.join(cfg.paths.save_dir, cfg.paths.dataset_name, "info", "{:04d}.info.0".format(spline_idx))
        # save_specs(json_path, vessel_info)


if __name__ == "__main__":
    main()
