import os
import hydra
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import random
import json

from fwd_projection_functions import *
from generate_main_branch import generate_main_branch
from tube_functions import *
from utils import *


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print config to console
    print(f"Generating vessels with config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Prepare directory to store generated vessels
    create_nested_dir(cfg.save_path, cfg.dataset_name)
    
    # Initialize random number generator
    random.seed(cfg.random_seed)
    rng = np.random.default_rng()

    num_projections = cfg.geometry.projections.num_projections
    jj = cfg.geometry.centerline.supersampling
    num_centerline_points = cfg.geometry.centerline.num_centerline_points # number of interpolated centerline points to save
    supersampled_num_centerline_points = jj * num_centerline_points #use larger number of centerline points to create solid surface for projections, if necessary
    num_branches = cfg.geometry.num_branches  # set to 0 if not adding side branches
    
    for spline_idx in tqdm(range(cfg.geometry.num_trees)):
        
        vessel_info = {'spline_index': int(spline_idx),
                       'tree_type': [],
                       'num_centerline_points': num_centerline_points,
                       'theta_array': [],
                       'phi_array': [],
                       'main_vessel': deepcopy(cfg.geometry.main_branch)}
        
        for branch_index in range(num_branches):
            vessel_info[f"branch{branch_index + 1}"] = deepcopy(cfg.geometry.branches)

        # default is RCA; LCx/LAD single vessels and LCA tree will be implemented in future
        vessel_info["tree_type"].append(cfg.geometry.vessel_type)

        # Construct main branch
        length = random.uniform(
            cfg.geometry.main_branch.min_length,
            cfg.geometry.main_branch.max_length
        )
        
        main_C, main_dC = generate_main_branch(
            cfg.geometry.vessel_type,
            length,
            supersampled_num_centerline_points,
            control_point_path="RCA_branch_control_points/moderate",
            rng=rng,
            shear=cfg.geometry.centerline.shear,
            warp=cfg.geometry.centerline.warp
        )

        # Construct side branches
        tree, dtree, connections = branched_tree_generator(
            main_C,
            main_dC,
            num_branches,
            supersampled_num_centerline_points,
            cfg.branches.side,
            curve_type=cfg.vessel_type
        )

        num_theta = 120
        spline_array_list = []
        surface_coords = []
        coords = np.empty((0,3))

        ##############################################################
        # Generate radii and surface coordinates for centerline tree #
        ##############################################################
        skip = False
        for ind in range(len(tree)):
            C = tree[ind]
            dC = dtree[ind]
            if ind == 0:
                rand_stenoses = np.random.randint(0, 3)
                key = "main_vessel"
                main_is_true = True
                max_radius = [random.uniform(0.004, cfg.geometry.main_branch.max_diameter) / 2]
            else:
                rand_stenoses = np.random.randint(0, 2)
                max_radius = [random.uniform(cfg.branches.side[ind]['min_radius'], cfg.branches.side[ind]['max_radius'])]
                key = "branch{}".format(ind)
                main_is_true = False

            percent_stenosis = None
            stenosis_pos = None
            num_stenosis_points = None

            if cfg.num_stenoses is not None:
                rand_stenoses = cfg.num_stenoses

            try:
                X,Y,Z, new_radius_vec, percent_stenosis, stenosis_pos, num_stenosis_points = get_vessel_surface(C, dC, connections, supersampled_num_centerline_points, num_theta, max_radius,
                                                                                                         is_main_branch = main_is_true,
                                                                                                         num_stenoses=rand_stenoses,
                                                                                                         constant_radius=cfg.constant_radius,
                                                                                                         stenosis_severity=cfg.stenosis_severity,
                                                                                                         stenosis_position=cfg.stenosis_position,
                                                                                                         stenosis_length=cfg.stenosis_length,
                                                                                                         stenosis_type="gaussian",
                                                                                                         return_surface=True)
            except ValueError:
                print(f"Invalid sampling, skipping {spline_idx}.")
                skip = True
                continue

            spline_array = np.concatenate((C, np.expand_dims(new_radius_vec, axis=-1)), axis=1)[::jj,:]
            spline_array_list.append(spline_array)

            branch_coords = np.stack((X.T,Y.T,Z.T)).T
            surface_coords.append(branch_coords)
            coords = np.concatenate((coords,np.stack((X.flatten(), Y.flatten(), Z.flatten())).T))

            vessel_info[key]['num_stenoses'] = int(rand_stenoses)
            vessel_info[key]['max_radius'] = float(new_radius_vec[0]*1000)
            vessel_info[key]['min_radius'] = float(new_radius_vec[-1]*1000)
            if connections[ind] is not None:
                vessel_info[key]['branch_point'] = int(connections[ind]/jj)
            if rand_stenoses > 0:
                vessel_info[key]['stenosis_severity'] = [float(i) for i in percent_stenosis]
                vessel_info[key]['stenosis_position'] = [int(i/jj) for i in stenosis_pos]
                vessel_info[key]['num_stenosis_points'] = [int(i/jj) for i in num_stenosis_points]

        if skip:
            continue

        # Plot 3D surface
        if cfg.save_visualization:
            if spline_idx < 10:
                plot_surface(surface_coords, os.path.join(cfg.save_path, cfg.dataset_name, f"{spline_idx:4d}_3Dsurface"))

        ###################################
        ######       projections     ######
        ###################################
        if cfg.generate_projections:
            img_dim = 512
            ImagerPixelSpacing = 0.35
            SID = 1.2

            vessel_info["ImagerPixelSpacing"] = ImagerPixelSpacing
            vessel_info["SID"] = SID

            # centering vessel at origin for cone-beam projections
            centered_coords = np.subtract(coords, np.mean(surface_coords[0].reshape(-1,3), axis=0))
            use_RCA_angles = cfg.vessel_type == "RCA"
            images, theta_array, phi_array = generate_projection_images(centered_coords, spline_idx,
                                                                        num_projections, img_dim, cfg.save_path, cfg.dataset_name,
                                                                        ImagerPixelSpacing, SID, RCA=use_RCA_angles)
            vessel_info['theta_array'] = [float(i) for i in theta_array.tolist()]
            vessel_info['phi_array'] = [float(j) for j in phi_array.tolist()]

        #saves geometry as npy file (X,Y,Z,R) matrix
        if not os.path.exists(os.path.join(cfg.save_path, cfg.dataset_name, "labels", cfg.dataset_name)):
            os.makedirs(os.path.join(cfg.save_path, cfg.dataset_name, "labels", cfg.dataset_name))
        if not os.path.exists(os.path.join(cfg.save_path, cfg.dataset_name, "info")):
            os.makedirs(os.path.join(cfg.save_path, cfg.dataset_name, "info"))

        #saves geometry as npy file (X,Y,Z,R) matrix
        tree_array = np.array(spline_array_list)
        np.save(os.path.join(cfg.save_path, cfg.dataset_name, "labels", cfg.dataset_name, "{:04d}".format(spline_idx)), tree_array)

        # writes a text file for each tube with relevant parameters used to generate the geometry
        with open(os.path.join(cfg.save_path, cfg.dataset_name, "info", "{:04d}.info.0".format(spline_idx)), 'w+') as outfile:
            json.dump(vessel_info, outfile, indent=2)


if __name__ == "__main__":
    main()
