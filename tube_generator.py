import os
import hydra
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy

import numpy as np
import random
import json

from fwd_projection_functions import *
from tube_functions import *
from utils import *


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print config to console
    print(f"Generating vessels with config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Prepare directory to store generated vessels
    save_path = cfg.save_path
    dataset_name = cfg.dataset_name
    create_nested_dir(cfg.save_path, cfg.dataset_name)
    
    random.seed(cfg.random_seed)
    rng = np.random.default_rng()
    
    num_trees = cfg.geometry.num_trees

    jj = cfg.geometry.centerline.centerline_supersampling
    num_projections = cfg.geometry.projections.num_projections
    num_centerline_points = cfg.geometry.num_centerline_points # number of interpolated centerline points to save
    supersampled_num_centerline_points = jj * num_centerline_points #use larger number of centerline points to create solid surface for projections, if necessary
    num_branches = cfg.geometry.num_branches  # set to 0 if not adding side branches
    order = 3
    
    for i in range(num_trees):
        spline_index = i
        if (i + 1) % 10 == 0:
            print(f"Completed {spline_index + 1}/{num_trees} vessels.")

        #############################
        # Construct main branch     #
        #############################
        vessel_info = {'spline_index': int(spline_index),
                       'tree_type': [],
                       'num_centerline_points': num_centerline_points,
                       'theta_array': [],
                       'phi_array': [],
                       'main_vessel': deepcopy(cfg.branches.main)}
        
        for branch_index in range(num_branches):
            vessel_info[f"branch{branch_index + 1}"] = deepcopy(cfg.geometry.branches)

        # default is RCA; LCx/LAD single vessels and LCA tree will be implemented in future
        branch_ID = 1
        vessel_info["tree_type"].append("RCA")

        length = random.uniform(cfg.branches.main[branch_ID]['min_length'], cfg.branches.main[branch_ID]['max_length']) # convert to [m] to stay consistent with projection setup
        sample_size = supersampled_num_centerline_points

        if cfg.geometry.vessel_type == 'cylinder':
            main_C, main_dC = cylinder(length, supersampled_num_centerline_points)
        elif cfg.geometry.vessel_type == 'spline':
            main_C, main_dC = random_spline(length, order, np.random.randint(order + 1, 10), sample_size)
        else:
            RCA_control_points = np.load(os.path.join(cfg.geometry.control_point_path, "RCA_ctrl_points.npy")) / 1000 # [m] instead of [mm]
            mean_ctrl_pts = np.mean(RCA_control_points, axis=0)
            stdev_ctrl_pts = np.std(RCA_control_points, axis=0)
            main_C, main_dC = RCA_vessel_curve(sample_size, mean_ctrl_pts, stdev_ctrl_pts, length, rng, shear=cfg.shear, warp=cfg.warp)

        tree, dtree, connections = branched_tree_generator(main_C, main_dC, num_branches, sample_size, cfg.branches.side, curve_type=cfg.vessel_type)

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
                max_radius = [random.uniform(0.004, cfg.branches.main[branch_ID]['max_diameter']) / 2]

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
                print(f"Invalid sampling, skipping {i}.")
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
            if i < 10:
                plot_surface(surface_coords, os.path.join(save_path, dataset_name, f"{spline_index:4d}_3Dsurface"))

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
            images, theta_array, phi_array = generate_projection_images(centered_coords, spline_index,
                                                                        num_projections, img_dim, save_path, dataset_name,
                                                                        ImagerPixelSpacing, SID, RCA=use_RCA_angles)
            vessel_info['theta_array'] = [float(i) for i in theta_array.tolist()]
            vessel_info['phi_array'] = [float(j) for j in phi_array.tolist()]

        #saves geometry as npy file (X,Y,Z,R) matrix
        if not os.path.exists(os.path.join(save_path, dataset_name, "labels", dataset_name)):
            os.makedirs(os.path.join(save_path, dataset_name, "labels", dataset_name))
        if not os.path.exists(os.path.join(save_path, dataset_name, "info")):
            os.makedirs(os.path.join(save_path, dataset_name, "info"))

        #saves geometry as npy file (X,Y,Z,R) matrix
        tree_array = np.array(spline_array_list)
        np.save(os.path.join(save_path, dataset_name, "labels", dataset_name, "{:04d}".format(spline_index)), tree_array)

        # writes a text file for each tube with relevant parameters used to generate the geometry
        with open(os.path.join(save_path, dataset_name, "info", "{:04d}.info.0".format(spline_index)), 'w+') as outfile:
            json.dump(vessel_info, outfile, indent=2)


if __name__ == "__main__":
    main()
