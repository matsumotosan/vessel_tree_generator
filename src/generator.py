import os
import random
import numpy as np

from src.config import *
from src.generate_main_branch import generate_main_branch
from src.tube_functions import generate_side_branches, get_vessel_surface
from src.utils.save_specs import plot_surface, save_specs


class Generator:
    """Generator class for procedurally generating vessel trees of arbitrary depth.
    
    Args:
        paths (Paths): Paths object
        flags (Flags): Flags object
        geometry (Geometry): Geometry object
        projection (Projection): Projection object
    """
    
    def __init__(
        self,
        paths: Paths,
        flags: Flags,
        geometry: Geometry,
        projection: Projection
    ) -> None:

        self.paths = paths
        self.flags = flags
        self.geometry = geometry
        self.projection = projection

        self.rng = np.random.default_rng(self.flags.random_seed)
        self.vessel_specs = dict()
   
        self.geometry.side_branch = [self.geometry.side_branch] * self.geometry.n_branches
        self.n_points = self.geometry.centerline.supersampling * self.geometry.centerline.num_centerline_points

    def generate_tree(self) -> None:
        """Generate vessel tree."""
        # Generate main branch centerline
        self.cl, self.d_cl = generate_main_branch(
            vessel_type=self.geometry.vessel_type,
            min_length=self.geometry.main_branch.min_length,
            max_length=self.geometry.main_branch.max_length,
            n_points=self.n_points,
            control_point_path=self.paths.control_point_path,
            rng=self.rng,
            shear=self.geometry.centerline.shear,
            warp=self.geometry.centerline.warp
        )
        
        # Generate side branch centerlines
        self.tree, self.d_tree, self.branch_points = generate_side_branches(
            self.cl,
            self.d_cl,
            self.geometry.n_branches,
            self.n_points,
            self.geometry.side_branch,
            curve_type=self.geometry.vessel_type
        )

    def generate_surface(self):
        """General vessel surface."""
        # Generate radial/surface coordinates for centerline
        spline_array_list = []      # centerline + radial coordinate
        surface_coords = []         # (x,y,z) surface coordinates
        coords = np.empty((0, 3))
        skip = False

        for tree_idx, (C, dC) in enumerate(zip(self.tree, self.d_tree)):
            if tree_idx == 0:   # main branch
                key = "main_vessel"
                is_main_branch = True
                
                rand_stenoses = np.random.randint(0, 3)
                max_radius = [random.uniform(0.004, self.geometry.main_branch.max_diameter) / 2]
            else:   # side branch
                key = f"branch{tree_idx}"
                is_main_branch = False
                
                rand_stenoses = np.random.randint(0, 2)
                max_radius = [
                    # random.uniform(
                    #     cfg.geometry.side_branch[tree_idx].min_radius,
                    #     cfg.geometry.side_branch[tree_idx].max_radius
                    # )
                    random.uniform(
                        self.geometry.side_branch[tree_idx - 1].min_radius,
                        self.geometry.side_branch[tree_idx - 1].max_radius
                    )
                ]

            percent_stenosis = None
            stenosis_pos = None
            num_stenosis_points = None

            try:
                surface, R, stenosis_percent, stenosis_pos, n_stenosis_points = get_vessel_surface(
                    curve=self.cl,
                    derivatives=self.d_cl,
                    branch_points=self.branch_points,
                    num_centerline_points=self.n_points,
                    num_circle_points=self.n_theta,
                    radius=self.max_radius,
                    is_main_branch=is_main_branch,
                    num_stenoses=self.geometry.stenoses.n_stenoses,
                    constant_radius=self.geometry.stenoses.constant_radius,
                    stenosis_severity=self.geometry.stenoses.severity,
                    stenosis_position=self.geometry.stenoses.position,
                    stenosis_length=self.geometry.stenoses.length,
                    stenosis_type=self.geometry.stenoses.type,
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

    def save_surface(self, filename: str) -> None:
        """Save list of surface coordinates."""
        plot_surface(surface_coords, filename)

    def save_tree(self, filename: str) -> None:
        """Save vessel tree specifications in dict.
        
        Args:
            filename (str): json file name.
        """
        save_specs(filename, self.vessel_specs)
    
    def generate_projection(self):
        """Generate projection of vessel tree."""
        pass
    
    def save_projection(self, filename: str) -> None:
        """Save projection of vessel tree.
        
        Args:
            filename (str): json file name.
        """
        pass