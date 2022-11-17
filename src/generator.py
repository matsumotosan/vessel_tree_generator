import os
import random
import numpy as np

from typing import Union

from src.config import (
    Paths,
    Flags,
    Geometry,
    Projection
)
from src.fwd_projection_functions import generate_projection_images
from src.generate_main_branch import generate_main_branch
from src.generate_side_branches import generate_side_branches
from src.get_vessel_surface import get_vessel_surface
from src.utils import plot_surface, save_specs, create_nested_dir


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
        """Generate vessel centerline."""
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
            parent_curve=self.cl,
            parent_curve_derivative=self.d_cl,
            num_branches=self.geometry.n_branches,
            sample_size=self.n_points,
            side_branch_properties=self.geometry.side_branch,
            curve_type=self.geometry.vessel_type
        )

    def generate_surface(self) -> None:
        """General vessel surface from centerlines."""
        # Generate radial/surface coordinates for centerline
        self.spline_array_list = []      # centerline + radial coordinate
        self.surface_coords = []         # (x,y,z) surface coordinates
        self.coords = np.empty((0, 3))

        for tree_idx, (C, dC) in enumerate(zip(self.tree, self.d_tree)):
            if tree_idx == 0:   # main branch
                key = "main_branch"
                is_main_branch = True
                
                rand_stenoses = np.random.randint(0, 3)
                max_radius = [random.uniform(0.004, self.geometry.main_branch.max_diameter) / 2]
            else:   # side branch
                key = f"branch_{tree_idx}"
                is_main_branch = False
                
                rand_stenoses = np.random.randint(0, 2)
                max_radius = [
                    random.uniform(
                        self.geometry.side_branch[tree_idx - 1].min_radius,
                        self.geometry.side_branch[tree_idx - 1].max_radius
                        )
                    ]

            percent_stenosis = None
            stenosis_pos = None
            num_stenosis_points = None
            
            # Generate surface from centerline
            surface, R, stenosis_percent, stenosis_pos, n_stenosis_points = get_vessel_surface(
                curve=C,
                derivatives=dC,
                branch_points=self.branch_points,
                num_centerline_points=self.n_points,
                num_circle_points=self.geometry.n_theta,
                radius=max_radius,
                is_main_branch=is_main_branch,
                num_stenoses=self.geometry.stenoses.n_stenoses,
                constant_radius=self.geometry.stenoses.constant_radius,
                stenosis_severity=self.geometry.stenoses.severity,
                stenosis_position=self.geometry.stenoses.position,
                stenosis_length=self.geometry.stenoses.length,
                stenosis_type=self.geometry.stenoses.type,
                return_surface=True
            )
            
            # Append to list of centerline + radial coordinate
            spline_array = np.concatenate((C, np.expand_dims(R, axis=-1)), axis=1)[::self.geometry.centerline.supersampling,:]
            self.spline_array_list.append(spline_array)

            # Append to list of surface coordinates (x,y,z) by branch
            self.surface_coords.append(surface)
            
            # Append to array coordinates of entire vessel
            self.coords = np.concatenate((self.coords, surface.reshape(-1, 3)))
    
    def save_surface(self, filename: str, split_by_branch: bool = False) -> None:
        """Save vessel surface coordinates.

        Args:
            filename (str): numpy file name
            bybranch (bool): if True, saves coordinates split by branch. Otherwise saved as one array.
        """
        if split_by_branch:
            np.save(filename, self.surface_coords)
        else:
            np.save(filename, self.coords)

    def save_surface_plot(self, filename: str) -> None:
        """Save surface plot.
        
        Args:
            filename (str): surface plot filename.
            show (bool): displays surface plot if True. Default is False.
        """
        plot_surface(self.surface_coords, filename)

    def save_specs(self, filename: str) -> None:
        """Save vessel tree specifications in dict.
        
        Args:
            filename (str): json file name.
        """
        save_specs(filename, self.vessel_specs)

    def generate_projections(self, spline_idx):
        """Generate projection of vessel tree.
        
        Args:
            spline_idx (int): index of vessel tree
        """
        centered_coords = np.subtract(
            self.coords,
            np.mean(self.surface_coords[0].reshape(-1,3), axis=0)
        )

        use_RCA_angles = self.geometry.vessel_type == "RCA"
        images, theta_array, phi_array = generate_projection_images(
            centered_coords,
            spline_idx,
            self.projection.num_projections,
            self.projection.image_dim,
            self.paths.save_dir,
            self.paths.dataset_name,
            self.projection.pixel_spacing,
            self.projection.sid,
            RCA=use_RCA_angles
        )
        
        return images

    def prepare_dirs(self, top_dir: str) -> None:
        """Prepares directories to save outputs.
        
        Args:
            top_dir (str): top-level folder
        """
        create_nested_dir(top_dir, "surface")
        create_nested_dir(top_dir, "specs")
        if self.flags.generate_projections:
            create_nested_dir(top_dir, "projections")