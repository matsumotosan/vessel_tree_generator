import numpy as np

from src.config import *
from src.generate_main_branch import generate_main_branch
from src.tube_functions import generate_side_branches, get_vessel_surface
from src.utils.save_specs import save_specs


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
        self.tree, self.d_tree, self.connections = generate_side_branches(
            self.cl,
            self.d_cl,
            self.geometry.n_branches,
            self.n_points,
            self.geometry.side_branch,
            curve_type=self.geometry.vessel_type
        )

    def generate_surface(self):
        """General vessel surface."""
        surface, R, stenosis_percent, stenosis_pos, n_stenosis_points = get_vessel_surface(
            curve=self.cl,
            derivatives=self.d_cl,
            branch_points=connections,
        )
    
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