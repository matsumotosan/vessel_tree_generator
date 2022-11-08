from dataclasses import dataclass
from typing import List, Union


__all__ = [
    "Paths",
    "Files",
    "Flags",
    "Centerline",
    "Branch",
    "Stenosis",
    "Geometry",
    "Projection",
    "VesselConfig"
]


@dataclass
class Paths:
    """Class for storing path-related parameters."""
    # log_dir: str
    # data_dir: str
    save_dir: str
    dataset_name: str

@dataclass
class Files:
    """Class for storing file-related parameters."""
    control_points: str

@dataclass
class Flags:
    """Class for storing various options and flags."""
    random_seed: int
    save_surface: bool
    generate_projections: bool

@dataclass
class Centerline:
    """Class for storing centerline-related parameters."""
    n_points: int
    supersampling: int
    shear: bool
    warp: bool

@dataclass
class Branch:
    """Class for storing branch-related parameters."""
    name: str
    min_length: float
    max_length: float
    max_diameter: float
    control_point_path: str
    parametric_position: List
    children: Union[None, List['Branch']]

@dataclass
class Stenosis:
    """Class for storing stenosis-related parameters."""
    n_stenoses: Union[int, None]
    n_points: List
    constant_radius: bool
    severity: Union[int, None]
    position: Union[int, None]
    length: Union[int, None]
    stenosis_type: str
    min_radius: Union[float, None]
    max_radius: Union[float, None]
    branch_point: Union[List, None]

@dataclass
class Geometry:
    """Super class for storing geometry-related parameters."""
    centerline: Centerline
    main_branch: Branch
    side_branches: List[Branch]
    stenosis: Stenosis
    # n_branches: int = len(side_branches)

@dataclass
class Projection:
    """Class for storing projection-related parameters."""
    generate_projections: bool
    n_projections: bool
    image_dim: int
    pixel_spacing: float
    sid: float

@dataclass
class VesselConfig:
    """Top-level class for storing all vessel generation-related parameters."""
    paths: Paths
    # files: Files
    geometry: Geometry
    projection: Projection
    flags: Flags