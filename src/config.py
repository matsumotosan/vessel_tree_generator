from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from nptyping import NDArray, Float
from typing import Any, List, Union


__all__ = [
    "Paths",
    "Files",
    "Flags",
    "Centerline",
    "Tree",
    "Stenosis",
    "Geometry",
    "Projection",
    "VesselConfig"
]


@dataclass
class Flags:
    """Class for storing various options and flags."""
    n_vessels: int
    random_seed: int
    split_by_branch: bool
    save_surface_plot: bool
    generate_projections: bool

@dataclass
class Paths:
    """Class for storing path-related parameters."""
    save_dir: str
    dataset_name: str

@dataclass
class Centerline:
    """Class for storing centerline-related parameters."""
    n_points: int
    supersampling: int
    shear: bool
    warp: bool

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
class Tree:
    """Class for storing branch-related parameters."""
    name: str
    min_length: float
    max_length: float
    max_diameter: float
    parametric_position: Union[None, List[float]]
    children: Union[None, List[Tree]]
    points: Union[None, List[float]]
    d_points: Union[None, List[float]]

@dataclass
class Geometry:
    """Super class for storing geometry-related parameters."""
    n_generations: int
    n_branches: int
    n_theta: int
    length_factor: float
    dia_factor: float
    vessel_type: str
    centerline: Centerline
    stenosis: Stenosis

@dataclass
class Projection:
    """Class for storing projection-related parameters."""
    n_projections: bool
    image_dim: int
    pixel_spacing: float
    sid: float

@dataclass
class VesselConfig:
    """Top-level class for storing all vessel generation-related parameters."""
    flags: Flags
    paths: Paths
    geometry: Geometry
    projection: Projection