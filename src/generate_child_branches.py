import os
import random
import numpy as np

from geomdl import BSpline, operations, utilities
from typing import List, Literal, Tuple

from src.config import Tree
from src.tube_functions import (
    cylinder,
    random_spline,
    RCA_vessel_curve,
    rotate_branch
)


def generate_child_branches(
    parent_points: List[float],
    children: List[Tree],
    n_points: int,
    curve_type: Literal["cylinder", "spline", "RCA"]
) -> List[Tree]:
    """Recursively generate child branches.

    Args:
        tree (List[Tree]): list of parent Tree branches
        n_points (int): number of points to generate for branch
        curve_type (Literal[&quot;cylinder&quot;, &quot;spline&quot;, &quot;RCA&quot;]): branch centerline type

    Returns:
        children (List[Tree]): list of child Tree branches
    """
    for i in range(len(children)):
        # Generate current child branch
        children[i] = generate_branch(
            parent_points,
            children[i],
            n_points,
            curve_type
        )
        
        # Recursively generate child's childrens' branches
        if children[i].children is not None:
            children[i].children = generate_child_branches(
                children[i].points,
                children[i].children,
                n_points,
                curve_type
            )

    return children


def generate_branch(
    parent_points: List[float],
    tree: Tree,
    n_points: int,
    curve_type: Literal["cylinder", "spline", "RCA"],
    **kwargs
) -> Tuple[List[float], List[float]]:
    """Generate a single child branch.

    Args:
        n_points (int): numer of points in branch
        curve_type (Literal[&quot;cylinder&quot;, &quot;spline&quot;, &quot;RCA&quot;]): branch centerline type

    Returns:
        points: (List[Float]): list of branch centerline points
        d_points: (List[Float]): list of branch centerline point derivatives
    """
    # Randomly choose branch length
    branch_length = random.uniform(tree.min_length, tree.max_length)

    # Randomly choose bifurcation location
    positions = [int(pp * n_points) for pp in tree.parametric_position]
    pos = random.randint(positions[0], positions[1])

    # Generate theta and phi
    theta = np.random.randint(30, 60) * (-1) ** random.getrandbits(1)
    phi = np.random.randint(45, 75) * (-1) ** random.getrandbits(1)

    if curve_type == "cylinder":
        C, _ = cylinder(branch_length, n_points)
        recentered_C = C - C[0, :] + parent_curve[pos, :]
        rotated_C = rotate_branch(recentered_C, theta, phi).tolist()
        branch_C = np.array(rotated_C)
        dC = np.subtract(np.array(rotated_C[1:]), np.array(rotated_C[:-1]))
    elif curve_type == "spline":
        num_control_points = random.randint(4,8)
        C, _ = random_spline(branch_length, 3, num_control_points, n_points)
        origin_centered_C = C - C[0,:]
        branch_C = origin_centered_C + parent_curve[pos,:]
        rotated_ctrl_pts = rotate_branch(branch_C, theta, phi).tolist()

        branch = BSpline.Curve()
        branch.degree = 3
        branch.ctrlpts = rotated_ctrl_pts
        branch.knotvector = utilities.generate_knot_vector(branch.degree, len(branch.ctrlpts))
        branch.delta = 0.01
        branch.n_points = n_points
        branch_C = np.array(branch.evalpts)

        ct1 = operations.tangent(branch, np.linspace(0, 1, branch.n_points).tolist(), normalize=True)
        curvetan = np.array(list((ct1)))  # ((x,y,z) (u,v,w)) format
        dC = curvetan[:, 1, :3]
    else:
        # can adjust rotations if branches are crossing/overlapping etc.
        rotations = np.array(
            [
                [-10 + random.randint(0, 5) * (-1) ** random.getrandbits(1), 0],
                [0, 15 + random.randint(0,5) * (-1) **random.getrandbits(1)],
                [-10 + random.randint(0, 5) * (-1) ** random.getrandbits(1), 10]
            ]
        )
        
        rng = np.random.default_rng()
        
        # ctrl_pts = np.load(os.path.join('RCA_branch_control_points/moderate', f"{tree.name}_ctrl_points.npy")) / 1000
        ctrl_pts = np.load(os.path.join('RCA_branch_control_points/moderate', "RCA_ctrl_points.npy")) / 1000
        mean_ctrl_pts = np.mean(ctrl_pts, axis=0)
        mean_ctrl_pts = np.mean(ctrl_pts, axis=0)
        std_ctrl_pts = np.std(ctrl_pts, axis=0)

        branch, d_cl = RCA_vessel_curve(n_points, mean_ctrl_pts, std_ctrl_pts, branch_length, rng, is_main=False, shear=True, warp=True)
        cl = branch - branch[0, :] + np.array(parent_points)[pos,:]
        # theta, phi = rotations[i]
        cl = rotate_branch(cl, theta, phi, center_rotation=False)

    # Fill points and d_points in tree (child branch)
    tree.points = cl.tolist()
    tree.d_points = d_cl.tolist()

    return tree