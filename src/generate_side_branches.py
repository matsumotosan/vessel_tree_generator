import os
import random
import numpy as np

from geomdl import BSpline, operations, utilities

from src.tube_functions import (
    cylinder,
    random_spline,
    RCA_vessel_curve,
    rotate_branch
)


def generate_side_branches(
    parent_curve,
    parent_curve_derivative,
    num_branches,
    sample_size,
    side_branch_properties,
    curve_type="spline"
):
    '''
    Generates centerlines of branches that attach to parent curve at random locations
    parent_curve: Nx3 array of centerline points and radii
    curve_derivative: slope of parent curve
    num_branches (int): number of branches to add to parent curve; if 0, returns single vessel
    sample_size: number of points in parent curve (branches will have same number of interpolated centerline points)
    side_branch_properties: dict containing name, length in [m], max radius, min radius, and parametric position of each branch
    curve_type (string): "spline" for spline_tube, "cylinder" for cylinder, "RCA" for RCA
    '''
    if curve_type not in ["spline", "cylinder", "RCA"]:
        ValueError("Unknown curve_type. Possible types are \"spline\", \"cylinder\", or \"RCA\"")

    centerlines = [parent_curve]
    derivatives = [parent_curve_derivative]
    connections = [None]

    for i in range(num_branches):
        branch_length = side_branch_properties[i]["length"] * random.uniform(0.8, 1.2)
        positions = (np.array(side_branch_properties[i]["parametric_position"]) * sample_size).astype("int")
        pos = random.randint(positions[0], positions[1])
        if i > 0:
            while np.any(np.abs(np.array(connections[1:])-pos) < 0.07*sample_size):
                pos = np.random.randint(0.1*sample_size, sample_size-0.1*sample_size)

        theta = np.random.randint(30, 60)*(-1)**random.getrandbits(1)
        phi = np.random.randint(45, 75)*(-1)**random.getrandbits(1)
        if curve_type == "spline":
            num_control_points = random.randint(4,8)
            C, _ = random_spline(branch_length, 3, num_control_points, sample_size)
            origin_centered_C = C - C[0,:]
            branch_C = origin_centered_C + parent_curve[pos,:]
            rotated_ctrl_pts = rotate_branch(branch_C, theta, phi).tolist()

            branch = BSpline.Curve()
            branch.degree = 3
            branch.ctrlpts = rotated_ctrl_pts
            branch.knotvector = utilities.generate_knot_vector(branch.degree, len(branch.ctrlpts))
            branch.delta = 0.01
            branch.sample_size = sample_size
            branch_C = np.array(branch.evalpts)

            ct1 = operations.tangent(branch, np.linspace(0, 1, branch.sample_size).tolist(), normalize=True)
            curvetan = np.array(list((ct1)))  # ((x,y,z) (u,v,w)) format
            dC = curvetan[:, 1, :3]

        elif curve_type == "cylinder":
            C, _ = cylinder(branch_length, sample_size)
            recentered_C = C - C[0, :] + parent_curve[pos, :]
            rotated_C = rotate_branch(recentered_C, theta, phi).tolist()
            branch_C = np.array(rotated_C)
            dC = np.subtract(np.array(rotated_C[1:]), np.array(rotated_C[:-1]))
        else:
            # can adjust rotations if branches are crossing/overlapping etc.
            rotations = np.array([[-10+random.randint(0,5)*(-1)**random.getrandbits(1),0], [0, 15+random.randint(0,5)*(-1)**random.getrandbits(1)], [-10+random.randint(0,5)*(-1)**random.getrandbits(1),10]])
            rng = np.random.default_rng()
            control_points = np.load(os.path.join('RCA_branch_control_points/moderate', "{}_ctrl_points.npy".format(side_branch_properties[i]["name"]))) / 1000
            mean_ctrl_pts = np.mean(control_points, axis=0)
            stdev_ctrl_pts = np.std(control_points, axis=0)

            branch, dC = RCA_vessel_curve(sample_size, mean_ctrl_pts, stdev_ctrl_pts, branch_length, rng, is_main=False, shear=True, warp=True)
            branch_C = branch - branch[0, :] + parent_curve[pos,:]
            theta, phi = rotations[i]
            branch_C = rotate_branch(branch_C, theta, phi, center_rotation=False)

        centerlines.append(branch_C)
        derivatives.append(dC)
        connections.append(pos)
        
    return centerlines, derivatives, connections
