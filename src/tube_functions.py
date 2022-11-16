from geomdl import BSpline, utilities, operations
import numpy as np
from src.fwd_projection_functions import *
import random
from src.augmentation import shear_centerlines, warp1

# we use the random library instead of numpy.random for most functions due to
# issues with numpy generating identical random numbers when using multiprocessing

def cylinder(length, num_points):
    '''
    Generates a straight cylinder
    :param max_radius: maximum radius of the cylinder
    :param length: length of cylinder (x direction)
    :return: Nx3 matrix of centerline points (C) and (N-1)x3 matrix of derivatives (dC)
    '''

    x = np.linspace(0, length, num_points)
    y = np.zeros((num_points,))
    z = np.zeros((num_points,))

    C = np.stack((x, y, z)).T
    dC = np.zeros((num_points,3))
    dC[:,0] = length/num_points

    return C, dC

def random_spline(length, degree, num_control_points, sample_size):
    '''
    produces a b-spline along the X-axis based on randomly generated control points and a uniform knot vector
    :param length: length of vessel (not path length)
    :param degree: B-spline order (typical values are 2-5)
    :param num_control_points:
    :param sample_size: number of discrete points to sample on centerline
    :return: NURBS-python curve object, which defines a b-spline
    '''

    max_curve_displacement = 0.3*length

    x = max_curve_displacement*np.array([random.random() for _ in range(num_control_points)])
    y = max_curve_displacement*np.array([random.random() for _ in range(num_control_points)])
    recentered_x = x - x[0]
    recentered_y = y - y[0]
    z = np.linspace(-length/2, length/2, num_control_points)
    ctrl_pts = np.stack((recentered_x, recentered_y, z), axis=1).tolist()
    rotated_ctrl_pts = rotate_branch(ctrl_pts, 0, 90, center_rotation=True).tolist() #switches orientation to X-axis, optional

    curve = BSpline.Curve()
    curve.degree = degree
    curve.ctrlpts = rotated_ctrl_pts
    #generates uniform knot vector
    curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
    curve.delta = 0.01
    curve.sample_size = sample_size

    C = np.array(curve.evalpts)

    ct1 = operations.tangent(curve, np.linspace(0, 1, curve.sample_size).tolist(), normalize=True)
    curvetan = np.array(list((ct1)))  # ((x,y,z) (u,v,w)) format
    dC = curvetan[:, 1, :3]

    return C, dC

def RCA_vessel_curve(sample_size, mean_ctrl_pts, stdev_ctrl_pts, length, rng, is_main=True, shear=False, warp=False):
    '''
    sample size: number of centerline points to interpolate
    mean_ctrl_pts: mean vessel control points to sample from
    stdev_ctrl_pts: standard deviation of mean vessel control points to sample from
    Typically 10-15 control points gives reasonable results
    length: desired length in [mm] of curve
    rng: numpy random generator instance
    is_main: determines if current vessel is the main branch or not
    shear: bool: apply shearing augmentation
    warp: bool: apply sin/cos based warping of point
    '''

    #random_ctrl_points = rng.normal(mean_ctrl_pts, stdev_ctrl_pts).reshape(-1,3) #if using for machine learning, avoid using gaussian sampling
    random_ctrl_points = rng.uniform(mean_ctrl_pts - 1.5*stdev_ctrl_pts, mean_ctrl_pts+stdev_ctrl_pts+1.5*stdev_ctrl_pts)
    if is_main:
        # for RCA, this ensures random sampling doesn't produce a non-physiological centerline
        # does not affect random splines or cylinders
        if random_ctrl_points[0,-1] - random_ctrl_points[1,-1] > 0.0001:
            alpha = rng.uniform(0.5,1)
            random_ctrl_points[1, -1] = random_ctrl_points[0,-1] + alpha*0.0015

    new_ctrl_points = random_ctrl_points.copy()

    if shear:
        new_ctrl_points = shear_centerlines(new_ctrl_points, 0.12)

    if warp:
        new_ctrl_points = warp1(new_ctrl_points, 0.1)

    curve = BSpline.Curve()
    curve.degree = 3
    curve.ctrlpts = new_ctrl_points.tolist()
    # generates uniform knot vector
    curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
    curve.delta = 0.01
    curve.sample_size = sample_size
    scaling = length/operations.length_curve(curve)
    curve = operations.scale(curve, scaling)

    C = np.array(curve.evalpts)

    ct1 = operations.tangent(curve, np.linspace(0, 1, curve.sample_size).tolist(), normalize=True)
    curvetan = np.array(list((ct1)))  # ((x,y,z) (u,v,w)) format
    dC = curvetan[:, 1, :3]

    return C, dC

def rotate_branch(control_points, theta, phi, center_rotation=False):
    '''
    :param control_points: Nx3 matrix of data points
    :param theta: axial rotation angle
    :param phi: elevation rotation angle
    :param center_rotation: if true, rotation origin is center of curve instead of first point in curve
    :return: rotated control points
    '''
    branch_points = np.array(control_points)
    if not center_rotation:
        rotation_point = branch_points[0,:]
    else:
        rotation_point = np.mean(branch_points, axis=0)

    translated_branch = np.subtract(branch_points, rotation_point)
    rotated_origin_branch = rotate_volume(phi, 0, theta, translated_branch)
    rotated_control_points = np.add(rotated_origin_branch, rotation_point)

    return rotated_control_points

def gaussian(mu, sigma, num_points):
    x = np.linspace(-2,2,num_points)
    bell_curve_vector = 1/(sigma * 2*np.pi)*np.exp(-0.5*((x-mu)/sigma)**2)
    return bell_curve_vector

def stenosis_generator(num_stenoses, radius_vector, branch_points, is_main = True, stenosis_severity=None, stenosis_position=None, stenosis_length=None, stenosis_type="gaussian"):
    '''
    :param num_stenoses: number of stenoses to create (typically 1 to 3)
    :param radius_vector: original radius at every point along centerline
    :param stenosis_severity: list: % diameter reduction for each stenosis. len(stenosis_severity) must equal num_stenoses
    :param stenosis_position: list: index of centerline point indicating location of stenosis
    :param stenosis_length: list: length of each stenosis w.r.t centerline coordinates. len(num_stenosis_points) must equal num_stenoses
    :param stenosis_type: string: Geometry of stenosis profile. Valid arguments are "gaussian" [TODO: implement "cosine"]
    :return: new radius vector containing stenoses
    '''
    # stenosis severity = % diameter reduction/2 since this is applied to the radius for 2-sided case
    if stenosis_severity is None:
        stenosis_severity = [random.uniform(0.3, 0.8) for _ in range(num_stenoses)]

    # stenosis position: don't want to be too close to the ends or bifurcation points
    if stenosis_position is None:
        num_centerline_points = len(radius_vector)
        threshold = 0.1*num_centerline_points
        if is_main and len(branch_points) > 1:
            possible_stenosis_positions = np.arange(int(0.1*num_centerline_points)+10,
                                    num_centerline_points - (int(0.1*num_centerline_points)+10))
            # first index of branch_points is None to signify that the main branch doesn't have a branch point, ignore it here:
            keep_inds = np.all(np.array([abs((possible_stenosis_positions-x)) > threshold for x in branch_points[1:]]), axis=0)
            possible_stenosis_positions = possible_stenosis_positions[keep_inds]
        else:
            possible_stenosis_positions = np.arange(int(0.1*num_centerline_points),
                                    num_centerline_points - (int(0.1*num_centerline_points)))

        stenosis_position = [np.random.choice(possible_stenosis_positions)]

        while len(stenosis_position) < num_stenoses:
            new_pos = np.random.choice(possible_stenosis_positions)
            keep_inds = np.array(abs((possible_stenosis_positions - new_pos)) > threshold)
            possible_stenosis_positions = possible_stenosis_positions[keep_inds]
            stenosis_position.append(new_pos)

    new_radius_vector = radius_vector.copy()

    if stenosis_length is not None:
        len_stenosis = stenosis_length
        if len(len_stenosis) < num_stenoses:
            len_stenosis = len_stenosis * num_stenoses
    else:
        #size (length of stenosis in points) must be an even number, otherwise indexing doesn't match up
        len_stenosis = [random.randint(int(0.08*num_centerline_points),int(0.12*num_centerline_points))*2 for i in range(num_stenoses)]
    for i in range(num_stenoses):
        pos = stenosis_position[i]
        if stenosis_type == "gaussian":
            mu = 0
            sigma = 0.5
            stenosis_vec = gaussian(mu, sigma, len_stenosis[i])
        # TODO
        # elif stenosis_type = "cosine":
        #     stenosis_vec = cosine_stenosis()

        scaled_vec = stenosis_vec/np.max(stenosis_vec)*stenosis_severity[i]
        new_radius_vector[pos-int(len_stenosis[i]/2):pos+int(len_stenosis[i]/2)] = new_radius_vector[pos-int(len_stenosis[i]/2):pos+int(len_stenosis[i]/2)] \
                                                             - np.multiply(scaled_vec,radius_vector[pos-int(len_stenosis[i]/2):pos+int(len_stenosis[i]/2)])
        vessel_stenosis_positions = stenosis_position
    return new_radius_vector, stenosis_severity, vessel_stenosis_positions, len_stenosis
