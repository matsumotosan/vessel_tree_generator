import random
import numpy as np

from src.tube_functions import stenosis_generator


def get_vessel_surface(
    curve,
    derivatives,
    branch_points,
    num_centerline_points,
    num_circle_points,
    radius,
    num_stenoses=0,
    is_main_branch=True,
    constant_radius=True,
    stenosis_severity=None,
    stenosis_position=None,
    stenosis_length=None,
    stenosis_type="gaussian",
    return_surface=False
):
    '''
    Generates a tubular surface with specified radius around any arbitrary centerline curve
    :param curve: Nx3 array of 3D points in centerline curve
    :param derivatives: (N-1)x3 array of centerline curve derivatives
    :param branch_points: indices where a branch connects to the main branch, to avoid stenosis in the same location
    :param num_centerline_points: N
    :param num_circle_points: number of radial points on each contour
    :param radius: single number (max radius) or Nx1 vector (radius at each centerline point)
    :param num_stenoses: number of stenoses in the vessel, typically 0-3
    :param is_main_branch: bool: whether vessel is main vessel or side branch
    :param constant_radius: bool: constant radius or tapered
    :param stenosis_severity: percent diameter reduction. If not specified, will be randomly sampled
    :param: stenosis_position: index of centerline matrix where stenosis is centered.
            If not specified, will be randomly sampled
    :param: stenosis_length: number of points that make up stenosis. If not specified, will be randomly sampled
    :param: stenosis_type: type of profile for stenosis geometry. Currently only "gaussian" is implemented
    :param: return_surface: bool: if True, will return list of points making up 3D vessel surface
    :return: stenosis parameters, optional: X,Y,Z surface points of vessel surface
    '''
    # based on https://www.mathworks.com/matlabcentral/fileexchange/5562-tubeplot and
    # https://www.mathworks.com/matlabcentral/fileexchange/25086-extrude-a-ribbon-tube-and-fly-through-it
    if len(radius) == 1 and constant_radius:
        r = np.tile(radius, num_centerline_points)
    elif len(radius) == 1 and not constant_radius:
        # added small gaussian noise so that the diameters aren't perfectly linear
        if is_main_branch:
            taper = random.uniform(0.3,0.4) #network learns and forces the absolute decrease in radius on new data if this value is constant
        else:
            taper = random.uniform(0.5,0.7)
        r = np.flip(np.multiply(np.tile(radius, num_centerline_points), np.linspace(taper, 1, num_centerline_points))+np.array([random.gauss(0,0.00001) for i in range(num_centerline_points)]))
    else:
        r = radius #vector containing user-specified radii along centerline

    # create stenoses
    new_r = r.copy()
    percent_stenosis = None
    stenosis_pos = None
    num_stenosis_points = 0
    if num_stenoses > 0:
        new_r, percent_stenosis, stenosis_pos, num_stenosis_points = stenosis_generator(num_stenoses, r, branch_points,
                                                                                        is_main=is_main_branch,
                                                                                        stenosis_severity=stenosis_severity,
                                                                                        stenosis_position=stenosis_position,
                                                                                        stenosis_length=stenosis_length,
                                                                                        stenosis_type=stenosis_type)

    if not return_surface:
        return new_r, percent_stenosis, stenosis_pos, num_stenosis_points

    t = np.linspace(0,2*np.pi, num_circle_points)
    C = curve
    dC = derivatives

    keep_inds = np.squeeze(np.argwhere(np.sum(abs(dC),1) != 0))
    dC = dC[keep_inds]
    C = C[keep_inds]

    normal_vector = np.zeros((3))
    idx = np.argmin(np.abs(C[1,:]))
    normal_vector[idx] = 1

    surface = []

    cfact = np.tile(np.cos(t), (3,1))
    sfact = np.tile(np.sin(t), (3,1))
    radial_threshold = int(0.3*num_circle_points)

    for k in range(C.shape[0]):
        convec = np.cross(normal_vector, dC[k,:])
        convec = convec/np.linalg.norm(convec)
        normal_vector = np.cross(dC[k,:], convec)
        normal_vector = normal_vector/np.linalg.norm(normal_vector)

        # add endcaps to vessel surface for projections
        if k == 0:
            surface_r = np.linspace(0,new_r[k], 50)[1:]
            surface_r_hat = np.linspace(0,r[k],50)[1:]

        elif k==C.shape[0]-1:
            surface_r = np.flip(np.linspace(0, new_r[k], 50)[1:])
            surface_r_hat = np.flip(np.linspace(0, r[k], 50)[1:])
        else:
            surface_r = [new_r[k]]
            surface_r_hat = [r[k]]

        for R, R_hat in zip(surface_r, surface_r_hat):
            points = np.tile(C[k,:], (num_circle_points,1)) + np.multiply(cfact.T, np.tile(R*normal_vector, (num_circle_points,1))) \
                                + np.multiply(sfact.T, np.tile(R*convec, (num_circle_points,1)))
            surface.append(points)

    surface = np.array(surface)

    return surface, new_r, percent_stenosis, stenosis_pos, num_stenosis_points