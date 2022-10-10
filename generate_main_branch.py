from tube_functions import *


def generate_main_branch(vessel_type, length, n_points, **kwargs):
    if vessel_type == 'cylinder':
        main_C, main_dC = cylinder(length, n_points)
    elif vessel_type == 'spline':
        main_C, main_dC = random_spline(length, kwargs.order, np.random.randint(kwargs.order + 1, 10), n_points)
    elif vessel_type == 'RCA':
        RCA_control_points = np.load(os.path.join(kwargs["control_point_path"], "RCA_ctrl_points.npy")) / 1000 # [m] instead of [mm]
        mean_ctrl_pts = np.mean(RCA_control_points, axis=0)
        stdev_ctrl_pts = np.std(RCA_control_points, axis=0)
        main_C, main_dC = RCA_vessel_curve(
            n_points,
            mean_ctrl_pts,
            stdev_ctrl_pts,
            length,
            kwargs["rng"],
            shear=kwargs["shear"],
            warp=kwargs["warp"]
        )
    else:
        raise ValueError(f"Tree generation for {vessel_type} vessel type is not available.")
    
    return main_C, main_dC