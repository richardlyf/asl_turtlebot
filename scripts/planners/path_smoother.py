import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    path = np.array(path)
    t = [0]
    t_max = 0
    t_segments = np.linalg.norm(path[1:] - path[:-1], axis=1) / V_des
    for t_segment in t_segments:
        t_max += t_segment
        t.append(t_max)
    t = np.array(t)

    t_smoothed = np.arange(0, t_max, dt)
    tck_x = scipy.interpolate.splrep(t, path[:, 0], s=alpha)
    tck_y = scipy.interpolate.splrep(t, path[:, 1], s=alpha)
    x_smoothed = scipy.interpolate.splev(t_smoothed, tck_x)
    y_smoothed = scipy.interpolate.splev(t_smoothed, tck_y)
    xd_smoothed = scipy.interpolate.splev(t_smoothed, tck_x, der=1)
    yd_smoothed = scipy.interpolate.splev(t_smoothed, tck_y, der=1)
    xdd_smoothed = scipy.interpolate.splev(t_smoothed, tck_x, der=2)
    ydd_smoothed = scipy.interpolate.splev(t_smoothed, tck_y, der=2)
    theta_smoothed = np.arctan2(yd_smoothed, xd_smoothed)
    traj_smoothed = np.vstack((x_smoothed, 
                                y_smoothed,
                                theta_smoothed,
                                xd_smoothed, 
                                yd_smoothed,
                                xdd_smoothed,
                                ydd_smoothed)).T
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed
