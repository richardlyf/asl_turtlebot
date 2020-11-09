import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    x, y, th = xvec
    V, om = u
    th_new = th + om * dt

    if np.abs(om) < EPSILON_OMEGA:
        x_new = x + V * np.cos(th) * dt
        y_new = y + V * np.sin(th) * dt
        Gx = np.array([
            [1, 0, -V * np.sin(th) * dt],
            [0, 1, V * np.cos(th) * dt],
            [0, 0, 1],
        ])
        Gu = np.array([
            [np.cos(th) * dt, -V * np.sin(th) * dt ** 2 / 2],
            [np.sin(th) * dt, V * np.cos(th) * dt ** 2 / 2],
            [0, dt],
        ])

    else:
        x_new = x + V * (np.sin(th_new) - np.sin(th)) / om
        y_new = y + V * (-np.cos(th_new) + np.cos(th)) / om
        Gx = np.array([
            [1, 0, V * (np.cos(th_new) - np.cos(th)) / om],
            [0, 1, V * (np.sin(th_new) - np.sin(th)) / om],
            [0, 0, 1],
        ])
        Gu = np.array([
            [(np.sin(th_new) - np.sin(th)) / om, V * (-om ** -2 * np.sin(th_new) + om ** -1 * np.cos(th_new) * dt + om ** -2 * np.sin(th))],
            [(-np.cos(th_new) + np.cos(th)) / om, V * (om ** -2 * np.cos(th_new) + om ** -1 * np.sin(th_new) * dt - om ** -2 * np.cos(th))],
            [0, dt],
        ])

    g = np.array([x_new, y_new, th_new])    
    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)
    x_base, y_base, th_base = x
    # Get camera in world frame
    Rz = np.array([
        [np.cos(th_base), -np.sin(th_base)],
        [np.sin(th_base), np.cos(th_base)]
    ])
    x_cam_rotated, y_cam_rotated = np.dot(Rz, tf_base_to_camera[:2])
    x_cam = x_cam_rotated + x_base
    y_cam = y_cam_rotated + y_base
    th_cam = th_base + tf_base_to_camera[2]

    # Get alpha and r in camera frame
    alpha_in_cam = alpha - th_cam
    cam_coord_norm = np.linalg.norm(np.array([x_cam, y_cam]))
    projection_vector = np.cos(np.arctan2(y_cam, x_cam) - alpha)
    r_projected = projection_vector * cam_coord_norm
    r_in_cam = r - r_projected
    h = np.array([alpha_in_cam, r_in_cam])

    # Compute derivatives of all parts of the chain rule
    dcos = np.sin(np.arctan2(y_cam, x_cam) - alpha)
    datan = 1.0 / (1 + (y_cam / x_cam) ** 2)
    dfrac_x = -y_cam / x_cam ** 2
    dfrac_y = 1.0 / x_cam
    x_bc, y_bc = tf_base_to_camera[:2]
    dx_cam = -x_bc * np.sin(th_base) - y_bc * np.cos(th_base)
    dy_cam = x_bc * np.cos(th_base) - y_bc * np.sin(th_base)
    dfrac_th = (dy_cam * x_cam - y_cam * dx_cam) / x_cam ** 2

    Hx = np.array([
        [0, 0, -1],
        [dcos * datan * dfrac_x * cam_coord_norm - projection_vector / cam_coord_norm * x_cam,
        dcos * datan * dfrac_y * cam_coord_norm - projection_vector / cam_coord_norm * y_cam,
        dcos * datan * dfrac_th * cam_coord_norm - projection_vector / cam_coord_norm * (x_cam * dx_cam + y_cam * dy_cam)
        ]
    ])
    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
