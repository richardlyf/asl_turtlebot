import numpy as np
import scipy.linalg  # You may find scipy.linalg.block_diag useful
import scipy.stats  # You may find scipy.stats.multivariate_normal.pdf useful
import turtlebot_model as tb

EPSILON_OMEGA = 1e-3

class ParticleFilter(object):
    """
    Base class for Monte Carlo localization and FastSLAM.

    Usage:
        pf = ParticleFilter(x0, R)
        while True:
            pf.transition_update(u, dt)
            pf.measurement_update(z, Q)
            localized_state = pf.x
    """

    def __init__(self, x0, R):
        """
        ParticleFilter constructor.

        Inputs:
            x0: np.array[M,3] - initial particle states.
             R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
        """
        self.M = x0.shape[0]  # Number of particles
        self.xs = x0  # Particle set [M x 3]
        self.ws = np.repeat(1. / self.M, self.M)  # Particle weights (initialize to uniform) [M]
        self.R = R  # Control noise covariance (corresponding to dt = 1 second) [2 x 2]

    @property
    def x(self):
        """
        Returns the particle with the maximum weight for visualization.

        Output:
            x: np.array[3,] - particle with the maximum weight.
        """
        idx = self.ws == self.ws.max()
        x = np.zeros(self.xs.shape[1:])
        x[:2] = self.xs[idx,:2].mean(axis=0)
        th = self.xs[idx,2]
        x[2] = np.arctan2(np.sin(th).mean(), np.cos(th).mean())
        return x

    def transition_update(self, u, dt):
        """
        Performs the transition update step by updating self.xs.

        Inputs:
            u: np.array[2,] - zero-order hold control input.
            dt: float        - duration of discrete time step.
        Output:
            None - internal belief state (self.xs) should be updated.
        """
        ########## Code starts here ##########
        # TODO: Update self.xs.
        # Hint: Call self.transition_model().
        # Hint: You may find np.random.multivariate_normal useful.
        us = np.random.multivariate_normal(u, self.R * dt, (self.M,))
        self.xs = self.transition_model(us, dt)
        ########## Code ends here ##########

    def transition_model(self, us, dt):
        """
        Propagates exact (nonlinear) state dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """
        raise NotImplementedError("transition_model must be overridden by a subclass of EKF")

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[2,I]   - matrix of I columns containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Output:
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        raise NotImplementedError("measurement_update must be overridden by a subclass of EKF")

    def resample(self, xs, ws):
        """
        Resamples the particles according to the updated particle weights.

        Inputs:
            xs: np.array[M,3] - matrix of particle states.
            ws: np.array[M,]  - particle weights.

        Output:
            None - internal belief state (self.xs, self.ws) should be updated.
        """
        r = np.random.rand() / self.M

        ########## Code starts here ##########
        # TODO: Update self.xs, self.ws.
        # Note: Assign the weights in self.ws to the corresponding weights in ws
        #       when resampling xs instead of resetting them to a uniform
        #       distribution. This allows us to keep track of the most likely
        #       particle and use it to visualize the robot's pose with self.x.
        # Hint: To maximize speed, try to implement the resampling algorithm
        #       without for loops. You may find np.linspace(), np.cumsum(), and
        #       np.searchsorted() useful. This results in a ~10x speedup.
        total_weights = np.sum(ws)
        u = (np.arange(self.M * 1.) / self.M + r) * total_weights
        c = np.cumsum(ws)
        idx = np.searchsorted(c, u)
        self.xs = xs[idx]
        self.ws = ws[idx]
        ########## Code ends here ##########

    def measurement_model(self, z_raw, Q_raw):
        """
        Converts raw measurements into the relevant Gaussian form (e.g., a
        dimensionality reduction).

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[2I,]   - joint measurement mean.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        raise NotImplementedError("measurement_model must be overridden by a subclass of EKF")


class MonteCarloLocalization(ParticleFilter):

    def __init__(self, x0, R, map_lines, tf_base_to_camera, g):
        """
        MonteCarloLocalization constructor.

        Inputs:
                       x0: np.array[M,3] - initial particle states.
                        R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
                map_lines: np.array[2,J] - J map lines in columns representing (alpha, r).
        tf_base_to_camera: np.array[3,]  - (x, y, theta) transform from the
                                           robot base to camera frame.
                        g: float         - validation gate.
        """
        self.map_lines = map_lines  # Matrix of J map lines with (alpha, r) as columns
        self.tf_base_to_camera = tf_base_to_camera  # (x, y, theta) transform
        self.g = g  # Validation gate
        super(self.__class__, self).__init__(x0, R)

    def transition_model(self, us, dt):
        """
        Unicycle model dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """

        ########## Code starts here ##########
        # TODO: Compute g.
        # Hint: We don't need Jacobians for particle filtering.
        # Hint: A simple solution can be using a for loop for each partical
        #       and a call to tb.compute_dynamics
        # Hint: To maximize speed, try to compute the dynamics without looping
        #       over the particles. If you do this, you should implement
        #       vectorized versions of the dynamics computations directly here
        #       (instead of modifying turtlebot_model). This results in a
        #       ~10x speedup.
        # Hint: This faster/better solution does not use loop and does 
        #       not call tb.compute_dynamics. You need to compute the idxs
        #       where abs(om) > EPSILON_OMEGA and the other idxs, then do separate 
        #       updates for them
        g = np.zeros((self.M, 3))
        x, y, th = self.xs[:, 0], self.xs[:, 1], self.xs[:, 2]
        V, om = us[:, 0], us[:, 1]
        th_new = th + om * dt
        g[:, 2] = th_new

        idx_less = np.where(np.abs(om) < EPSILON_OMEGA)[0]
        idx_more = np.where(np.abs(om) >= EPSILON_OMEGA)[0]
        # When om < threshold
        g[idx_less, 0] = x[idx_less] + V[idx_less] * np.cos(th[idx_less]) * dt
        g[idx_less, 1] = y[idx_less] + V[idx_less] * np.sin(th[idx_less]) * dt
        # When om >= threshold
        g[idx_more, 0] = x[idx_more] + V[idx_more] * (np.sin(th_new[idx_more]) - np.sin(th[idx_more])) / om[idx_more]
        g[idx_more, 1] = y[idx_more] + V[idx_more] * (-np.cos(th_new[idx_more]) + np.cos(th[idx_more])) / om[idx_more]
        ########## Code ends here ##########

        return g

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[2,I]   - matrix of I columns containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Output:
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        xs = np.copy(self.xs)
        ws = np.zeros_like(self.ws)

        ########## Code starts here ##########
        # TODO: Compute new particles (xs, ws) with updated measurement weights.
        # Hint: To maximize speed, implement this without looping over the
        #       particles. You may find scipy.stats.multivariate_normal.pdf()
        #       useful.
        # Hint: You'll need to call self.measurement_model()
        vs, Q = self.measurement_model(z_raw, Q_raw)
        # 0 mean gaussian. When difference between measurements is near 0, has high weight value
        ws = scipy.stats.multivariate_normal.pdf(vs, cov=Q, allow_singular=True)
        ########## Code ends here ##########

        self.resample(xs, ws)

    def measurement_model(self, z_raw, Q_raw):
        """
        Assemble one joint measurement and covariance from the individual values
        corresponding to each matched line feature for each particle.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[M,2I]  - joint measurement mean for M particles.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        vs = self.compute_innovations(z_raw, np.array(Q_raw))

        ########## Code starts here ##########
        # TODO: Compute Q.
        # Hint: You might find scipy.linalg.block_diag() useful
        I = vs.shape[1] / 2
        Q = scipy.linalg.block_diag(*Q_raw)
        ########## Code ends here ##########

        return vs, Q

    def compute_innovations(self, z_raw, Q_raw):
        """
        Given lines extracted from the scanner data, tries to associate each one
        to the closest map entry measured by Mahalanobis distance.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: np.array[I,2,2] - I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            vs: np.array[M,2I] - M innovation vectors of size 2I
                                 (predicted map measurement - scanner measurement).
        """
        def angle_diff(a, b):
            a = a % (2. * np.pi)
            b = b % (2. * np.pi)
            diff = a - b
            if np.size(diff) == 1:
                if np.abs(a - b) > np.pi:
                    sign = 2. * (diff < 0.) - 1.
                    diff += sign * 2. * np.pi
            else:
                idx = np.abs(diff) > np.pi
                sign = 2. * (diff[idx] < 0.) - 1.
                diff[idx] += sign * 2. * np.pi
            return diff

        ########## Code starts here ##########
        # TODO: Compute vs (with shape [M x I x 2]).
        # Hint: Simple solutions: Using for loop, for each particle, for each 
        #       observed line, find the most likely map entry (the entry with 
        #       least Mahalanobis distance).
        # Hint: To maximize speed, try to eliminate all for loops, or at least
        #       for loops over J. It is possible to solve multiple systems with
        #       np.linalg.solve() and swap arbitrary axes with np.transpose().
        #       Eliminating loops over J results in a ~10x speedup.
        #       Eliminating loops over I results in a ~2x speedup.
        #       Eliminating loops over M results in a ~5x speedup.
        #       Overall, that's 100x!
        # Hint: For the faster solution, you might find np.expand_dims(), 
        #       np.linalg.solve(), np.meshgrid() useful.
        I = z_raw.shape[1]
        J = self.map_lines.shape[1]
        # (M, 2, J)
        hs = self.compute_predicted_measurements()
        # (M, 2, I)
        zs = np.repeat(z_raw[np.newaxis, ...], self.M, axis=0)
        # (M, 2, I, J)
        hs_expanded = np.repeat(hs[..., np.newaxis, :], I, axis=2)
        zs_expanded = np.repeat(zs[..., np.newaxis], J, axis=3)
        # (M, I, J)
        alpha_diff = angle_diff(zs_expanded[:, 0], hs_expanded[:, 0])
        r_diff = zs_expanded[:, 1] - hs_expanded[:, 1]
        # (M, I, J, 2)
        vs = np.stack((alpha_diff, r_diff), axis=3)
        # (I, J, 2, 2)
        Q = np.repeat(Q_raw[:, np.newaxis, ...], J, axis=1)
        # (M, I, J, 2, 2)
        Q = np.repeat(Q[np.newaxis, ...], self.M, axis=0)
        # (M, I, J)
        ds = np.squeeze(np.matmul(vs[..., np.newaxis, :], np.matmul(np.linalg.inv(Q), vs[..., np.newaxis])), axis=(3, 4))
        d_min_idx = np.argmin(ds, axis=2)
        # (M, I, 2)
        vs = np.squeeze(np.take_along_axis(vs, d_min_idx[..., np.newaxis, np.newaxis], axis=2), axis=2)
        ########## Code ends here ##########

        # Reshape [M x I x 2] array to [M x 2I]
        return vs.reshape((self.M,-1))  # [M x 2I]

    def compute_predicted_measurements(self):
        """
        Given a single map line in the world frame, outputs the line parameters
        in the scanner frame so it can be associated with the lines extracted
        from the scanner measurements.

        Input:
            None
        Output:
            hs: np.array[M,2,J] - J line parameters in the scanner (camera) frame for M particles.
        """
        ########## Code starts here ##########
        # TODO: Compute hs.
        # Hint: We don't need Jacobians for particle filtering.
        # Hint: Simple solutions: Using for loop, for each particle, for each 
        #       map line, transform to scanner frmae using tb.transform_line_to_scanner_frame()
        #       and tb.normalize_line_parameters()
        # Hint: To maximize speed, try to compute the predicted measurements
        #       without looping over the map lines. You can implement vectorized
        #       versions of turtlebot_model functions directly here. This
        #       results in a ~10x speedup.
        # Hint: For the faster solution, it does not call tb.transform_line_to_scanner_frame()
        #       or tb.normalize_line_parameters(), but reimplement these steps vectorized.
        def transform_line_to_scanner_frame(lines, xs, tf_base_to_camera):
            """
            Given map lines in the world frame, outputs the line parameters
            in the scanner frame so it can be associated with the lines extracted
            from the scanner measurements.

            Input:
                lines: np.array[2,J] - map line (alpha, r) in world frame.
                xs: np.array[M,3] - pose of base (x, y, theta) in world frame.
                tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
            Outputs:
                hs: np.array[M,2,J]  - line parameters in the scanner (camera) frame.
            """
            J = lines.shape[1]
            alpha, r = lines
            x_base, y_base, th_base = xs[:, 0], xs[:, 1], xs[:, 2]
            # Rotation matrix (2, 2, M)
            Rz = np.array([
                [np.cos(th_base), -np.sin(th_base)],
                [np.sin(th_base), np.cos(th_base)]
            ])
            # (M, 2, 2)
            Rz = np.transpose(Rz, (2, 0, 1))
            # (M, 2)
            cam_rotated = np.dot(Rz, tf_base_to_camera[:2])
            # (M,)
            x_cam_rotated, y_cam_rotated = cam_rotated[:, 0], cam_rotated[:, 1]
            x_cam = x_cam_rotated + x_base
            y_cam = y_cam_rotated + y_base
            th_cam = th_base + tf_base_to_camera[2]

            # Get alpha and r in camera frame
            # (M, J)
            alpha_in_cam = np.repeat(alpha[np.newaxis, :], self.M, axis=0) - np.repeat(th_cam[:, np.newaxis], J, axis=1)
            # (M,)
            cam_coord_norm = np.linalg.norm(np.array([x_cam, y_cam]), axis=0)
            # (M, J)
            projection_vector = np.cos(np.repeat(np.arctan2(y_cam, x_cam)[:, np.newaxis], J, axis=1) - np.repeat(alpha[np.newaxis, :], self.M, axis=0))
            r_projected = projection_vector * cam_coord_norm.reshape(-1, 1)
            r_in_cam = r - r_projected
            # (M, 2, J)
            hs = np.stack((alpha_in_cam, r_in_cam), axis=1)

            return hs

        def normalize_line_parameters(hs):
            """
            Ensures that r is positive and alpha is in the range [-pi, pi].

            Inputs:
                 hs: np.array[M,2,J]  - line parameters (alpha, r).
            Outputs:
                 hs: np.array[M,2,J]  - normalized parameters.
            """
            alpha, r = hs[:, 0], hs[:, 1]
            neg_r_idx = np.where(r < 0)
            alpha[neg_r_idx] += np.pi
            r[neg_r_idx] *= -1
            alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
            hs = np.stack((alpha, r), axis=1)

            return hs

        hs = transform_line_to_scanner_frame(self.map_lines, self.xs, self.tf_base_to_camera)
        hs = normalize_line_parameters(hs)
        ########## Code ends here ##########

        return hs

