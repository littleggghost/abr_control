"""
***NOTE*** there are two ways to use this filter
1: wrt to timesteps
- the dmp is created during the instantiation of the class and the next step
along the path is returned by calling the `step()` function

2: wrt to time
- after instantiation, calling `generate_path_function()` interpolates the dmp
to the specified time limit. Calling the `next_timestep(t)` function at a
specified time will return the end-effector state at that point along the path
planner. This ensures that the path will reach the desired target within the
time_limit specified in `generate_path_function()`

"""
import numpy as np

try:
    import pydmps
except ImportError:
    print('\npydmps library required, see github.com/studywolf/pydmps\n')

from .path_planner import PathPlanner


class FirstOrderArc(PathPlanner):
    """
    PARAMETERS
    ----------
    n_timesteps: int, Optional (Default: 3000)
        the number of steps to break the path into
    error_scale: int, Optional (Default: 1)
        the scaling factor to apply to the error term, increasing error passed
        1 will increase the speed of motion
    """
    def __init__(self, n_timesteps=3000, error_scale=1):
        self.n_timesteps = n_timesteps
        self.error_scale = error_scale
        self.dt = 1/n_timesteps


    def generate_path(self, position, target_pos, plot=False):
        """
        Calls the step function self.n_timestep times to pregenerate
        the entire path planner

        PARAMETERS
        ----------
        position: numpy.array
            the current position of the system
        target_pos: numpy.array
            the target position
        plot: boolean, optional (Default: False)
            plot the path after generating if True
        """
        radius = np.sqrt((target_pos[0] ** 2) + (target_pos[1] ** 2))
        obj_angle = np.arctan(abs(target_pos[0]) / abs(target_pos[1])) + np.pi / 2

        # pick angle that is offset by some minimum amount
        while True:
            dep_angle = np.random.uniform() * (np.pi / 2) # + np.pi / 2
            if abs(obj_angle - dep_angle) >= 0.1 * np.pi:
                break

        dep_x = -np.cos(dep_angle) * radius
        dep_y = -np.sin(dep_angle) * radius
        dep_xyz = [dep_x, dep_y, target_pos[2]]

        arc_points = np.linspace(0, np.pi, 100)

        obj_nearest = min(arc_points, key=lambda x:abs(x-obj_angle))
        dep_nearest = min(arc_points, key=lambda x:abs(x-dep_angle))

        if obj_nearest > dep_nearest:
            arc_points = list(reversed(arc_points))

        arc_init = np.where(arc_points == obj_nearest)[0][0]
        arc_dest = np.where(arc_points == dep_nearest)[0][0]
        arc_slice = arc_points[arc_init:arc_dest+1]

        dmps_traj = np.array([-np.cos(arc_slice) * radius,
                              -np.sin(arc_slice) * radius,
                              np.ones(len(arc_slice)) * target_pos[2]])

        self.dmps = pydmps.DMPs_discrete(n_dmps=3, n_bfs=50, dt=self.dt)
        self.dmps.imitate_path(dmps_traj)

        # self.reset(target_pos=target_pos, position=position)
        #

        self.position, self.velocity, _ = self.dmps.rollout(
            timesteps=self.n_timesteps)

        # in other filter we set origin to current pos, do we need this?
        # self.position = np.array([traj + self.origin for traj in self.position])

        # reset trajectory index
        self.n = 0

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(self.position)
            plt.legend(['X', 'Y', 'Z'])
            plt.show()

        return self.position, self.velocity


    # TODO: do we want or need this function anymore?
    def reset(self, target_pos, position):
        """
        Resets the dmp path planner to a new state and target_pos

        PARAMETERS
        ----------
        target_pos: list of 3 floats
            the target_pos end-effector position in cartesian coordinates [meters]
        position: list of 3 floats
            the current end-effector cartesian position [meters]
        """
        self.origin = position
        self.dmps.reset_state()
        self.dmps.goal = target_pos - self.origin


    def _step(self, error=None):
        """
        Steps through the dmp, returning the next position and velocity along
        the path planner.
        """
        if error is None:
            error = 0
        # get the next point in the target trajectory from the dmp
        position, velocity, _ = self.dmps.step(error=error * self.error_scale)
        # add the start position offset since the dmp starts from the origin
        #position = position + self.origin

        return position, velocity
