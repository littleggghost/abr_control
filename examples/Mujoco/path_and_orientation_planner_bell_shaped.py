"""
Running operational space control using Mujoco. The controller will
move the end-effector to the target object's position and orientation.

This example controls all 6 degrees of freedom (position and orientation),
and applies a second order path planner to both position and orientation targets

After termination the script will plot results
"""
import numpy as np
import glfw
import time

from abr_control.controllers import OSC, Damping, path_planners
from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.interfaces.mujoco import Mujoco
from abr_control.utils import transformations


# initialize our robot config
robot_config = arm('jaco2')

# damp the movements of the arm
damping = Damping(robot_config, kv=10)
# create opreational space controller
ctrlr = OSC(
    robot_config,
    kp=30,  # position gain
    kv=20,
    ko=180,  # orientation gain
    null_controllers=[damping],
    vmax=None,  # [m/s, rad/s]
    # control all DOF [x, y, z, alpha, beta, gamma]
    ctrlr_dof = [True, True, True, True, True, True])

# create our interface
interface = Mujoco(robot_config, dt=.001)
interface.connect()
interface.send_target_angles(robot_config.START_ANGLES)

feedback = interface.get_feedback()
hand_xyz = robot_config.Tx('EE', feedback['q'])

def _get_approach(target_loc, approach_buffer=0.03, z_offset=0):
    """
    Takes the target location, and returns an
    orientation to approach the target, along with a target position that
    is approach_buffer meters short of the target, with a z offset
    determined by z_offset in meters. The orientation is set to be a vector
    that would connect the robot base to the target, with the z dimension
    zeroed out to keep the orientation parallel with the ground

    Parameters
    ----------
    target_loc: list of 3 floats
        xyz cartesian loc of target of interest [meters]
    approach_buffer: float, Optional (Default: 0.03)
        we want to approach the target along the orientation that connects
        the base of the arm to the target, but we want to stop short before
        going for the grasp. This variable sets that distance to stop short
        of the target [meters]
    z_offset: float, Optional (Default: 0.2)
        sometimes it is desirable to approach a target from above or below.
        This gets added to the final target position
    """
    target_z = np.copy(target_loc[2])
    target_loc = np.copy(target_loc)

    # get signs of target directions so our approach target is between the
    # arm and final target
    target_sign = target_loc / abs(target_loc)
    dist_to_target = np.linalg.norm(target_loc)
    approach_vector = target_loc
    approach_vector /= np.linalg.norm(approach_vector)

    # normalize our approach vector
    approach_pos = approach_vector * (dist_to_target - approach_buffer)
    approach_vector[2] = 0

    # world z pointing up, rotate by pi/2 to be parallel with ground
    theta1 = np.pi/2
    q1 = [np.cos(theta1/2),
          0,
          np.sin(theta1/2),
          0
          ]
    # now we rotate about z to get x pointing up
    theta2 = np.arctan2(target_loc[1], target_loc[0])
    q2 = [np.cos(theta2/2),
         0,
         0,
         np.sin(theta2/2),
         ]

    # get our combined rotation
    q3 = transformations.quaternion_multiply(q2, q1)

    return approach_pos, q3


def get_approach_path(
        robot_config, q, path_planner, max_reach_dist=1, target_position=None,
        **kwargs):
    """
    Parameters
    ----------
    feedback: dict of q and dq [rad and rad/sec]
    """
    if target_position is None:
        target_position = (np.random.random(3) - 0.5)*2 -0.2
        # do you want bad things to happen? because reaching through the floor is
        # how bad things happen
        if target_position[2] < 0.4:
            target_position[2] = 0.4

    # normalize to make sure our target is within reaching distance
    target_position = target_position / np.linalg.norm(target_position) * max_reach_dist

    # get our EE starting orientation and position
    starting_orientation = robot_config.quaternion('EE', feedback['q'])
    starting_pos = robot_config.Tx('EE', q)

    # get our target approach position and orientation
    approach_pos, approach_orient = _get_approach(target_loc=target_position, **kwargs)

    # generate our path to our approach position
    traj_planner.generate_path(
        position=starting_pos, target_pos=approach_pos)

    # generate our orientation planner
    _, orientation_planner = traj_planner.generate_orientation_path(
        orientation=starting_orientation,
        target_orientation=approach_orient)

    return traj_planner, orientation_planner, target_position, approach_orient

def instantiate_path_planner(n_timesteps=2000, error_scale=0.01, **kwargs):
    """
    Define your path planner of choice here
    """
    # pregenerate our path and orientation planners
    traj_planner = path_planners.BellShaped(
        error_scale=error_scale, n_timesteps=n_timesteps)
    return traj_planner

# set up lists for tracking data
ee_track = []
ee_angles_track = []
target_track = []
target_angles_track = []


try:
    count = 0

    print('\nSimulation starting...\n')
    while 1:
        if interface.viewer.exit:
            glfw.destroy_window(interface.viewer.window)
            break

        # get arm feedback
        feedback = interface.get_feedback()
        hand_xyz = robot_config.Tx('EE', feedback['q'])

        # get a new target and path planner
        if count % 3000 == 0:
            traj_planner = instantiate_path_planner()
            (traj_planner, orientation_planner, target_position,
                target_orientation) = get_approach_path(
                    robot_config, feedback['q'], traj_planner, max_reach_dist=0.9)
            # set our final target pos and orient object
            interface.set_mocap_xyz('target_orientation', target_position)
            interface.set_mocap_orientation('target_orientation', target_orientation)


        # get our next path step
        pos, vel = traj_planner.next()
        orient = orientation_planner.next()
        target = np.hstack([pos, orient])

        # set our filtered target object
        interface.set_mocap_xyz('path_planner_orientation', target[:3])

        # convert to quaternion, if using VREP instead of Mujoco you leave it
        # in euler angles (rxyz)
        interface.set_mocap_orientation('path_planner_orientation',
            transformations.quaternion_from_euler(
                orient[0], orient[1], orient[2], 'rxyz'))

        u = ctrlr.generate(
            q=feedback['q'],
            dq=feedback['dq'],
            target=target,
            )

        # apply the control signal, step the sim forward
        interface.send_forces(u)

        # track data
        ee_track.append(np.copy(hand_xyz))
        ee_angles_track.append(transformations.euler_from_matrix(
            robot_config.R('EE', feedback['q']), axes='rxyz'))
        target_track.append(np.copy(target[:3]))
        target_angles_track.append(np.copy(target[3:]))
        count += 1

finally:
    # stop and reset the simulation
    interface.disconnect()

    print('Simulation terminated...')

    ee_track = np.array(ee_track).T
    ee_angles_track = np.array(ee_angles_track).T
    target_track = np.array(target_track).T
    target_angles_track = np.array(target_angles_track).T

    if ee_track.shape[0] > 0:
        # plot distance from target and 3D trajectory
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611
        label_pos = ['x', 'y', 'z']
        label_or = ['a', 'b', 'g']
        c = ['r', 'g', 'b']

        fig = plt.figure(figsize=(8,12))
        ax1 = fig.add_subplot(311)
        ax1.set_ylabel('3D position (m)')
        for ii, ee in enumerate(ee_track):
            ax1.plot(ee, label='EE: %s' % label_pos[ii], c=c[ii])
            ax1.plot(target_track[ii], label='Target: %s' % label_pos[ii],
                     c=c[ii], linestyle='--')
        ax1.legend()

        ax2 = fig.add_subplot(312)
        for ii, ee in enumerate(ee_angles_track):
            ax2.plot(ee, label='EE: %s' % label_or[ii], c=c[ii])
            ax2.plot(target_angles_track[ii], label='Target: %s'%label_or[ii],
                     c=c[ii], linestyle='--')
        ax2.set_ylabel('3D orientation (rad)')
        ax2.set_xlabel('Time (s)')
        ax2.legend()

        ee_track = ee_track.T
        target_track = target_track.T
        ax3 = fig.add_subplot(313, projection='3d')
        ax3.set_title('End-Effector Trajectory')
        ax3.plot(ee_track[:, 0], ee_track[:, 1], ee_track[:, 2], label='ee_xyz')
        ax3.plot(target_track[:, 0], target_track[:, 1], target_track[:, 2],
                 label='ee_xyz', c='g', linestyle='--')
        ax3.scatter(target_track[-1, 0], target_track[-1, 1],
                    target_track[-1, 2], label='target', c='g')
        ax3.legend()
        plt.show()
