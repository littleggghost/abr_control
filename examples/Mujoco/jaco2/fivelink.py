"""
Example of working through an arm model one link at a time
"""

import glfw
import sys
import numpy as np

import mujoco_py as mjp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from abr_control.controllers import OSC, Damping, RestingConfig
from abr_control.interfaces.mujoco import Mujoco
from abr_control.arms.mujoco_config import MujocoConfig


if len(sys.argv) > 1:
    show_plot = sys.argv[1]
else:
    show_plot = False
# NOTE: the meshes file from MuJoCo_Unity_UR5/xmls needs to be in the
# abr_control/arms/jaco2/meshes folder for this to load properly
model_filename = 'jaco2_fivelink'
q_init = np.array([0, 0, 0, 0, 0])

robot_config = MujocoConfig(model_filename)

dt = 0.005
interface = Mujoco(robot_config, dt=dt)
interface.connect()

ctrlr = OSC(robot_config, kp=30, kv=20,
            ctrlr_dof=[True, True, True, False, False, False])

interface.send_target_angles(q_init)

target = np.array([-0.2, 0.2, 0.4, 0, 0, 0])
interface.set_mocap_xyz('target', target[:3])

interface.set_mocap_xyz('hand', np.array([.2, .4, 1]))

q_track = []
u_track = []

count = 0
link_name = 'EE'
try:
    while True:

        if interface.viewer.exit:
            glfw.destroy_window(interface.viewer.window)
            break

        feedback = interface.get_feedback()
        hand_xyz = robot_config.Tx(link_name)
        u = ctrlr.generate(
            q=feedback['q'],
            dq=feedback['dq'],
            target=target,
            ref_frame=link_name,
            )
        interface.send_forces(u)

        if count %500 == 0:
            print('ee_xyz: ', hand_xyz)
            print('target', target)
            print('u: ', u)
            print('q: ', feedback['q'])

        if np.linalg.norm(hand_xyz[:3] - target[:3]) < 0.01:
            target[0] = np.random.uniform(0.2, 0.3) * np.sign(np.random.uniform(-1, 1))
            target[1] = np.random.uniform(0.2, 0.25) * np.sign(np.random.uniform(-1, 1))
            target[2] = np.random.uniform(0.4, 0.5)
            interface.set_mocap_xyz('target', target[:3])

        q_track.append(feedback['q'])
        u_track.append(u)
        count += 1

finally:
    interface.disconnect()

    q_track = np.asarray(q_track)

    if show_plot:
        plt.figure(figsize=(30, 30))

        plt.subplot(211)
        plt.plot(q_track)
        plt.ylabel('Joint Angles [rad]')
        plt.legend(['0', '1'])

        plt.subplot(212)
        plt.plot(u_track)
        plt.ylabel('Joint Force [Nm]')
        plt.legend(['0', '1'])

        plt.show()