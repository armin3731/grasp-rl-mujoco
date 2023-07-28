import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from utils import QT, mujoco_reset_env, finger_bend

import torch


# * Settings ========================================================
LOAD_FOLDER = "models/"  # the folder to load the models
XML_PATH = "RoboticHand.xml"  # xml path
TIME_TO_HOLD = 1.5  # the maximum time that hand should keep the object
NUMBER_OF_SIMULATIONS = 5  # it will simulate the grasp, 5 times

N_ACTIONS = 2  # The number of actions for robotic hand
N_OBSERVATIONS = 6  # The the number of state observations [TIP_POSITION[x y z], END_POSITION[[x y z]]]

# * Defining Finger QT Models =============================================
# if GPU is to be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the Networks for each finger
index_QT = QT(N_OBSERVATIONS, N_ACTIONS, name="index_finger").to(device)
middle_QT = QT(N_OBSERVATIONS, N_ACTIONS, name="middle_finger").to(device)
ring_QT = QT(N_OBSERVATIONS, N_ACTIONS, name="ring_finger").to(device)
pinky_QT = QT(N_OBSERVATIONS, N_ACTIONS, name="pinky_finger").to(device)
thumb_QT = QT(N_OBSERVATIONS, N_ACTIONS, name="thumb_finger").to(device)

# load Models
index_QT.load_parameters(os.path.join(LOAD_FOLDER, "%s.pt" % (index_QT.name)))
middle_QT.load_parameters(os.path.join(LOAD_FOLDER, "%s.pt" % (middle_QT.name)))
ring_QT.load_parameters(os.path.join(LOAD_FOLDER, "%s.pt" % (ring_QT.name)))
pinky_QT.load_parameters(os.path.join(LOAD_FOLDER, "%s.pt" % (pinky_QT.name)))
thumb_QT.load_parameters(os.path.join(LOAD_FOLDER, "%s.pt" % (thumb_QT.name)))


# * Simulation Default Functionality ================================
# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0
bendForce = 29.5


def controller(
    mj_model,
    mj_data,
    actions,
):
    # put the controller here
    for each_finger_idx in range(5):
        mj_model, mj_data = finger_bend(
            each_finger_idx, actions[each_finger_idx], mj_model, mj_data
        )


def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(mj_model, mj_data)
        mj.mj_forward(mj_model, mj_data)


def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    button_middle = (
        glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    )
    button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

    # update mouse position
    glfw.get_cursor_pos(window)


def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(mj_model, action, dx / height, dy / height, scene, cam)


def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(mj_model, action, 0.0, -0.05 * yoffset, scene, cam)


# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname, XML_PATH)
xml_path = abspath

# MuJoCo data structures
mj_model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
mj_data = mj.MjData(mj_model)  # MuJoCo data
cam = mj.MjvCamera()  # Abstract camera
opt = mj.MjvOption()  # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Robotic Hand", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(mj_model, maxgeom=10000)
context = mj.MjrContext(mj_model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Camera settings
cam.azimuth = -115.3410740203193
cam.distance = 5.4363140749365115
cam.elevation = -32.198838896952104
cam.lookat = np.array([0.64240141, -0.27601188, 0.45069815])

print("Total number of DoFs in the model:", mj_model.nv)


for sim_num in range(NUMBER_OF_SIMULATIONS):
    mj_model, mj_data = mujoco_reset_env(mj_model, mj_data)  # reset mujoco env
    holding_object = 0  # reset hold_object reward
    mj.mj_forward(mj_model, mj_data)
    tip_location = mj_data.site_xpos[0]  # Object's tip location
    end_location = mj_data.site_xpos[1]  # Object's end location

    # Action from policy_net
    current_state = np.double(np.append(tip_location, end_location))
    # Select actions
    index_action = index_QT.predict(current_state)
    middle_action = middle_QT.predict(current_state)
    ring_action = ring_QT.predict(current_state)
    pinky_action = pinky_QT.predict(current_state)
    thumb_action = thumb_QT.predict(current_state)
    actions = [index_action, middle_action, ring_action, pinky_action, thumb_action]

    # set the controller
    mj.set_mjcb_control(controller(mj_model, mj_data, actions))
    while not glfw.window_should_close(window):
        sim_start = mj_data.time
        while mj_data.time - sim_start < 1.0 / 24.0:
            mj.mj_step(mj_model, mj_data)

        # Stop conditions
        if mj_data.time >= TIME_TO_HOLD:
            holding_object = 1
            break
        if tip_location[2] <= 0 or end_location[2] <= 0:
            holding_object = 0
            break
        # get framebuffer viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        # Update scene and render
        mj.mjv_updateScene(
            mj_model, mj_data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene
        )
        mj.mjr_render(viewport, scene, context)
        # swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(window)
        # process pending GUI events, call GLFW callbacks
        glfw.poll_events()
    print(
        "Simulation %i of %i : Reward = %i"
        % (sim_num + 1, NUMBER_OF_SIMULATIONS, holding_object)
    )

glfw.terminate()
