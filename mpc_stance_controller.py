"""A torque based stance controller framework."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from typing import Any, Sequence, Tuple

import numpy as np
import pybullet as p  # pytype: disable=import-error

import mpc_osqp as convex_mpc  # pytype: disable=import-error

_FORCE_DIMENSION = 3
# The QP weights in the convex MPC formulation. See the MIT paper for details:
#   https://ieeexplore.ieee.org/document/8594448/
# Intuitively, this is the weights of each state dimension when tracking a
# desired CoM trajectory. The full CoM state is represented by
# (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder).
# _MPC_WEIGHTS = (5, 5, 0.2, 0, 0, 10, 0.5, 0.5, 0.2, 0.2, 0.2, 0.1, 0)
# This worked well for in-place stepping in the real robot.
# _MPC_WEIGHTS = (5, 5, 0.2, 0, 0, 10, 0., 0., 0.2, 1., 1., 0., 0)
_MPC_WEIGHTS = (5, 5, 0.2, 0, 0, 10, 0., 0., 1., 1., 1., 0., 0)
_PLANNING_HORIZON_STEPS = 10
_PLANNING_TIMESTEP = 0.025


class TorqueStanceLegController():
  """A torque based stance leg controller framework.

  Takes in high level parameters like walking speed and turning speed, and
  generates necessary the torques for stance legs.
  """
  def __init__(
      self,
      robot: Any,
      desired_speed: Tuple[float, float] = (0, 0),
      desired_twisting_speed: float = 0,
      desired_body_height: float = 0.3,
      body_mass: float = 9.5,
      body_inertia: Tuple[float, float, float, float, float, float, float,
                          float, float] = (0.07335, 0, 0, 0, 0.25068, 0, 0, 0,
                                           0.25447),
      num_legs: int = 4,
      friction_coeffs: Sequence[float] = (0.45, 0.45, 0.45, 0.45),
      qp_solver = convex_mpc.QPOASES):
    """Initializes the class.

    Tracks the desired position/velocity of the robot by computing proper joint
    torques using MPC module.

    Args:
      robot: A robot instance.
      gait_generator: Used to query the locomotion phase and leg states.
      state_estimator: Estimate the robot states (e.g. CoM velocity).
      desired_speed: desired CoM speed in x-y plane.
      desired_twisting_speed: desired CoM rotating speed in z direction.
      desired_body_height: The standing height of the robot.
      body_mass: The total mass of the robot.
      body_inertia: The inertia matrix in the body principle frame. We assume
        the body principle coordinate frame has x-forward and z-up.
      num_legs: The number of legs used for force planning.
      friction_coeffs: The friction coeffs on the contact surfaces.
    """
    self._robot = robot
    self.desired_speed = desired_speed
    self.desired_twisting_speed = desired_twisting_speed

    self._desired_body_height = desired_body_height
    self._body_mass = body_mass
    self._num_legs = num_legs
    self._friction_coeffs = np.array(friction_coeffs)
    body_inertia_list = list(body_inertia)
    weights_list = list(_MPC_WEIGHTS)
    self.footposition = []
    self._cpp_mpc = convex_mpc.ConvexMpc(
        body_mass,
        body_inertia_list,
        self._num_legs,
        _PLANNING_HORIZON_STEPS,
        _PLANNING_TIMESTEP,
        weights_list,
        1e-5,
        qp_solver)

  def reset(self, current_time):
    del current_time

  def update(self, current_time):
    del current_time

  def get_contact_forces(self):
    """Computes the torque for stance legs."""
    self.footposition = []
    desired_com_position = np.array((0., 0., self._desired_body_height),
                                    dtype=np.float64)
    desired_com_velocity = np.array(
        (self.desired_speed[0], self.desired_speed[1], 0.), dtype=np.float64)
    desired_com_roll_pitch_yaw = np.array((0., 0., 0.), dtype=np.float64)
    desired_com_angular_velocity = np.array(
        (0., 0., self.desired_twisting_speed), dtype=np.float64)
    foot_contact_state = np.array(
        [(leg_state == 1)
         for leg_state in self._robot.foot_contact],np.int32)
    print(foot_contact_state)
    #print(self.footposition)
    # We use the body yaw aligned world frame for MPC computation.
    com_roll_pitch_yaw = np.array(self._robot.GetBaseRollPitchYaw(),
                                  dtype=np.float64)
    com_roll_pitch_yaw[2] = 0

    #predicted_contact_forces=[0]*self._num_legs*_FORCE_DIMENSION
    # print("Com Vel: {}".format(self._state_estimator.com_velocity_body_frame))
    # print("Com RPY: {}".format(self._robot.GetBaseRollPitchYawRate()))
    # print("Com RPY Rate: {}".format(self._robot.GetBaseRollPitchYawRate()))
    p.submitProfileTiming("predicted_contact_forces")
    #print(foot_contact_state)
    predicted_contact_forces = self._cpp_mpc.compute_contact_forces(
        [0],  #com_position
        np.asarray(self._robot.com_velocity_body_frame,
                   dtype=np.float64),  #com_velocity
        np.array(com_roll_pitch_yaw, dtype=np.float64),  #com_roll_pitch_yaw
        # Angular velocity in the yaw aligned world frame is actually different
        # from rpy rate. We use it here as a simple approximation.
        np.asarray(self._robot.GetBaseRollPitchYawRate(),
                   dtype=np.float64),  #com_angular_velocity
        foot_contact_state,  #foot_contact_states
        np.array(self._robot.GetFootPositionsInBaseFrame().flatten(),
                 dtype=np.float64),  #foot_positions_base_frame
        self._friction_coeffs,  #foot_friction_coeffs
        desired_com_position,  #desired_com_position
        desired_com_velocity,  #desired_com_velocity
        desired_com_roll_pitch_yaw,  #desired_com_roll_pitch_yaw
        desired_com_angular_velocity  #desired_com_angular_velocity
    )
    p.submitProfileTiming()

    contact_forces = {}
    for i in range(self._num_legs):
      contact_forces[i] = np.array(
          predicted_contact_forces[i * _FORCE_DIMENSION:(i + 1) *  #  _FORCE_DIMENSION = 3 -> 3*i : 3*(i+1)
                                   _FORCE_DIMENSION])

    self._robot.contact_forces = contact_forces

    return contact_forces
