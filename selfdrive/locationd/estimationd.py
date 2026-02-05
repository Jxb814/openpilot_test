#!/usr/bin/env python3
import os
import math
import numpy as np
import capnp

import cereal.messaging as messaging
from cereal import car, custom
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process
from opendbc.car.vehicle_model import VehicleModel
from openpilot.selfdrive.locationd.helpers import PoseCalibrator, Pose

MIN_SPEED = 10.0  # m/s
INIT_P = 100 * np.eye(2)
INIT_THETA = np.array([[2.], [0.]]) #np.zeros((2, 1))
LAMBDA = 1.0
MAX_STEER_RATE = 1.0
MAX_LONG_ACCEL = 0.1
ACC_G = 9.81  # m/s^2


def rls(yt, ut, P, theta):
  # Form regressor vector
  H = np.array([[ut], [1.0]])
  # Prediction error
  e = yt - (H.T @ theta)[0, 0]
  # Gain vector
  K = P @ H / (LAMBDA + (H.T @ P @ H)[0, 0])
  # Update parameter estimates
  theta = theta + K * e
  # Update covariance
  P = (P - K @ H.T @ P) / LAMBDA
  return theta, P


class understeerGradientEstimator:
  def __init__(self, CP: car.CarParams):
    self.CP = CP
    self.VM = VehicleModel(self.CP)
    self.calibrator = PoseCalibrator()
    self.lat_accel_from_curv = 0.0
    self.adjusted_steering_angle = 0.0
    self.v_ego = 0.0
    self.SR = self.CP.steerRatio
    self.mass = self.CP.mass
    self.wheelbase = self.CP.wheelbase
    self.steering_angle_deg = 0.0
    self.angleOffsetDeg = 0.0
    self.roll = 0.0
    self.yaw_rate = 0.0
    self.yaw_rate_std = 0.0
    self.pose_valid = False
    self.unestimated = True
    self.reset()

  def reset(self):
    self.P = INIT_P.copy()
    self.theta = INIT_THETA.copy()

  def handle_log(self, t: float, which: str, msg: capnp._DynamicStructReader):
    if which == "carState":
      # self.yaw_rate = msg.yawRate
      self.v_ego = msg.vEgo
      self.steering_angle_deg = msg.steeringAngleDeg
    elif which == "liveParameters":
      self.angleOffsetDeg =msg.angleOffsetDeg # from kalman filter
      self.roll = msg.roll
    elif which == "liveCalibration":
      self.calibrator.feed_live_calib(msg)
    elif which == "livePose":
      device_pose = Pose.from_live_pose(msg)
      calibrated_pose = self.calibrator.build_calibrated_pose(device_pose)
      self.yaw_rate = calibrated_pose.angular_velocity.yaw
      self.yaw_rate_std = calibrated_pose.angular_velocity.yaw_std
      self.pose_valid = msg.angularVelocityDevice.valid and msg.posenetOK and msg.inputsOK
    self.t = t

  def get_msg(self, valid: bool, debug: bool = False) -> capnp._DynamicStructBuilder:
    msg = messaging.new_message('paramEst')
    msg.valid = valid
    paramEst = msg.paramEst
    paramEst.usgEst = float(self.theta[0])
    paramEst.angleOffsetEst = float(self.theta[1])
    paramEst.covarianceP = self.P.flatten().tolist()   # np.array(covarianceP).reshape(2,2) would give back the 2 by 2 P matrix
    paramEst.latAccelEst = float(self.lat_accel_from_curv)
    paramEst.massEst = self.mass
    if self.unestimated:
      paramEst.status = custom.ParamEst.Status.unestimated
    else:
      if np.linalg.det(self.P) < 1e-3:
        paramEst.status = custom.ParamEst.Status.converged
      else:
        paramEst.status = custom.ParamEst.Status.unconverged

    if debug:
      paramEst.debug = True
    return msg

  def understeer_gradient_calc(self):
    measured_curvature = -self.VM.calc_curvature(math.radians(self.steering_angle_deg - self.angleOffsetDeg), self.v_ego, self.roll)
    self.lat_accel_from_curv = measured_curvature * self.v_ego ** 2
    # only update when data met some creteria and no division by zero
    if self.v_ego > MIN_SPEED and self.SR > 0:
      self.adjusted_steering_angle = ACC_G * (self.steering_angle_deg / self.SR - self.wheelbase / self.v_ego * self.yaw_rate)
      self.theta, self.P = rls(self.adjusted_steering_angle, self.lat_accel_from_curv, self.P, self.theta)
      self.unestimated = False
    else:
      return

def main():
  # real-time pocess with CPU pinning, run on specified cores
  # priority 5 (low), run only on cores 0-3
  config_realtime_process([0, 1, 2, 3], 5)
  DEBUG = bool(int(os.getenv("DEBUG", "0")))

  pm = messaging.PubMaster(['paramEst'])
  sm = messaging.SubMaster(['livePose', 'liveCalibration', 'carState', 'liveParameters', 'carParams'], poll='livePose')
  params = Params()
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  estimator = understeerGradientEstimator(CP)

  while True:
    sm.update()
    if sm.all_checks():
      for which in sorted(sm.updated.keys(), key=lambda x: sm.logMonoTime[x]):
        if sm.updated[which]:
          t = sm.logMonoTime[which] * 1e-9
          estimator.handle_log(t, which, sm[which])

      estimator.understeer_gradient_calc()
      est_msg = estimator.get_msg(sm.all_checks(), DEBUG)
      est_msg_dat = est_msg.to_bytes()
      pm.send('paramEst', est_msg_dat)
