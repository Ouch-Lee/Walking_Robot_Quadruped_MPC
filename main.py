import pybullet as p
import pybullet_data as pyd
import numpy as np
import matplotlib.pyplot as plt
import QuadrupedSim as QS
import Planner as PL
import mpc_stance_controller as mpc_controller

def start_Sim():
    qs = QS.QuadrupedSim()
    pl = PL.Planner()
    controller = mpc_controller.MPCStanceController(qs)
    pl.init_trot_params(0.06, 0.12, 0.08)
    qs.run(pl, controller)


if __name__ == '__main__':
    start_Sim()
