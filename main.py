import time
import pybullet as p
import pybullet_data
import pybullet_data as pyd
import numpy as np
import matplotlib.pyplot as plt
import QuadrupedSim as QS
import Planner as PL
import mpc_stance_controller as mpc_controller
import mpc_osqp
import pybullet_data as pd
def start_Sim():
    print(pd.getDataPath())
    qs = QS.QuadrupedSim()
    pl = PL.Planner()
    controller = mpc_controller.TorqueStanceLegController(qs, (0.0, 0.0), 0.0, 0.3, 9.5, (0.07335, 0, 0, 0, 0.25068, 0, 0, 0,
                                           0.25447), 4, (0.45, 0.45, 0.45, 0.45), mpc_osqp.QPOASES)
    pl.init_trot_params(0.06, 0.12, 0.08)
    qs.run(pl, controller)



if __name__ == '__main__':
    start_Sim()
