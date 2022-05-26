import time
import pybullet as p
import pybullet_data as pyd
import numpy as np
import matplotlib.pyplot as plt
import QuadrupedSim as QS
import Planner as PL

def start_Sim():
    qs = QS.QuadrupedSim()
    pl = PL.Planner()
    pl.init_trot_params(0.06, 0.1, 1)
    qs.run(pl)


if __name__ == '__main__':
    start_Sim()
