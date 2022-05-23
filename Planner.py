import numpy as np

class Planner(object):
    '''
    this class calculate the position for toes in base coordinate
    '''


    # def __init__(self, step_size, step_height, T):
    #     self.S = step_size
    #     self.H = step_height
    #     self.T = T


    def __init__(self):
        self.S = None
        self.H = None
        self.T = None
        self.plot_test = np.zeros(200)

    def init_trot_params(self, step_size, step_height, T):
        self.S = step_size
        self.H = step_height
        self.T = T

    def trot_traj_plan_swing(self, t):
        """
        plan a 'Compound cycloidal trajectories'
        refer to https://blog.csdn.net/weixin_41045354/article/details/105219092
        During T/2: swing phase
        During T/2 ~ T: support phase

        :param S: step length
        :param T: period
        :param H: leg raise height
        :return: a vec3 list (arr)
        """

        t = t % self.T if t > self.T else t  # TODO, strange?
        x = self.S * (t / self.T - np.sin(2 * np.pi * t / self.T) / (2 * np.pi)) #- self.S / 2
        fE = t / self.T - np.sin(4 * np.pi * t / self.T) / (4 * np.pi)
        z = self.H * (np.sign(self.T / 2 - t) * (2 * fE - 1) + 1)
        y = 0
        return [x, y, z]


    def trot_traj_plan_support(self, t):
        t = t % self.T if t > self.T else t
        x = self.S * ((2 * self.T - t) / self.T + np.sin(2 * np.pi * t / self.T) / (2 * np.pi) - 1) #- self.S / 2
        z = 0
        y = 0
        return [x, y, z]

    def stand_up_traj(self, stand_height, init_height, t, stand_T):
        z =  (stand_height - init_height) / 2 * np.sin(2 * np.pi / stand_T * t + np.pi / 2) - (
                stand_height - init_height)/2
        x = y = 0
        return [x, y, z]

    def traj_2_base(self, traj, delta_h):
        '''

        :param traj: this param is 4x3 list calculated in previous trajectory planner
        : param delta_H : is the initial delta height between toes and body
        :return: 4x3 list, the position for traj in base coordinate
        '''
        # print('height caled by planner:', traj[0][2])
        self.plot_test[:-1] = self.plot_test[1:]
        self.plot_test[-1] = traj[0][2]
        dz = -delta_h # if dz = z for base position , the toe will always on plane
        leg_xy_offset = [[0.19, -0.11, dz], [0.19, 0.11, dz], [-0.19, -0.11, dz],
                         [-0.19, 0.11, dz]]
        p_in_base = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(4):
            for j in range(3):
                p_in_base[i][j] = traj[i][j] + leg_xy_offset[i][j]
        # print('height in body coord:', p_in_base[0][2])
        return p_in_base