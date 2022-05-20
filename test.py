import time
import pybullet as p
import pybullet_data as pyd
import numpy as np
import matplotlib.pyplot as plt




class QuadrupedSim(object):
    def __init__(self, set_color=True, set_sliders=False, set_cameras=True, start_sim=False):
        self.physicsClient = p.connect(p.GUI)  # connect to simulation server, the return value is a client ID
        # set visualization angle, pitch is up/down, yaw is right/left
        # self.camera_focus = [0, 0, 0.3] # could change this to move camera
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0,
                                     cameraPitch=0, cameraTargetPosition=[0, 0, 0.3])
        p.setGravity(0, 0, -9.8)  # the fall down performance is a little strange?
        p.setAdditionalSearchPath(pyd.getDataPath())
        self.floor = p.loadURDF('plane.urdf')
        startPos = [0, 0, 0.2]  # float and fixed in air
        self.robot = p.loadURDF('mini_cheetah/mini_cheetah.urdf', startPos)

        self.n_j = 12
        self.simu_f = 200
        self.q_vec = np.zeros(self.n_j)
        self.dq_vec = np.zeros(self.n_j)
        self.q_mat = np.zeros((self.simu_f * 3, self.n_j))
        self.q_d_mat = np.zeros((self.simu_f * 3, self.n_j))

        self.joints = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        self.FR_leg = [0, 1, 2]
        self.FL_leg = [4, 5, 6]
        self.RR_leg = [8, 9, 10]
        self.RL_leg = [12, 13, 14]


        self.torso_ID = -1
        # the IDs and names correspond to:
        # [fl, fr, hl, hr]
        self.abad_link_IDs = [0, 4, 8, 12]
        self.abad_joint_Names = ['torso_to_abduct_fl',
                                 'torso_to_abduct_fr',
                                 'torso_to_abduct_hl',
                                 'torso_to_abduct_hr']
        self.thigh_link_IDs = [1, 5, 9, 13]
        self.thigh_joint_Names = ['abduct_fl_to_thigh_fl',
                                  'abduct_fl_to_thigh_fr',
                                  'abduct_fl_to_thigh_hl',
                                  'abduct_fl_to_thigh_hr']
        self.shank_link_IDs = [2, 6, 10, 14]
        self.shank_joint_Names = ['thigh_fl_to_knee_fl',
                                  'thigh_fl_to_knee_fr',
                                  'thigh_fl_to_knee_hl',
                                  'thigh_fl_to_knee_hr']
        self.toe_link_IDs = [3, 7, 11, 15]  # there are 16 links (base-excluded) and 16 joints, but the toe joint
        self.leg_xy_offset = [[0.19, -0.11], [0.19, 0.11], [-0.19, -0.11],
                              [-0.19, 0.11]]  # offset of [fl, fr, hl, hr] from CoM
        if set_color: self.color_links()
        if set_sliders: self.add_custom_sliders()  # add sliders for debug use
        self.init_leg_states = [0, -1.5, 2.4]  # initialize a state for standing
        self.init_sim_states()
        self.camera_sliders = []
        if set_cameras: self.add_camera_sliders()  # adjust [pitch, yaw] for now
        self.trot_phase = 0  # 0 is when fl and hr are in swing ([1, 0, 0, 1]), 1 is otherwise ([0, 1, 1, 0])
        self.pre_trot_phase = 0
        self.swing_phase_cnt = 0  # this % 2 = 0 when trot_phase = 0
        self.foot_contact = [0, 0, 0, 0]
        # self.pre_foot_contact = [0, 0, 0, 0] # no use?
        if start_sim: p.stepSimulation()
        self.base_position = (0,0,0.5)
        self.base_orientation = None

        # paras for plot
        self.base_h_mat = np.zeros((self.simu_f * 3, 1))
        self.toe_h_mat = np.zeros((self.simu_f*3, 1))
        self.desire_toe_h_mat = np.zeros((self.simu_f*3,1))
        self.plot_mat = np.zeros((self.simu_f, 3))
        self.plot_test()




    def reset(self):
        p.resetSimulation()  # TODO: need to test
        pass  # add something more?

    def add_camera_sliders(self):
        self.camera_sliders.append(p.addUserDebugParameter('pitch', 180, 360, 180))
        self.camera_sliders.append(p.addUserDebugParameter('yaw', 0, 360, 180))
        self.camera_sliders.append(p.addUserDebugParameter('distance', 0, 5, 1))

    def update_camera_vision(self):
        p.resetDebugVisualizerCamera(cameraDistance=p.readUserDebugParameter(self.camera_sliders[2]),
                                     cameraYaw=p.readUserDebugParameter(self.camera_sliders[1]),
                                     cameraPitch=p.readUserDebugParameter(self.camera_sliders[0]),
                                     cameraTargetPosition=[0, 0, 0.3])

    def init_sim_states(self):
        for _i in range(4):
            p.resetJointState(self.robot, self.abad_link_IDs[_i], self.init_leg_states[0])
            p.resetJointState(self.robot, self.thigh_link_IDs[_i], self.init_leg_states[1])
            p.resetJointState(self.robot, self.shank_link_IDs[_i], self.init_leg_states[2])

    def color_links(self):
        p.changeVisualShape(self.robot, self.torso_ID, rgbaColor=[1, 1, 1, 0.5])
        for abad_id in self.abad_link_IDs:
            p.changeVisualShape(self.robot, abad_id, rgbaColor=[1, 0.9451, 0.6745, 0.5])
        for thigh_id in self.thigh_link_IDs:
            p.changeVisualShape(self.robot, thigh_id, rgbaColor=[0.9765, 0.7373, 0.8667, 0.5])
        for shank_id in self.shank_link_IDs:
            p.changeVisualShape(self.robot, shank_id, rgbaColor=[0.8353, 0.6431, 0.8118, 0.5])
        for toe_id in self.toe_link_IDs:
            p.changeVisualShape(self.robot, toe_id, rgbaColor=[0.7137, 0.5373, 0.6902, 0.5])

    def add_custom_sliders(self):
        for _i in range(4):
            p.addUserDebugParameter(self.abad_joint_Names[_i], -3.14, 3.14, 0)
            p.addUserDebugParameter(self.thigh_joint_Names[_i], -3.14, 3.14, 0)
            p.addUserDebugParameter(self.shank_joint_Names[_i], -3.14, 3.14, 0)

    def IK_cal2(self, four_vec3_dest):
        '''

        :param four_vec3_dest: target position for the end effector, which is a 4x3 list in cauchy coordinate
        :return: 3 position for joints for each leg， which is also a 4x3 list， standing for joints position
        '''
        q_4 = []
        q_list = p.calculateInverseKinematics2(qs.robot, self.toe_link_IDs, four_vec3_dest)
        for _i in range(4):
            # [[fl_abad, fl_thigh, fl_shank],
            #  [fr_abad, fr_thigh, fr_shank],
            #  [hl_abad, hl_thigh, hl_shank],
            #  [hr_abad, hr_thigh, hr_shank]]
            # could use np.array() and change to np.ndarray
            q_4.append([q_list[_i * 3], q_list[_i * 3 + 1], q_list[_i * 3 + 2]])
        return q_4

    def IK_cal3(self, p_in_shoulder):
        q_4 = []
        p_in_base = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]] # CANNOT directly plus 4, I don't know why
        for i in range(4):
            for j in range(3):
                if j != 2:
                    p_in_base[i][j] = p_in_shoulder[i][j] + self.leg_xy_offset[i][j]
                else:
                    p_in_base[i][j] = p_in_shoulder[i][j]
        # print("p_in_shoulder : ", p_in_shoulder)
        # print("p_in_body :", p_in_body)
        p_in_cauchy = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
        for m in range(4):
            for n in range(3):
                p_in_cauchy[m][n] = p_in_base[m][n] + self.base_position[n]
        # print("p_in_cauchy", p_in_cauchy)
        q_list = p.calculateInverseKinematics2(qs.robot, self.toe_link_IDs, p_in_cauchy)
        for _i in range(4):
            q_4.append([q_list[_i * 3], q_list[_i * 3 + 1], q_list[_i * 3 + 2]])
        return q_4

    def base_2_cauchy(self, p_toes):
        '''
        :param p_toes:4x3 list , the position for traj in base coordinate
        :return: a 4x3 list for toes in cauchy coordinate
        '''
        p_toe_in_cauchy = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        # cal the Rotation matrix and d
        # tmp_or = (0, 0.1 , 0, 0.1)
        R = p.getMatrixFromQuaternion(self.base_orientation)
        slices = [list(R[0: 3]), list(R[3: 6]), list(R[6: 9])]
        d = list(self.base_position)
        for i in range(4):
            p_toe_in_cauchy[i] = np.dot(slices, p_toes[i]) + d
        self.plot_mat[:-1, 1] = self.plot_mat[1:, 1]
        self.plot_mat[-1, 1] = p_toe_in_cauchy[0][2]
        print('height in base cor:', p_toes[0][2] + d[2])
        print('desire toe height:', p_toe_in_cauchy[0][2])
        return p_toe_in_cauchy



    def run(self, pl):
        dt = 1 / qs.simu_f
        simulation_times = 5000
        times_cnt = 0
        t = 0

        while times_cnt < 50:
            self.update_base_pos_ori()
            self.update_camera_vision()
            p.stepSimulation()
            self.plot_mat[:-1, 0] = self.plot_mat[1:, 0]  # update_body position
            self.plot_mat[-1, 0] = self.base_position[2]
            time.sleep(dt)
            times_cnt += 1

        self.stand_up(dt, pl)

        while times_cnt < simulation_times:
            # this is test for getJointInfo function
            # _, _, _, _, _, _,_,_,_,_,_,_,_,_, toe_p ,_, _, = p.getJointInfo(self.robot, 7)
            # print("the toe height: ", toe_p)
            t = t + dt
            # print("current t :", t)
            self.update_camera_vision()
            [swing_x, y1, swing_z] = pl.trot_traj_plan_swing(t)
            [support_x, y2, support_z] = pl.trot_traj_plan_support(t)
            # change phase when a period T has passed, start with fl and hr to be swing phase
            phase = t // T % 2
            fl_hr_x = swing_x if phase == 0 else support_x
            fl_hr_z = swing_z if phase == 0 else support_z
            fr_hl_x = support_x if phase == 0 else swing_x
            fr_hl_z = support_z if phase == 0 else swing_z
            # get the q_d_vec in 3x4 form
            p_by_traj = [[fl_hr_x, 0, fl_hr_z], [fr_hl_x, 0, fr_hl_z], [fr_hl_x, 0, fr_hl_z], [fl_hr_x, 0, fl_hr_z]]
            p_in_base = pl.traj_2_base(p_by_traj)
            p_in_cauchy = self.base_2_cauchy(p_in_base)

            q_list = p.calculateInverseKinematics2(qs.robot, self.toe_link_IDs, p_in_cauchy)
            q_4 = []
            for _i in range(4):
                q_4.append([q_list[_i * 3], q_list[_i * 3 + 1], q_list[_i * 3 + 2]])
            self.step(q_4, 0)
            if 0 == times_cnt % 20:
                self.update_plot()
            times_cnt += 1
            time.sleep(1 / self.simu_f)
        self.close_sim()



        pass # TODO

    def stand_up(self, dt, planner, stand_height=0.25, init_height=0.1265, stand_T=0.5):
        _time_cnt = 0
        _t = 0
        while _time_cnt * dt < stand_T / 2:
            p.stepSimulation()
            self.update_base_pos_ori()
            _t += dt
            _, _, d_z = planner.stand_up_traj(stand_height, init_height, _t, stand_T)
            test_dz.append(d_z)
            test_z.append(self.base_position[2])
            # the calculate get a
            q_in_shoulder = [[0, 0, d_z]] * 4
            q_4_list = self.IK_cal3(q_in_shoulder)
            self.set_motor_pos_and_vel(q_4_list, 0)
            time.sleep(dt)
            _time_cnt += 1

    def update_base_pos_ori(self):
        self.base_position = p.getBasePositionAndOrientation(self.robot)[0]
        self.base_orientation = p.getBasePositionAndOrientation(self.robot)[1]


    def set_motor_pos_and_vel(self, q_d_vec, dq_d_vec):
        """
        update 12 + 12 = 24 variables for motor control

        :param q_d_vec: position circle, [fl, fr, hl, hr] x [abad, thigh, shank]
        :param dq_d_vec: velocity circle
        """
        p.setJointMotorControlArray(self.robot, jointIndices=self.FR_leg, controlMode=p.POSITION_CONTROL,
                                        targetPositions=q_d_vec[0]) # joints q_ for fr leg
        p.setJointMotorControlArray(self.robot, jointIndices=self.FL_leg, controlMode=p.POSITION_CONTROL,
                                        targetPositions=q_d_vec[1])
        p.setJointMotorControlArray(self.robot, jointIndices=self.RR_leg, controlMode=p.POSITION_CONTROL,
                                        targetPositions=q_d_vec[2])
        p.setJointMotorControlArray(self.robot, jointIndices=self.RL_leg, controlMode=p.POSITION_CONTROL,
                                    targetPositions=q_d_vec[3])


    def set_motor_torque_array(self, torque_array=None):
        '''
        :param torque_array: the torque of [fl, fr, hl, hr] x [abad, thigh, shank]
        '''
        if torque_array is None:
            torque_array = np.zeros(self.n_j)
        for j in range(len(self.joints)):
            p.setJointMotorControl2(self.robot, self.joints[j], p.TORQUE_CONTROL, force=torque_array[j])


    def step(self, q_d_vec, dq_d_vec):
        """
        call this inside a while circle (run) to update several states
        :return:
        1. joint angle: q_vec
        2. joint angular velocity: dq_vec
        """
        self.set_motor_pos_and_vel(q_d_vec, dq_d_vec)
        p.stepSimulation()  # do not need to run in outside loop
        self.update_camera_vision()
        self.update_base_pos_ori()
        # return None  # TODO
        self.plot_mat[:-1, 0] = self.plot_mat[1:, 0] # update_body position
        self.plot_mat[-1, 0] = self.base_position[2]
        self.q_mat[:-1] = self.q_mat[1:]
        self.q_mat[-1] = self.q_vec
        return self.get_joint_states()

    def step2(self, torque_array):
        self.set_motor_torque_array(torque_array)
        p.stepSimulation()
        self.q_mat[:-1] = self.q_mat[1:]
        self.q_mat[-1] = self.q_vec
        return self.get_joint_states()


    def get_joint_states(self):
        '''
        :return: q_vec: joint angle, dq_vec: joint angular velocity
        not sure here the dim is 12 or 16
        '''
        q_vec = np.zeros(12)
        dq_vec = np.zeros(12)
        for j in range(12):
            q_vec[j], dq_vec[j], _, _ = p.getJointState(self.robot, self.joints[j])
        return q_vec, dq_vec

    def update_foot_contact_state(self):
        """
        get contact situations by `p.getContactPoints()`, then update contact situation
        Body 0: plane;
        Body 1: robot
        """

        for cp in p.getContactPoints(0, 1):
            if cp[4] in self.toe_link_IDs:
                self.foot_contact[cp[4] // 4] = 1
                # p.addUserDebugText('%s: %d' % (foot_name_list[cp[4] % 4], 1), p.getLinkState(self.robot, cp[4])[0], lifeTime=0.1)
            else:
                qs.foot_contact[cp[4] // 4] = 0

    def finite_state_controller(self):
        """
        change between stance and swing states
        :return:
        """
        if self.foot_contact == [1, 0, 0, 1]:
            self.trot_phase = 0
        elif self.foot_contact == [0, 1, 1, 0]:
            self.trot_phase = 1
        else:
            print('[ERR]: contact error')
        if self.pre_trot_phase != self.trot_phase:
            self.pre_trot_phase = self.trot_phase
            self.swing_phase_cnt += 1

    def cal_q_d_vec_and_d_q_d_vec(self,q_d_vec):
        q_d_vec_tmp = np.zeros(12)
        for i_ in range(4):
            for j_ in range(3):
                q_d_vec_tmp[3 * i_ + j_] = q_d_vec[i_][j_]
        # print(q_d_vec)
        # print("----")
        # print(q_d_vec_tmp)
        dq_d_vec = np.r_[q_d_vec_tmp - self.q_vec] * self.simu_f
        return q_d_vec_tmp, dq_d_vec


    def joint_controller(self, q_d_vec):
        '''

        :param t: current time
        :param q_d_vec: target position for each joints calculated by IK2, which is 12x1
        :return: virtual torque
        '''
        q_d_vec_tmp = np.zeros(12)
        for i_ in range(4):
            for j_ in range(3):
                q_d_vec_tmp[3*i_+j_] = q_d_vec[i_][j_]
        dq_d_vec = np.r_[q_d_vec_tmp - self.q_vec] * self.simu_f
        # print("target p:", q_d_vec_tmp)
        # print("current p:",self.q_vec)
        # print("target v:",dq_d_vec)
        # print("current v:",self.dq_vec)
        self.q_d_mat[:-1] = self.q_d_mat[1:]
        self.q_d_mat[-1] = q_d_vec_tmp
        return self.joint_impedance_controller(self.q_vec, self.dq_vec, q_d_vec_tmp, dq_d_vec)

    def joint_impedance_controller(self, q_vec, dq_vec, q_d_vec, dq_d_vec):
        k = 10
        b = 1
        torque_array = k * (q_d_vec - q_vec) + b * (dq_d_vec - dq_vec)
        # print("torque:",torque_array)
        # print("--------")
        return torque_array

    def joint_planner(self):
        """
        calculate positions of each joint, then use IK_cal2 to calculate joint q_d_vec. dq_d_vec could be calculated by (q_d_vec - q_d_vec_pre) / dt
        :return:
        """

        pass

    def plot_test(self):
        '''
        this function is used to plot important para for debugging
        :return:
        '''
        self.fig = plt.figure(figsize=(5, 9))
        obj_parm = ['body_heght', 'desire_FR_toe_H','actual_FR_toe_H']
        # num_of_para = np.shape(self.plot_mat[1])
        self.toe_d_hs = []
        self.toe_hs = []
        self.body_hs = []
        plt.subplot(2, 1,  1)
        body_h, = plt.plot(self.plot_mat[:, 0], '-')
        plt.ylim([0, 0.6])
        plt.subplot(2, 1, 2)
        toe_d_h, = plt.plot(self.plot_mat[:,1], '-')
        toe_h, = plt.plot(self.plot_mat[:,2], '-')
        plt.ylim([0, 0.2])
        self.body_hs.append(body_h)
        print('body_hs',self.body_hs)
        self.toe_d_hs.append(toe_d_h)
        self.toe_hs.append(toe_h)
        plt.xlabel('Simulation steps')
        self.fig.legend(['q_d', 'q'], loc='lower center', ncol=2, bbox_to_anchor=(0.49, 0.97), frameon=False)
        self.fig.tight_layout()
        plt.draw()

    def update_plot(self):
        self.body_hs[0].set_ydata(self.plot_mat[:, 0])
        self.toe_d_hs[0].set_ydata(self.plot_mat[:, 1])
        self.toe_hs[0].set_ydata(self.plot_mat[:, 2])

        # for i in range(6):
        #     self.q_d_lines[i].set_ydata(self.q_d_mat[:, i])
        #     self.q_lines[i].set_ydata(self.q_mat[:, i])
        plt.draw()
        plt.pause(0.001)




    def close_sim(self):
        p.disconnect(self.physicsClient)

class Planner:
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
        z = - (stand_height - init_height) / 2 * np.sin(2 * np.pi / stand_T * t - np.pi / 2) - (
                stand_height - init_height) / 2 - init_height
        x = y = 0
        return [x, y, z]

    def traj_2_base(self, traj):
        '''

        :param traj: this param is 4x3 list calculated in previous trajectory planner
        :return: 4x3 list, the position for traj in base coordinate
        '''
        dz = -0.25  # if dz = z for base position , the toe will always on plane
        leg_xy_offset = [[0.19, -0.11, dz], [0.19, 0.11, dz], [-0.19, -0.11, dz],
                         [-0.19, 0.11, dz]]
        p_in_base = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(4):
            for j in range(3):
                p_in_base[i][j] = traj[i][j] + leg_xy_offset[i][j]
        return p_in_base




if __name__ == '__main__':
    qs = QuadrupedSim(start_sim=True)
    test_x_list = []
    test_z_list = []
    test_z = []
    test_dz = []
    t_test = 0

    T = 0.02
    height = 0.2  # height from the float surface
    pl = Planner()
    pl.init_trot_params(0.04, 0.15, T)
    # for i in range(1000):
    #     t_test = t_test + 1/ qs.simu_f
    #
    #     x, y, z = pl.trot_traj_plan_swing(t_test)
    #     traj = [[x,y,z],[x,y,z],[x,y,z],[x,y,z]]
    #     traj_in_base = pl.traj_2_base(traj)
    #     x_in_base = traj_in_base[0][0]
    #     z_in_base = traj_in_base[0][2]
    #     test_x_list.append(x_in_base)
    #     test_z_list.append(z_in_base)
    # plt.plot(test_x_list, test_z_list)
    # plt.show()
    qs.run(pl)
    # p_in_shoulder = [[0,0,0]] * 4
    # qs.IK_cal3(p_in_shoulder)
    # simulation_times = 800
    # times_cnt = 0
    # t = 0

    # while times_cnt < simulation_times:
    #     t = t + dt
    #     qs.update_camera_vision()
    #     swing_x, swing_z = pl.traj_plan_swing(t)
    #     support_x, support_z = pl.traj_plan_support(t)
    #     # change phase when a period T has passed, start with fl and hr to be swing phase
    #     phase = t // T % 2
    #     fl_hr_x = swing_x if phase == 0 else support_x
    #     fl_hr_z = swing_z if phase == 0 else support_z
    #     fr_hl_x = support_x if phase == 0 else swing_x
    #     fr_hl_z = support_z if phase == 0 else swing_z
    #     q_4_list = qs.IK_cal2([[fl_hr_x + qs.leg_xy_offset[0][0], qs.leg_xy_offset[0][1], fl_hr_z + (0.5 - height)],
    #                            [fr_hl_x + qs.leg_xy_offset[1][0], qs.leg_xy_offset[1][1], fr_hl_z + (0.5 - height)],
    #                            [fr_hl_x + qs.leg_xy_offset[2][0], qs.leg_xy_offset[2][1], fr_hl_z + (0.5 - height)],
    #                            [fl_hr_x + qs.leg_xy_offset[3][0], qs.leg_xy_offset[3][1], fl_hr_z + (0.5 - height)]])
    #     for i in range(4):
    #         q_list = q_4_list[i]
    #         p.resetJointState(qs.robot, qs.abad_link_IDs[i], q_list[0])
    #         p.resetJointState(qs.robot, qs.thigh_link_IDs[i], q_list[1])
    #         p.resetJointState(qs.robot, qs.shank_link_IDs[i], q_list[2])
    #     qs.q_vec, qs.dq_vec = qs.get_joint_states()
    #
    #     time.sleep(dt)
    #     times_cnt += 1
    #
    # qs.close_sim()
