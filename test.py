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
        # self.robot = p.loadURDF('mini_cheetah/mini_cheetah.urdf', startPos)
        self.robot = p.loadURDF('mini_cheetah2.urdf', startPos)

        self.n_j = 12
        self.simu_f = 200
        self.q_vec = np.zeros(self.n_j)
        self.joints_p = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.dq_vec = np.zeros(self.n_j)
        self.joints_v = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.toe_position = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
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
        # self.init_sim_states()
        self.camera_sliders = []
        if set_cameras: self.add_camera_sliders()  # adjust [pitch, yaw] for now
        self.trot_phase = 0  # 0 is when fl and hr are in swing ([1, 0, 0, 1]), 1 is otherwise ([0, 1, 1, 0])
        self.pre_trot_phase = 0
        self.swing_phase_cnt = 0  # this % 2 = 0 when trot_phase = 0
        self.foot_contact = [0, 0, 0, 0]
        # self.pre_foot_contact = [0, 0, 0, 0] # no use?
        if start_sim: p.stepSimulation()
        self.base_position = (0, 0, 0.5)
        self.base_orientation = None

        # torque control
        self.maxForceId = p.addUserDebugParameter("maxForce", 0, 100, 20)

        # paras for plot
        self.base_h_mat = np.zeros((self.simu_f * 3, 1))
        self.toe_h_mat = np.zeros((self.simu_f*3, 1))
        self.desire_toe_h_mat = np.zeros((self.simu_f*3,1))
        self.plot_mat = np.zeros((self.simu_f, 3))
        self.planner_plot = np.zeros(200)
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

    # def init_sim_states(self):
    #     for _i in range(4):
    #         p.resetJointState(self.robot, self.abad_link_IDs[_i], self.init_leg_states[0])
    #         p.resetJointState(self.robot, self.thigh_link_IDs[_i], self.init_leg_states[1])
    #         p.resetJointState(self.robot, self.shank_link_IDs[_i], self.init_leg_states[2])

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
        q_list = p.calculateInverseKinematics2(self.robot, self.toe_link_IDs, four_vec3_dest)
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
        q_list = p.calculateInverseKinematics2(self.robot, self.toe_link_IDs, p_in_cauchy)
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
        # print('height in base cor:', p_toes[0][2] + d[2])
        # print('desire toe height:', p_toe_in_cauchy[0][2])
        return p_toe_in_cauchy



    def run(self, pl):
        dt = 1 / self.simu_f
        simulation_times = 5000
        simulation_times2 = 100
        times_cnt = 0
        times_cnt2 = 0
        t = 0
        for j in range(12):
            # Disable motor in order to use direct torque control.
            info = p.getJointInfo(self.robot, self.joints[j])
            print(info)

        while times_cnt < 40:
            self.update_base_pos_ori()
            self.update_camera_vision()
            self.update_joints_p_and_v()
            p.stepSimulation()
            self.plot_mat[:-1, 0] = self.plot_mat[1:, 0]  # update_body position
            self.plot_mat[-1, 0] = self.base_position[2]
            time.sleep(dt)
            times_cnt += 1

        self.stand_up(dt, pl)
        self.init_motor()
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
            p_in_base = pl.traj_2_base(p_by_traj, 0.25)
            p_in_cauchy = self.base_2_cauchy(p_in_base)
            self.planner_plot = pl.plot_test
            q_list = p.calculateInverseKinematics2(self.robot, self.toe_link_IDs, p_in_cauchy)  # the target position for each joints in 12xx1
            q_4 = []
            for _i in range(4):
                q_4.append([q_list[_i * 3], q_list[_i * 3 + 1], q_list[_i * 3 + 2]])
            torque = self.joint_controller(q_4)
            self.step2(torque)
            # self.step(q_4, 0)
            # p.applyExternalForce(self.robot, -1, (0, -10, 0), (0,0,0), p.LINK_FRAME ) # the force (up,cross, forward)
            # Jacobian = self.ComputeJacobian(0)
            # if 0 == times_cnt % 20:
            #     self.update_plot()
            time.sleep(1 / self.simu_f )


        self.close_sim()


    def stand_up(self, dt, planner, stand_height=0.25, init_height=0.1265, stand_T=0.5):
        _time_cnt = 0
        _t = 0
        while _time_cnt * dt < stand_T / 2:
            p.stepSimulation()
            self.update_base_pos_ori()
            _t += dt
            _, _, d_z = planner.stand_up_traj(stand_height, init_height, _t, stand_T)
            q_in_shoulder = [[0, 0, d_z]] * 4
            q_in_base = planner.traj_2_base(q_in_shoulder, 0.1265)
            q_4_list = self.base_2_cauchy(q_in_base)
            q_vec = p.calculateInverseKinematics2(self.robot, self.toe_link_IDs, q_4_list)
            q_4 = []
            for _i in range(4):
                q_4.append([q_vec[_i * 3], q_vec[_i * 3 + 1], q_vec[_i * 3 + 2]])
            # self.set_motor_pos_and_vel(q_4, 0)
            # self.step2(torque)
            self.step(q_4, 0)

            ## update q and v for joints
            self.update_joints_p_and_v()
            time.sleep(dt)
            _time_cnt += 1

    def update_base_pos_ori(self):
        self.base_position = p.getBasePositionAndOrientation(self.robot)[0]
        self.base_orientation = p.getBasePositionAndOrientation(self.robot)[1]


    def init_motor(self):
        maxForce = 0
        mode = p.VELOCITY_CONTROL
        for i in range(12):
            p.setJointMotorControl2(self.robot, self.joints[i], controlMode=mode, force = maxForce)

    def set_motor_pos_and_vel(self, q_d_vec, dq_d_vec):
        """
        update 12 + 12 = 24 variables for motor control

        :param q_d_vec: position circle, [fl, fr, hl, hr] x [abad, thigh, shank]
        :param dq_d_vec: velocity circle
        """
        p.setJointMotorControlArray(self.robot, jointIndices=self.FR_leg, controlMode=p.POSITION_CONTROL,
                                    targetPositions=q_d_vec[0])  # joints q_ for fr leg
        p.setJointMotorControlArray(self.robot, jointIndices=self.FL_leg, controlMode=p.POSITION_CONTROL,
                                    targetPositions=q_d_vec[1])
        p.setJointMotorControlArray(self.robot, jointIndices=self.RR_leg, controlMode=p.POSITION_CONTROL,
                                    targetPositions=q_d_vec[2])
        p.setJointMotorControlArray(self.robot, jointIndices=self.RL_leg, controlMode=p.POSITION_CONTROL,
                                    targetPositions=q_d_vec[3])


    def set_motor_torque_array(self, torque_array):
        '''
        :param torque_array: the torque of [fl, fr, hl, hr] x [abad, thigh, shank]
        '''
        maxForce = p.readUserDebugParameter(self.maxForceId)
        if torque_array is None:
            torque_array = np.zeros(self.n_j)
        p.setJointMotorControlArray(self.robot, jointIndices=self.FR_leg, controlMode=p.TORQUE_CONTROL,
                                    forces=torque_array[0])  # joints q_ for fr leg
        p.setJointMotorControlArray(self.robot, jointIndices=self.FL_leg, controlMode=p.TORQUE_CONTROL,
                                    forces=torque_array[1])
        p.setJointMotorControlArray(self.robot, jointIndices=self.RR_leg, controlMode=p.TORQUE_CONTROL,
                                    forces=torque_array[2])
        p.setJointMotorControlArray(self.robot, jointIndices=self.RL_leg, controlMode=p.TORQUE_CONTROL,
                                    forces=torque_array[3])
        # for j in range(12):
        #     p.setJointMotorControl2(self.robot, self.joints[j], p.TORQUE_CONTROL, force=torque_array[j], positionGain = 0, velocityGain = 0)


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
        self.update_joints_p_and_v()
        # return None  # TODO
        self.plot_mat[:-1, 0] = self.plot_mat[1:, 0] # update_body position
        self.plot_mat[-1, 0] = self.base_position[2]
        self.q_mat[:-1] = self.q_mat[1:]
        self.q_mat[-1] = self.q_vec
        return self.get_joint_states()

    def step2(self, torque_array):
        self.set_motor_torque_array(torque_array)
        p.stepSimulation()
        ## some parmas need to be updated
        self.update_camera_vision()
        self.update_base_pos_ori()
        self.update_joints_p_and_v()
        self.get_toes_position()
        self.plot_mat[:-1, 0] = self.plot_mat[1:, 0]  # update_body position
        self.plot_mat[-1, 0] = self.base_position[2]
        self.q_mat[:-1] = self.q_mat[1:]
        self.q_mat[-1] = self.q_vec

        # return self.get_joint_states()


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

    def update_joints_p_and_v(self):
        self.q_vec, self.dq_vec = self.get_joint_states()
        for i in range(4):
            for j in range(3):
                self.joints_p[i][j] = self.q_vec[3*i + j]
                self.joints_v[i][j] = self.dq_vec[3*i + j]


    def get_toes_position(self):
        for i in range(4):
            tmp_list = p.getLinkState(self.robot, self.toe_link_IDs[i])[0]
            self.toe_position[i] = tmp_list


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
                self.foot_contact[cp[4] // 4] = 0

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
        :param q_d_vec: target position for each joints, which is 4x3
        :return: virtual torque
        '''
        torque = []
        q_d_vec_tmp = np.zeros(12)
        for i_ in range(4):
            for j_ in range(3):
                q_d_vec_tmp[3*i_+j_] = q_d_vec[i_][j_]
        dq_d_vec = (np.array(q_d_vec) - np.array(self.joints_p)) * self.simu_f
        # dq_d_vec = np.r_[q_d_vec_tmp - self.q_vec] * self.simu_f
        # dq_d_vec = q_d_vec - self.joints_p
        # print("target p:", q_d_vec_tmp)
        # print("current p:",self.q_vec)
        # print("target v:",dq_d_vec)
        # print("current v:",self.dq_vec)
        self.q_d_mat[:-1] = self.q_d_mat[1:]
        self.q_d_mat[-1] = q_d_vec_tmp
        for i in range(4):
            torque.append(self.joint_impedance_controller(i, self.joints_p[i], self.joints_v[i], q_d_vec[i], dq_d_vec[i]))
        return torque

    def joint_impedance_controller(self,leg_ID, q_vec, dq_vec, q_d_vec, dq_d_vec):
        k = [12.5, 12.5, 12.5, 12.5]
        b = [1.25, 1.25, 1.25, 1.25]

        torque_array = k[leg_ID] * (np.array(q_d_vec) -np.array( q_vec)) + b[leg_ID] * (np.array(dq_d_vec) - np.array(dq_vec))
        torque_array = list(torque_array)
        # print("torque:",torque_array)
        print("--------")
        return torque_array

########################### MPC #######################

    def compute_jacobian(self, robot, link_id):
        """Computes the Jacobian matrix for the given link.

        Args:
          robot: A robot instance.
          link_id: The link id as returned from loadURDF.

        Returns:
          The 3 x N transposed Jacobian matrix. where N is the total DoFs of the
          robot. For a quadruped, the first 6 columns of the matrix corresponds to
          the CoM translation and rotation. The columns corresponds to a leg can be
          extracted with indices [6 + leg_id * 3: 6 + leg_id * 3 + 3].
        """
        all_joint_angles = self.q_vec
        zero_vec = [0] * len(all_joint_angles)
        # it's different from discirbtion in guide
        jv, _ = p.calculateJacobian(robot, link_id,
                                                       (0, 0, 0), all_joint_angles,
                                                       zero_vec, zero_vec)
        jacobian = np.array(jv)
        assert jacobian.shape[0] == 3
        print('the Jacobian Matrix:')
        return jacobian


    def ComputeJacobian(self, leg_id):
        """Compute the Jacobian for a given leg."""
        # Does not work for Minitaur which has the four bar mechanism for now.
        assert len(self.toe_link_IDs) == 4 # 如果不满足这个报错，就是
        return self.compute_jacobian(
            robot=self.robot,
            link_id=self.toe_link_IDs[leg_id],
        )

    def MapContactForceToJointTorques(self, leg_id, contact_force):
        """Maps the foot contact force to the leg joint torques."""
        jv = self.ComputeJacobian(leg_id)  # 计算目标控制腿的雅可比矩阵
        all_motor_torques = np.matmul(contact_force, jv)  # 是不是少了一个R？
        motor_torques = {}
        motors_per_leg = self.num_motors // self.num_legs
        # 将计算得到的力矩传给关节
        com_dof = 6
        for joint_id in range(leg_id * motors_per_leg,
                              (leg_id + 1) * motors_per_leg):
            motor_torques[joint_id] = all_motor_torques[
                                          com_dof + joint_id] * self._motor_direction[joint_id]

        return motor_torques
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
        self.planner_heights = []
        plt.subplot(3, 1,  1)
        body_h, = plt.plot(self.plot_mat[:, 0], '-')
        plt.ylim([0, 0.6])
        plt.subplot(3, 1, 2)
        toe_d_h, = plt.plot(self.plot_mat[:,1], '-')
        toe_h, = plt.plot(self.plot_mat[:,2], '-')
        planner_height, = plt.plot(self.planner_plot[:], '-')
        plt.ylim([0, 0.2])

        self.body_hs.append(body_h)
        print('body_hs', self.body_hs)
        self.toe_d_hs.append(toe_d_h)
        self.toe_hs.append(toe_h)
        self.planner_heights.append(planner_height)
        plt.xlabel('Simulation steps')
        self.fig.legend(['q_d', 'q'], loc='lower center', ncol=2, bbox_to_anchor=(0.49, 0.97), frameon=False)
        self.fig.tight_layout()
        plt.draw()

    def update_plot(self):
        self.body_hs[0].set_ydata(self.plot_mat[:, 0])
        self.toe_d_hs[0].set_ydata(self.plot_mat[:, 1])
        self.toe_hs[0].set_ydata(self.plot_mat[:, 2])
        self.planner_heights[0].set_ydata(self.planner_plot[:])
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
    plot_test = np.zeros(200)

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

        t = t % self.T #if t > self.T else t
        x = self.S * (t / self.T - np.sin(2 * np.pi * t / self.T) / (2 * np.pi)) #- self.S / 2
        fE = t / self.T - np.sin(4 * np.pi * t / self.T) / (4 * np.pi)
        z = self.H * (np.sign(self.T / 2 - t) * (2 * fE - 1) + 1)
        y = 0
        return [x, y, z]

    def cal_step(self, current_p, target_p):
        '''
        this funciton is used to programming a target trajectory accrooding to current position
        and target postion for toe
        param current_p:
        '''
        return np.sqrt((target_p[0] - current_p[0])**2 + (target_p[1] - current_p[1])**2)


    def trot_swing_realtime_traj(self, t, S, H):
        t = t % self.T
        x = S * (t / self.T - np.sin(2 * np.pi * t / self.T) / (2 * np.pi))
        fE = t / self.T - np.sin(4 * np.pi * t / self.T) / (4 * np.pi)
        z =H * (np.sign(self.T / 2 - t) * (2 * fE - 1) + 1)
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




if __name__ == '__main__':
    qs = QuadrupedSim(start_sim=True)
    test_x_list = []
    test_z_list = []
    test_z = []
    test_dz = []
    t_test = 0

    T = 0.08 # simu step dt = 0.005, so T should higher than 0.2
    pl = Planner()
    pl.init_trot_params(0.06, 0.12, T)
    qs.run(pl)

