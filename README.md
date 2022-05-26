# Walking_Robot_Quadruped_MPC
A program in Walking Robot course.

## Introduction about each file:

* the 'main.py' file is the entrance of the program, you can adjust the paramaters, like step period, step height, in this file
* the 'QuadrupedSim.py' is the core code for the program, primary simulation function and control function are all included in this file;
* the main function for 'Planner.py' is design a target trajectory for the toes, currently we use the compenont curve as the trajactory.
* the 'test.py' is original file for the program, 3 files above are base on this file
* ‘mini_cheetach2.urdf’ is downloaded from Google, which is xacro form, it's more convinient to modify paramaters for robot compared with the model in pybullet dirctory
* 'meshes' includes file used for modeling



