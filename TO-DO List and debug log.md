# TO-DO List and debug log

5.21

* 目前存在的问题是，力矩控制电机运动很慢，根本达到不了位置控制的效果

  * 就算是力矩给0，机器人也没有萎下去，难道不应该萎下去吗; lab1吊起来的那个腿 力矩给0的时候是很软的，难道是因为关节的阻尼太大了？

    Lab1 机器人的阻尼和摩擦是0; 测试发现mini cheetah也没有阻尼和摩擦



5.23

* 继续解决电机力矩异常的问题

  思路：既然lab1机器人是正常的，尝试将lab1机器人导入，检查是机器人问题还是仿真本身问题。

  result: 机器人没问题，看来是仿真的问题，检查一下仿真逻辑

  从头来过，正好把逻辑检查一遍

  

  最后发现，还是他妈的力矩问题，只能给一定范围内才能保证既不卡死，又不飞

  新发现：<font color = red> 使用力矩控制模式之前要将速度模式的力设置为0，具体原因未知</font>

  

5.24

* 找到每个link对应的ID



5.28
* What we have done:

  Robot sim
  
  State estimator (Directly got from pybullet)
  
  Swing controller
  
  Stance controller

* What we need to do:

  Gait



