# Reorganization of the script structure
In order to capture result consistently, the various script version will be modified, combined, standardized.

## Reach task with two robot:
The benchmark will is the precision of the robot, i.e. end effector error. For the dual setup this error will be averaged over the two arms. 
Feedback: show that the concept is feasible, make some video, evaluate the success rate of the reaching task. Why it works.

### To do:
* keep the pose_orientation_two_robots.py similar
* create a version of the script with the gripper robot only. Accomodate the scene, the  observation and action space, and reward for 1 robot, but don't change architectural decisions.

Not sure that it will be done.

## Grasp task: 
The script is read for benchmarking. Need to define the benchmark and write it.

### To do:
* benchmark the numbers of successful grasp.

#### Main question regarding the two previous scripts: 
Given the effort on those was minimal and no sim2real methods such as DR, impedance control and such were operated, need to define how we mention them in regard to the research question. Maybe only in the annex ? 

## Reach task with deployed effort:
Multiple script version exists. The pose_orientation_gripper_robot.py is redundant I think. Some versions of pose_orientation_sim2real.py need to be put aside. The v0/v1 is redundant as the observation space is different. The v2 is the baseline for task without feedforward.
The v3 is baseline for the feedforward. It is also the setup where most parameter search was done. The v4 just has the gripper hand in it.
The v5 and v6 are the version with observation stacking and LSTM NN. It needs to be decided if they are mentioned in the thesis because they were made with very low effort to make them work.

### To do:
* understand what task are redundant more in detail.
* Organize v2, v3 for benchmarking.
* Put the gripper in v2, v3 and suppress v4, as the real setup has the gripper. I don't want to check if this gripper is needed or not. It will be kept unactuated as the real setup can't work with gripper yet.
* Probably redo a parameter search over the selected script for benchmarking.

# Task to do regarding the impedance controller
The impedance controller is a recent add and result have shown that it hasn't been properly tuned before. To benchmark could run it non-tuned, but doesn't make a lot of sense. I can also compare running the best policy in simulation with and without the impedance controller to justify its use.

cite the study where they compared action space, and controller types for the robot. Sow the impedance controller 

### To do:
* Test the effect of the tuning on the real world robot with our best policy as of now and see if the result differs a lot.
* Define can I put this impedance controller in the research question
* Detail the justification for the tuning method used for this controller and explain the method for it. (1st part go in related work and other part in method)
* Plot relevant metrics.

# Manuscript writing:

<<<<<<< HEAD
To do:
## Write the related work section
* search for source for Rl part
* source Sim2real gap part
precise preffering torque or low level position
change the real time operating system part as it is kinda wrong.
* source impedance part


=======
To do after milie interview:
* Write the related work section
>>>>>>> ddba3e82b02686b42efd73f32cdd7ffc9273144b
* In the methods write the section that detail the impedance controller tuning.
* Write the results 
* Write the discussion

## Impedance controller focus: 
Explain the tuning method, the results and the justification for using it. This will be a section in the methods and one in the results. It will be mentioned in the discussion as well.

## for the results, should be factual and quite plain.
ablation study on: 
* domain randomisation
* system identification
* impedance 
* show that it make the robot more robust.


# For the discussion
Take a more global discussion on the data. Discern global trends on the implementation of the techniques.
What is still left to address. How does it tie to the current state, what for the future. all the talks about the tradeoff.
structure the discussion. What 
on the sim2real gap: more effort put on closing the gap, but less on bridging it: talk about learning method to overcome it. LSTM
talk about how the different tasks goes into the robot woodworking.
tradeof made during the thesis.

future works, implication of the work






