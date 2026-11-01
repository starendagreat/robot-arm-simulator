# Robotic Arm Simulator
Made by Steven Zhao
## Introduction
I built this project to investigate possible methods for space debris removal, to test my skills, and to learn about sensor fusion and machine learning. I was advised by Professor Du Liang (Villanova University). 
<img width="1136" height="830" alt="robotarmsimgif" src="https://github.com/user-attachments/assets/fa71cf2b-fb17-4e0f-a9ff-38782f7d7896" />

## Features
I built a network of arms that grab and catch simulated boxes. It also simulates a network of sensors where the arms have no idea of where the boxes are other than distances between the sensors and the objects.

In order to build the simulation, I used vPython for the graphics and Python for everything else. The simulation generates a network of arms and random boxes, with some stationery and others moving.

The system uses the distances to calculate the exact possible position of each object. With four distances, an object's exact position can be pinpointed down to an exact answer. Because there are multiple objects and, as a result of that, a jumbled list of distances, the system checks if the distances can possibly trilaterate into a single object in two phases. First, it checks if the 4 distances can possibly exist through the cheaper calculation method of checking if the distances fulfill the triangle inequality theorem. Then, after the vast majority of distance pairings are phased out, it can check the remaining to see if it trilaterates properly. IF it does, it is an object and the system attempts to catch it.

For the catching, the network of arm calculates each object's future position using the calculated positions and the velocities which are derived from the positions. The system assigns arms to boxes to try to maximize the number of boxes that can be caught, prioritizing moving objects over stationery ones. 

Once an arm is assigned, it moves towards the target position. I started out with creating a basic system with three degrees of freedom for each arm. Using inverse kinematics, each arm rotates to face the target placement and the two joints angles are set to the desired angle through a PID controller. The PID controller can constantly adapt to drag (which is disabled in my tests) and gravity. 

In order to further improve the arm movement, I implemented an object avoidance algorithm. Each arm calculates a "sweep" that it would have to travel through to reach the next position. If the sweep overlaps with an object, it will find a different close position using the A* algorithm, and then calculate a path that can ultimately reach the desired point.

For the catching, I implemented machine learning in order to train an agent to catch the boxes. I used the soft actor critic algorithm with Stable Baselines 3 in order to implement this functionality. At first, I wanted to have the whole arm controlled by an agent, however the accuracy ultimately was worse than the accuracy through inverse kinematics. I settled on just using machine learning for the claw, which resulted in a lot higher accuracy for caatching.

I implemented training through several stages. At first, I was just teaching the agent to close the claw when the box was close at a directly perpendicular angle and constant speeds. Then, I added in different speeds and off-center boxes in order to further improve the system's accuracy. I was able to reach close to 95% catch accuracy after training.

<img width="710" height="574" alt="armcatchinggif" src="https://github.com/user-attachments/assets/7c7e6bcb-4364-443a-b3c9-9266513763b1" />

After the boxes are caught, the system attempts to transfer over the objects to a set position. This simulates an array of robotic arms catching space debris and then collecting them all at a single spot for easy disposal. The transfer works when both arms are free and the object is placed in the middle between the two adjacent arms. The objects all eventually travel to the very end point.

<img width="734" height="558" alt="armtransfergif" src="https://github.com/user-attachments/assets/5b56ca15-b5e6-4974-a196-7f2e5ecde819" />

I hope to  improve this simulation in the future by making the physics less clunky At times, the collisions would result in an object that is being grabbed to be suddenly flung away wben it should be grabbed. Other times, when an object is being released, it can also be flung away because parts of the box may collide with the claw and take the on the claw's velocity. I also hope to return on trying to get the simulation to run faster and be lighter because at some times it can freeze up.

## How to Use
If you want to view my simulation yourself, make sure to have python installed. Download this repo and just run the simulation file in order to see the simulation for yourself. You can see buttons to end, reset, and pause the simulation along with options to turn on the auto-catch and the learnt claw grasp. There are other graphs for the arms values that I used when I was building the simulation and the inverse kinematics.
