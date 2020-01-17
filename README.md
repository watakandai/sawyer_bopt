# Gazebo Simulation of "Motion learning using Bayesian optimization in an environment with an unkown object

## Gazebo Installation
Before doing anything, turn off pyenv or any other python version management package.
Just use defualt python 2. <br>


1. Install ROS Melodic by following the official tutorials.
- http://wiki.ros.org/melodic/Installation/Ubuntu
- http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment

2. After installing ROS, install [Sawyer Simulation](http://nu-msr.github.io/embedded-course-site/notes/baxter_introduction.html#sawyer-build-instructions). 
```
cd ~/catkin_ws/src/
wstool init
wstool merge https://gist.githubusercontent.com/jarvisschultz/f65d36e3f99d94a6c3d9900fa01ee72e/raw/sawyer_packages.rosinstall
wstool update
cd ..
catkin_make
```

3. Another package we need is [sawyer_pykdl](https://github.com/rupumped/sawyer_pykdl)

```
cd ~/catkin_ws/src/
git clone https://github.com/rupumped/sawyer_pykdl.git
cd ..
catkin_make
```

4. clone this repository. 
```
cd ~/catkin_ws/src/
git clone git@github.com:watakandai/sawyer_bopt.git
cd ..
catkin_make
cd ~/catkin_ws/src/sawyer_bopt
./install.sh
``` 

5. Manually, download `traj.bag` from my [Google Drive](https://drive.google.com/file/d/1QcMMU8FDnmpvHfBZvd4fnifWIUG64hnU/view?usp=sharing). Place it under `assets/`


## Image
![gazebo_simulation](assets/gazebo_simulation.png)

## To run
```
roslaunch sawyer_bopt bopt_experiment.launch
```

### options
#### Change world 
```
roslaunch sawyer_bopt bopt_experiment.launch world_name:=bopt_exp_60
```

world_name has options
- bopt_exp_30
- bopt_exp_60

#### Change experiment
```
roslaunch sawyer_bopt bopt_experiment.launch experiment:=convergence
```

experiment has options
- comparison 
- convergence
