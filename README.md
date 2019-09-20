# Gazebo Simulation of "Motion learning using Bayesian optimization in an environment with an unkown object

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
