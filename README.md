# Simple 2D fluid simulation

## Description

This fluid sim is currently very WIP and may never be fully working. currently you mostly need to go into the script and edit variables in order to change the simulation conditions. 

I hoping to add in a GUI after adding in some command line arguments for easier running even without GUI. I also want to add custom boudry conditions, and heat simulation after an equalibrium flow is found.

the flow is currently incompressable, and may very well remain that way for 2 reasons. compressable flow would make the fluid and heat sim interact nonnegligably, and doing a compressable fluid sim requires a lot more complexity as well as Equations of State in order to get a decent simulation. 

Disclamer: I am not a simulation engineer or anything. I am a chemistry major having some fun with programming. also I'm not focused on this, so there's a high chance of abandonment. 

## Scripts

### Fluid Sim Script

Currently the simulation variables, and initial conditions must be set my changing the scripts code. I'll give brief explanations of code sections you'll need to look at.

***Initial Flow***
```
# set up initial conditions of simulation
init = np.zeros((SpaceSize,SpaceSize,2))
space = init.shape
init[40:60,:,Y] = 5
```
By default the simulation is set up to be a column of flow upward. this section is at the beginning of the main function.

***Fluid Properties***
```
# fluid properties
density = 1
kenVisc = 1
```
self explanitory, sets the density and kinetic viscocity of the fluid.

***Space Properties***
```
# simulation space properties
h = 0.25
SpaceSize = 100
```
`h` is the space between grid spaces and `SpaceSize` is the amount grid spaces. In this case the simulation space would be 25x25 length units

***Arguments***
```
-O, --Output                Sets path to save Sim Data .json to
-E, --Endtime               Sets simulation end time. 
-S, --Samples               Sets amount of samples to save to .json
```


### Simulation Viewer

The simulation viewer actually had arguments already built in. 
```
-S, --Save                  Boolean, if set will render video to file
-I, --Input                 Sets path to sim data .json file to view
-O, --Output                Sets path to save location is -S is set
-D, --Display               Boolean, if set will display and Save
-T, --Delay                 Sets delay between frames in miliseconds
```
