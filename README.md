# solarsystem
Runge-Kutta integrator for simulating the solar system

### About
This code contains a custom 4th order Runge-Kutta itegrator acting on the equations of motion of the bodies of the solar system. The initial positions and velocities are found using `astropy` which allows you to find the orbital characteristics of any body contained in its database at a given time. The code finds the positions of each planet in the solar system at the current time, and then integrates forward in time by steps to calculate where each body will be subsequently.

### Getting started
The integration time can be changed by changing the number of ouputs and by changing the timestep. A longer total time will take longer to integrate. A smaller timestep will give more accurate results, however even large timesteps are relatively accurate when using this integrator as long as the bodies don't approach each other too closely. Units of time are years while units of distance are astronomical units, where 1 astronomical unit is equal to the distance from the Earth to the Sun, and masses are measured in solar units.

### Plotting
Both 2D and 3D animations can be created using the code. FFMpegWriter must be installed to save them correctly.

### Requirements
numpy, matplotlib, astropy, mpl_toolkits, 
