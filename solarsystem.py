# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
from timeit import default_timer as timed
from itertools import product
from matplotlib import animation

G = 4*np.pi**2                  # gravitational constant
t = Time.now()                 # gets current time - required for astropy to get the locations of all planets at that time
noutputs = 1000               # number of outputs
h = 0.01                        # integration timestep

# names of bodies for astropy to search
bodies = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']

# get_body_barycentric_posvel finds the position and velocity of solar system bodies from database
positions = [get_body_barycentric_posvel(bodies[i], t)[0].xyz for i in range(len(bodies))]
velocities = [get_body_barycentric_posvel(bodies[i], t)[1].xyz*365.25 for i in range(len(bodies))]

# masses of solar system bodies
M = np.array([1, 1.66e-7, 2.45e-6, 5.97e-6, 3.23e-7, 9.55e-4, 2.86e-4, 4.37e-5, 5.15e-5])
W = np.hstack((positions, velocities)) # create array of positions and velocities together for integrator

# %%

def force(W):
    ''' this function governs the motion of bodies as a result of the gravitational pull of surrounding bodies '''
    g = np.zeros((len(W),6))
    for i in range(len(W)):
        dvdt = np.sum([-G*M[j]*(W[i,0:3]-W[j,0:3])/(np.linalg.norm(W[i,0:3]-W[j,0:3])**3) for j in range(len(W)) if j != i], axis=0)
        g[i] = np.hstack((W[i,3:6], dvdt))
    return g

def rungekutta(W):
    ''' this is a 4th order Runge-Kutta integrator '''
    W_total = np.zeros((noutputs, len(M), 6))
    
    for i in range(noutputs): 
        fa = force(W)
        Wb = W + h/2*fa
        fb = force(Wb)
        Wc = W + h/2*fb
        fc = force(Wc)
        Wd = W + h*fc
        fd = force(Wd)    
        W = W + h/6*fa + h/3*fb + h/3*fc + h/6*fd
        W_total[i] = W
    return W_total    

timer = timed()
W_total = rungekutta(W) # 6 dimensional phase space for each particle at each timestep
print(timed()-timer)

# %%
lim = 30
fig, axes = plt.subplots(1, figsize=(10, 10))

axes.set_xlabel('distance from sun (AU)')
axes.set_ylabel('distance from sun (AU)')
axes.set_xlim([-lim,lim])
axes.set_ylim([-lim,lim])

sun, = axes.plot([], [], 'o', label="Sun", color='yellow')

mercury, = axes.plot([], [], 'o', label="Mercury", color='darkorchid')
venus, = axes.plot([], [], 'o', label="Venus", color='mediumseagreen')
earth, = axes.plot([], [], 'o', label="Earth", color='deepskyblue')
mars, = axes.plot([], [], 'o', label="Mars", color='indianred')
jupiter, = axes.plot([], [], 'o', label="Jupiter", color='darkgoldenrod')
saturn, = axes.plot([], [], 'o', label="Saturn", color='darkkhaki')
uranus, = axes.plot([], [], 'o', label="Uranus", color='lightblue')
neptune, = axes.plot([], [], 'o', label="Neptune", color='steelblue')

mercuryline, = axes.plot([], [], linewidth=1, color='darkorchid')
venusline, = axes.plot([], [], linewidth=1, color='mediumseagreen')
earthline, = axes.plot([], [], linewidth=1, color='deepskyblue')
marsline, = axes.plot([], [], linewidth=1, color='indianred')
jupiterline, = axes.plot([], [], linewidth=1, color='darkgoldenrod')
saturnline, = axes.plot([], [], linewidth=1, color='darkkhaki')
uranusline, = axes.plot([], [], linewidth=1, color='lightblue')
neptuneline, = axes.plot([], [], linewidth=1, color='steelblue')

text = axes.text(-lim+1, lim-2, lim-1, fontsize=15)

axes.legend()

times = np.linspace(0, noutputs*h, noutputs) # for displaying time elapsed in plot

def animate(i):
    sun.set_data(W_total[i,0,0], W_total[i,0,1])
    
    mercury.set_data(W_total[i,1,0], W_total[i,1,1])
    venus.set_data(W_total[i,2,0], W_total[i,2,1])
    earth.set_data(W_total[i,3,0], W_total[i,3,1])
    mars.set_data(W_total[i,4,0], W_total[i,4,1])
    jupiter.set_data(W_total[i,5,0], W_total[i,5,1])
    saturn.set_data(W_total[i,6,0], W_total[i,6,1])
    uranus.set_data(W_total[i,7,0], W_total[i,7,1])
    neptune.set_data(W_total[i,8,0], W_total[i,8,1])
    
    mercuryline.set_data(W_total[0:i,1,0], W_total[0:i,1,1])
    venusline.set_data(W_total[0:i,2,0], W_total[0:i,2,1])
    earthline.set_data(W_total[0:i,3,0], W_total[0:i,3,1])
    marsline.set_data(W_total[0:i,4,0], W_total[0:i,4,1])
    jupiterline.set_data(W_total[0:i,5,0], W_total[0:i,5,1])
    saturnline.set_data(W_total[0:i,6,0], W_total[0:i,6,1])
    uranusline.set_data(W_total[0:i,7,0], W_total[0:i,7,1])
    neptuneline.set_data(W_total[0:i,8,0], W_total[0:i,8,1])
    
    text.set_text('{:.2f} Years'.format(times[i]))
    return sun, mercury, venus, earth, mars, jupiter, saturn, uranus, neptune, mercuryline, venusline, earthline, marsline, jupiterline, saturnline, uranusline, neptuneline, text
    
im_ani = FuncAnimation(fig, animate, frames=noutputs, interval=1, blit=True)
# %%
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

f = f'vid/solarsystemlarge.mp4' 
writervideo = animation.FFMpegWriter(fps=40) # ffmpeg must be installed
im_ani.save(f, writer=writervideo)

# %%

def time(m, v):
    return 0.5 * m * v
    
def velocity(mi, mj, ri, rj):
    return G * mi * mj / np.linalg.norm(ri - rj)

def energy(m, w, num):
    v = np.sqrt((w[:,3]**2) + w[:,4]**2 + w[:,5]**2)
    ekini = time(m, v)
    epot = 0
    
    for j in range(len(W)):
        if j != num:
            epot_int = velocity(m, M[j], np.sqrt(x[:,num]**2 + y[:,num]**2), np.sqrt(x[:,j]**2 + y[:,j]**2))
        if j == num:
            epot_int = 0
        
        epot = epot + epot_int
        
    return ekini + epot_int

total_energy = np.sum([energy(M[i], W_total[:,i], i) for i in range(0, len(W))], axis=0)

plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, h*noutputs, noutputs), total_energy)
plt.xlabel('time (yr)')
plt.ylabel(r'energy (M$_o$ AU$^2$ yr$^{-2}$)')