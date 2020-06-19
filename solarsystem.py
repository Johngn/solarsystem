# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
from timeit import default_timer as timed
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d

G = 4*np.pi**2                  # gravitational constant - 4pi when distance is in AU and time is in years
t = Time.now()                  # gets current time - required for astropy to get the locations of all planets at that time
noutputs = 1500                 # number of outputs
h = 0.01                        # integration timestep
totaltime = noutputs*h          # total time of simulation in years

# names of bodies for astropy to search
bodies = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']

# get_body_barycentric_posvel finds the position and velocity of solar system bodies from database
positions = [get_body_barycentric_posvel(bodies[i], t)[0].xyz for i in range(len(bodies))]
velocities = [get_body_barycentric_posvel(bodies[i], t)[1].xyz*365.25 for i in range(len(bodies))]

# masses of solar system bodies in solar masses
M = np.array([1, 1.66e-7, 2.45e-6, 5.97e-6, 3.23e-7, 9.55e-4, 2.86e-4, 4.37e-5, 5.15e-5])
W0 = np.hstack((positions, velocities)) # create array of positions and velocities together for integrator
# %%
# this section creates a black hole and includes it in the simulation - optional
BHM = 0.5           # mass of black hole in solar masses
BHW = np.reshape(np.array([5,20,5,-1,-1,-0.5]), (1,6)) # x,y,z and vx,vy,vz of black hole

M = np.append(M, BHM)  # append black hole mass to all masses
W0 = np.vstack((W0, BHW)) # append position and velocity of black hole to other bodies
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
W = rungekutta(W0) # 6 dimensional phase space for each particle at each timestep
print(timed()-timer)

# %%
lim = 6
fig, ax = plt.subplots(1, figsize=(10, 10))

ax.set_xlabel('distance from sun (AU)')
ax.set_ylabel('distance from sun (AU)')
ax.set_xlim([-lim,lim])
ax.set_ylim([-lim,lim])

sun, = ax.plot([], [], 'o', color='yellow', markersize=5)
bh, = ax.plot([], [], 'o', label='Black hole', color='black', markersize=15)

merc, = ax.plot([], [], 'o', label="Mercury", color='plum', markersize=5)
ven, = ax.plot([], [], 'o', label="Venus", color='mediumaquamarine', markersize=8)
earth, = ax.plot([], [], 'o', label="Earth", color='deepskyblue', markersize=8)
mars, = ax.plot([], [], 'o', label="Mars", color='lightcoral', markersize=8)
jup, = ax.plot([], [], 'o', label="Jupiter", color='darkgoldenrod', markersize=12)
sat, = ax.plot([], [], 'o', label="Saturn", color='darkkhaki', markersize=12)
urn, = ax.plot([], [], 'o', label="Uranus", color='lightblue', markersize=10)
nep, = ax.plot([], [], 'o', label="Neptune", color='steelblue', markersize=10)


sunl, = ax.plot([], [], linewidth=1, color='yellow')
bhl, = ax.plot([], [], linewidth=1, color='black')

mercl, = ax.plot([], [], linewidth=1, color='plum')
venl, = ax.plot([], [], linewidth=1, color='mediumaquamarine')
earthl, = ax.plot([], [], linewidth=1, color='deepskyblue')
marsl, = ax.plot([], [], linewidth=1, color='lightcoral')
jupl, = ax.plot([], [], linewidth=1, color='darkgoldenrod')
satl, = ax.plot([], [], linewidth=1, color='darkkhaki')
urnl, = ax.plot([], [], linewidth=1, color='lightblue')
nepl, = ax.plot([], [], linewidth=1, color='steelblue')

text = ax.text(-lim+1, lim-1, s='', fontsize=15)
# ax.grid()
ax.legend()

times = np.linspace(0, noutputs*h, noutputs)+2020 # for displaying time elapsed in plot

def animate(i):
    sun.set_data(W[i,0,0], W[i,0,1])
    bh.set_data(W[i,9,0], W[i,9,1])
    
    merc.set_data(W[i,1,0], W[i,1,1])
    ven.set_data(W[i,2,0], W[i,2,1])
    earth.set_data(W[i,3,0], W[i,3,1])
    mars.set_data(W[i,4,0], W[i,4,1])
    jup.set_data(W[i,5,0], W[i,5,1])
    sat.set_data(W[i,6,0], W[i,6,1])
    urn.set_data(W[i,7,0], W[i,7,1])
    nep.set_data(W[i,8,0], W[i,8,1])
    
    sunl.set_data(W[0:i,0,0], W[0:i,0,1])
    bhl.set_data(W[0:i,9,0], W[0:i,9,1])
    
    mercl.set_data(W[0:i,1,0], W[0:i,1,1])
    venl.set_data(W[0:i,2,0], W[0:i,2,1])
    earthl.set_data(W[0:i,3,0], W[0:i,3,1])
    marsl.set_data(W[0:i,4,0], W[0:i,4,1])
    jupl.set_data(W[0:i,5,0], W[0:i,5,1])
    satl.set_data(W[0:i,6,0], W[0:i,6,1])
    urnl.set_data(W[0:i,7,0], W[0:i,7,1])
    nepl.set_data(W[0:i,8,0], W[0:i,8,1])
    
    text.set_text('{:.0f}'.format(times[i]))
    
    return sun,merc,ven,earth,mars,jup,sat,urn,nep,mercl,venl,earthl,marsl,jupl,satl,urnl,nepl,bh,sunl,bhl,text
    
im_ani = animation.FuncAnimation(fig, animate, frames=noutputs, interval=1, blit=True)
# %%
lim = 6
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('distance from sun (AU)')
ax.set_ylabel('distance from sun (AU)')
ax.set_zlabel('distance from sun (AU)')
ax.set_xlim([-lim,lim])
ax.set_ylim([-lim,lim])
ax.set_zlim([-lim,lim])

sun, = ax.plot([],[],[], 'o', color='yellow', markersize=5)
bh, = ax.plot([],[],[], 'o', label="Black hole", color='black', markersize=15)

merc, = ax.plot([],[],[], 'o', label="Mercury", color='plum', markersize=4)
ven, = ax.plot([],[],[], 'o', label="Venus", color='mediumaquamarine', markersize=6)
earth, = ax.plot([],[],[], 'o', label="Earth", color='deepskyblue', markersize=6)
mars, = ax.plot([],[],[], 'o', label="Mars", color='lightcoral', markersize=6)
jup, = ax.plot([],[],[], 'o', label="Jupiter", color='darkgoldenrod', markersize=10)
sat, = ax.plot([],[],[], 'o', label="Saturn", color='darkkhaki', markersize=10)
urn, = ax.plot([],[],[], 'o', label="Uranus", color='lightblue', markersize=8)
nep, = ax.plot([],[],[], 'o', label="Neptune", color='steelblue', markersize=8)


sunl, = ax.plot([],[],[], linewidth=0.5, color='yellow')
bhl, = ax.plot([],[],[], linewidth=0.5, color='black')

mercl, = ax.plot([],[],[], linewidth=0.5, color='plum')
venl, = ax.plot([],[],[], linewidth=0.5, color='mediumaquamarine')
earthl, = ax.plot([],[],[], linewidth=0.5, color='deepskyblue')
marsl, = ax.plot([],[],[], linewidth=0.5, color='lightcoral')
jupl, = ax.plot([],[],[], linewidth=0.5, color='darkgoldenrod')
satl, = ax.plot([],[],[], linewidth=0.5, color='darkkhaki')
urnl, = ax.plot([],[],[], linewidth=0.5, color='lightblue')
nepl, = ax.plot([],[],[], linewidth=0.5, color='steelblue')

text = ax.text(-lim+1, lim-.0, lim-1, s='', fontsize=15)
ax.legend()

times = np.linspace(0, noutputs*h, noutputs)+2020 # for displaying time elapsed in plot

def animate(i):
    sun.set_data(W[i,0,0], W[i,0,1])
    bh.set_data(W[i,9,0], W[i,9,1])
    
    merc.set_data(W[i,1,0], W[i,1,1])
    ven.set_data(W[i,2,0], W[i,2,1])
    earth.set_data(W[i,3,0], W[i,3,1])
    mars.set_data(W[i,4,0], W[i,4,1])
    jup.set_data(W[i,5,0], W[i,5,1])
    sat.set_data(W[i,6,0], W[i,6,1])
    urn.set_data(W[i,7,0], W[i,7,1])
    nep.set_data(W[i,8,0], W[i,8,1])
    
    sun.set_3d_properties(W[i,0,2])
    bh.set_3d_properties(W[i,9,2])
    
    merc.set_3d_properties(W[i,1,2])
    ven.set_3d_properties(W[i,2,2])
    earth.set_3d_properties(W[i,3,2])
    mars.set_3d_properties(W[i,4,2])
    jup.set_3d_properties(W[i,5,2])
    sat.set_3d_properties(W[i,6,2])
    urn.set_3d_properties(W[i,7,2])
    nep.set_3d_properties(W[i,8,2])
    
    sunl.set_data(W[0:i,0,0], W[0:i,0,1])
    bhl.set_data(W[0:i,9,0], W[0:i,9,1])
    
    mercl.set_data(W[0:i,1,0], W[0:i,1,1])
    venl.set_data(W[0:i,2,0], W[0:i,2,1])
    earthl.set_data(W[0:i,3,0], W[0:i,3,1])
    marsl.set_data(W[0:i,4,0], W[0:i,4,1])
    jupl.set_data(W[0:i,5,0], W[0:i,5,1])
    satl.set_data(W[0:i,6,0], W[0:i,6,1])
    urnl.set_data(W[0:i,7,0], W[0:i,7,1])
    nepl.set_data(W[0:i,8,0], W[0:i,8,1])
    
    sunl.set_3d_properties(W[0:i,0,2])
    bhl.set_3d_properties(W[0:i,9,2])
    
    mercl.set_3d_properties(W[0:i,1,2])
    venl.set_3d_properties(W[0:i,2,2])
    earthl.set_3d_properties(W[0:i,3,2])
    marsl.set_3d_properties(W[0:i,4,2])
    jupl.set_3d_properties(W[0:i,5,2])
    satl.set_3d_properties(W[0:i,6,2])
    urnl.set_3d_properties(W[0:i,7,2])
    nepl.set_3d_properties(W[0:i,8,2])
    
    text.set_text('{:.0f}'.format(times[i]))
    
    return sun,merc,ven,earth,mars,jup,sat,urn,nep,mercl,venl,earthl,marsl,jupl,satl,urnl,nepl,bh,sunl,bhl,text
    
im_ani = animation.FuncAnimation(fig, animate, frames=noutputs, interval=1, blit=True)
# %%
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

writervideo = animation.FFMpegWriter(fps=60) # ffmpeg must be installed
im_ani.save('solarsysteminnerBH3D.mp4', writer=writervideo)

# %%
# this code calulates the total energy of the system - should be relatively constant
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