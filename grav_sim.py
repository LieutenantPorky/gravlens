import moderngl
import struct
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time

#Useful global variables

num_rays = 20
num_steps = 2000
grav_source = "dvec3(0)"
mass = 0.05
dt = 0.01
compute_code_name = "raytrace.glsl"

const_dict = {"%ray_num%": num_rays, "%grav_source%":grav_source, "%mass%":mass, "%dt%":dt}



#First, need to create an openGL context
ctx = moderngl.create_context(standalone=True)

#read the glsl code:
with open(compute_code_name, 'r') as f:
    compute_code = f.read()
# Since there are some useful constants that need to be defined at the start of the GLSL code, and we want to
# make them dynamic based on the python code, they're written as tokens that need to be replaced
    for key, value in const_dict.items():
        compute_code = compute_code.replace(key,str(value))


#bind the compute shader
compute_shader = ctx.compute_shader(compute_code)

#create a vector buffer to store ray info
# each ray will follow a series of timesteps, with each timestep being saved as [double4 pos, double4 dir].
# While having 4 entries for direction is superfluous, it's forced by how opengl treats alignment in buffers.
# This means the final 4 bytes of every ray can be either ignored or used to store some other useful scalar.

#function to create some test data
init_data_1 = []
for i in range(num_rays):

    t = ((np.pi/4) * (np.random.rand() - 0.5) + np.pi/2)
    p = ((np.pi) * np.random.rand() -np.pi/2)

    z = np.cos(t)
    y = np.sin(p)*np.sin(t)
    x = np.cos(p)*np.sin(t)
    init_data_1.append([-5,0,0,0,x,y,z,0])


init_data_2 = []
for i in range(num_rays):
    z = ((10) * np.random.rand() - 5)
    y = ((10) * np.random.rand() - 5)
    x = -10
    init_data_2.append([x,y,z,0,1,0,0,0])

init_data_3 = []
for p in np.linspace(-np.pi/2, np.pi/2, num_rays):

    z = 0
    y = np.sin(p)
    x = np.cos(p)
    init_data_3.append([-5,0,0,0,x,y,z,0])

init_data_4 = []
for p in np.linspace(0.15, np.pi/2, num_rays):

    z = 0
    y = np.sin(p)
    x = np.cos(p)
    init_data_4.append([-5,0,0,0,x,y,z,0])

# Cast the initial data to a continuous memory block as a double (float64)
init_bytes=np.array(init_data_4, dtype=np.double).tobytes()

# define a memory buffer with the data, and bind it to buffer 0 on the GPU
ray_buffer = ctx.buffer(init_bytes,dynamic=False)
ray_buffer.bind_to_storage_buffer(0)

#create an array to store ray positions at each timestep
ray_timeline = []

for i in range(num_steps):
    start = time.time()
    compute_shader.run(num_rays,1,1)
    end=time.time()
    after=np.frombuffer(ray_buffer.read(),dtype=np.double).reshape(-1,8)
    ray_timeline.append(after)

print("elapsed time = " + str(end-start))
#clean up after ourselves, and release the buffer to avoid memory leak
ray_buffer.release()

#get range for Shapiro delay
delay = np.moveaxis(ray_timeline[-1],0,-1)[3]
delayMax = max(delay)
delayMin = min(delay)


# put the rays in a numpy array, and format it so that it can easily be plotted
rays = np.moveaxis(np.array(ray_timeline), 0,-1)
print(np.moveaxis(rays, 0,-1).shape)
print()

# plot the rays in 3d

# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# for ray in rays:
#     ax.plot(ray[0],ray[1],ray[2], color=plt.cm.plasma((ray[3][-1]-delayMin)/(delayMax-delayMin)))
#
# from matplotlib.lines import Line2D
# custom_lines = [Line2D([0], [0], color=plt.cm.plasma(0.), lw=4),
#                 Line2D([0], [0], color=plt.cm.plasma(1.), lw=4)]
#
#
# ax.legend(custom_lines, ['Low delay', 'High delay'])
# ax.plot([0],[0],[0], 'o')

#plot the rays in 2d (overhead)

# fig, ax = plt.subplots()
# for ray in rays:
#     ax.plot(ray[0],ray[1], color=plt.cm.plasma((ray[3][-1]-delayMin)/(delayMax-delayMin)))
#     plt.xlabel(r"x distance $\left(\frac{G}{M}\right)$")
#     plt.ylabel(r"y distance $\left(\frac{G}{M}\right)$")
#
# from matplotlib.lines import Line2D
# custom_lines = [Line2D([0], [0], color=plt.cm.plasma(0.), lw=4),
#                 Line2D([0], [0], color=plt.cm.plasma(1.), lw=4)]
#
#
# ax.legend(custom_lines, ['Low delay', 'High delay'])
# plt.plot([0],[0], 'o')

#Plot an analysis of the deflection angle

theory = []
sim = []
for ray in rays:
    source = np.array([-5,0,0])
    observer = np.array([ray[0][-1], ray[1][-1], ray[2][-1]])

    initial_d = np.array([ray[4][0], ray[5][0], ray[6][0]])
    final_d = np.array([ray[4][-1], ray[5][-1], ray[6][-1]])
    sim_alpha = np.arccos(np.dot(initial_d, final_d))

    Dl = np.sqrt(np.sum(observer ** 2))
    theta = np.arccos(np.dot(observer,final_d)/Dl)
    alpha = 4 * mass / (theta * Dl)
    sim.append(sim_alpha)
    theory.append(alpha)

    print(theta, alpha, sim_alpha)

plt.plot(np.array(sim)*180/np.pi, np.array(sim)/np.array(theory))

plt.xlabel(r"Simulated deflection angle $\hat{\alpha}_{sim}$ (deg)")
plt.ylabel(r"$\hat{\alpha}_{sim}$ / $\hat{\alpha}_{theory}$")


plt.show()
