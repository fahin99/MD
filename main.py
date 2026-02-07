import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#Parameters
N = 100
rho = 0.8
L = (N / rho)**(1/3)
dt = 0.019
steps = 500

sigma = 1.0
epsilon = 1.0
mass = 1.0

#Initialization
def init_positions(N, L):
    n = int(np.ceil(N**(1/3)))
    coords = np.linspace(0, L, n, endpoint=False)
    pos = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1,3)
    return pos[:N]

def init_velocities(N, T=1.0):
    v = np.random.normal(0, np.sqrt(T/mass), (N,3))
    v -= np.mean(v, axis=0)
    return v

#Force Calculation
def lj_force(pos, L):
    N = len(pos)
    forces = np.zeros_like(pos)
    U = 0.0
    for i in range(N):
        for j in range(i+1, N):
            rij = pos[i] - pos[j]
            rij -= L * np.round(rij/L)  # minimal image
            r2 = np.dot(rij, rij)
            if r2 < (3*sigma)**2:
                inv_r2 = 1.0 / r2
                inv_r6 = inv_r2**3
                inv_r12 = inv_r6**2
                f = 24*epsilon*inv_r2*(2*inv_r12 - inv_r6) * rij
                forces[i] += f
                forces[j] -= f
                U += 4*epsilon*(inv_r12 - inv_r6)
    return forces, U

#Velocity-Verlet
def velocity_verlet(pos, vel, L, dt):
    forces, U = lj_force(pos, L)
    pos += vel*dt + 0.5*forces/mass*dt**2
    pos %= L
    new_forces, U_new = lj_force(pos, L)
    vel += 0.5*(forces + new_forces)/mass*dt
    K = 0.5 * mass * np.sum(vel**2)
    return pos, vel, K + U_new


#Simulation Setup
pos = init_positions(N, L)
vel = init_velocities(N, T=1.0)

#Animation Loop
fig = plt.figure(figsize=(12, 5))
ax_sim = fig.add_subplot(121, projection="3d")
ax_energy = fig.add_subplot(122)
speeds = np.linalg.norm(vel, axis=1)
scat = ax_sim.scatter(pos[:,0], pos[:,1], pos[:,2],c=speeds,cmap="plasma",s=30)
cbar = fig.colorbar(scat, ax=ax_sim, orientation="vertical", pad=0.1)
cbar.set_label("Particle Speed (Energy Intensity)")

ax_sim.set_xlim([0, L])
ax_sim.set_ylim([0, L])
ax_sim.set_zlim([0, L])
ax_sim.set_xlabel("x")
ax_sim.set_ylabel("y")
ax_sim.set_zlabel("z")

time_data = []
energy_data = []
energy_line, = ax_energy.plot([], [], lw=2)
ax_energy.set_xlabel("Time")
ax_energy.set_ylabel("Total Energy")
ax_energy.set_title("Energy Conservation")
ax_energy.set_xlim(0, steps * dt)

def update(frame):
    global pos, vel
    pos, vel, E=velocity_verlet(pos, vel, L, dt)
    #Store data
    t = frame * dt
    time_data.append(t)
    energy_data.append(E)
    #Update scatter
    speeds = np.linalg.norm(vel, axis=1)
    scat._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
    scat.set_array(speeds)
    #Rotate camera slightly
    ax_sim.view_init(elev=25, azim=0.4 * frame)
    #Update energy plot
    energy_line.set_data(time_data, energy_data)
    ax_energy.relim()
    ax_energy.autoscale_view()
    return scat, energy_line


ani = FuncAnimation(fig,update,frames=steps,interval=60,blit=False)
plt.tight_layout()
plt.show()