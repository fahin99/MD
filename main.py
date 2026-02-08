import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#ParticleSystem ->positions, velocities, and masses
#ForceField ->for lennard-jones forces
#Integrator ->velocity-verlet
#Simulation ->setup and animation loop

class ParticleSystem:
    def __init__(self, N, rho, mass=1.0):
        self.N=N
        self.mass=mass
        self.L=(N/rho)**(1/3)
        self.pos=self.init_pos()
        self.vel=self.init_vel()

    def init_pos(self):
        n=int(np.ceil(self.N**(1/3)))
        coords=np.linspace(0,self.L,n,endpoint=False)
        pos=np.array(np.meshgrid(coords,coords,coords)).T.reshape(-1,3)
        return pos[:self.N]
    
    def init_vel(self, T=1.0):
        v=np.random.normal(0, np.sqrt(T/self.mass),(self.N, 3))
        v-=np.mean(v, axis=0)
        return v
    
#this gives us system.pos, .vel and .L

class ForceField:
    def __init__(self, epsilon=1.0, sigma=1.0):
        self.epsilon=epsilon
        self.sigma=sigma

    def compute(self, pos, L):
        N=len(pos)
        forces=np.zeros_like(pos)
        U=0.0
        for i in range(N):
            for j in range(i+1, N):
                r_ij=pos[i]-pos[j]
                r_ij -=L*np.round(r_ij/L)
                r2=np.dot(r_ij, r_ij)
                
                if r2<(3*self.sigma)**2:
                    inv_r2=1.0/r2
                    inv_r6=inv_r2**3
                    inv_r12=inv_r6**2
                    f =24*self.epsilon*inv_r2*(2*inv_r12-inv_r6)*r_ij
                    forces[i]+=f
                    forces[j]-=f
                    U+=4*self.epsilon*(inv_r12-inv_r6)
        return forces, U
    
#this gives us system.forces and overall U, using LJ potential theory

class Integrator:
    def __init__(self, dt):
        self.dt=dt

    def step(self, system, force_field):
        forces,U=force_field.compute(system.pos, system.L)
        system.pos +=system.vel*self.dt + 0.5*forces/system.mass*self.dt**2
        system.pos %=system.L
        new_forces, U_new =force_field.compute(system.pos, system.L)

        system.vel +=0.5*(forces+new_forces)/system.mass*self.dt
        K =0.5*system.mass*np.sum(system.vel**2)
        return U_new+K
    
#this helps us to update the pos and vel of the system at each time step as well as calc the net energy

class Simulation:
    def __init__(self, N=100, rho=0.8, dt=0.0005):
        self.system=ParticleSystem(N, rho)
        self.force_field=ForceField()
        self.integrator=Integrator(dt)
    
    def step(self):
        return self.integrator.step(self.system, self.force_field)

#plotting
steps=500
sim=Simulation()
fig=plt.figure(figsize=(12,5))
ax_sim =fig.add_subplot(121, projection='3d')
ax_energy=fig.add_subplot(122)

pos=sim.system.pos
vel=sim.system.vel
L=sim.system.L
speeds=np.linalg.norm(vel, axis=1)
scat =ax_sim.scatter(pos[:,0], pos[:,1], pos[:,2], c=speeds, cmap='plasma', s=30)
cbar=fig.colorbar(scat, ax=ax_sim)
cbar.set_label('Particle Speed')

ax_sim.set_xlim([0,L])
ax_sim.set_ylim([0,L])
ax_sim.set_zlim([0,L])
time_data=[]
energy_data=[]
energy_line, = ax_energy.plot([],[],'r-')

ax_energy.set_xlabel('Time')
ax_energy.set_ylabel('Net Energy')

#animation
def update(frame):
    E=sim.step()
    pos=sim.system.pos
    vel=sim.system.vel

    if frame%10==0:
        mean_speed=np.mean(np.linalg.norm(vel, axis=1))
        print(f"Step {frame}, Time {frame*sim.integrator.dt:.2f}, Energy {E:.3f}, Mean Speed {mean_speed:.3f}")
    speeds=np.linalg.norm(vel, axis=1)
    scat._offsets3d =(pos[:,0], pos[:,1], pos[:,2])
    scat.set_array(speeds)
    ax_sim.view_init(elev=25, azim=frame*0.4)

    time_data.append(frame*sim.integrator.dt)
    energy_data.append(E)
    energy_line.set_data(time_data, energy_data)
    ax_energy.relim()
    ax_energy.autoscale_view()

    return scat, energy_line

ani =FuncAnimation(fig,update,frames=steps,interval=60)
plt.tight_layout()
plt.show()
