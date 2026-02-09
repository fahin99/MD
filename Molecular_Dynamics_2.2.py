import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import stats

kB = 1.0  # Boltzmann constant (reduced units)

CONFIG = {
    'N': 108,           # Number of particles (perfect cube)
    'rho': 0.5,         # Number density
    'T_target': 1.0,    # Target temperature
    'dt': 0.005,        # REDUCED time step for stability
    'steps': 2000,      # More steps to compensate
    'epsilon': 1.0,     # LJ energy scale
    'sigma': 1.0,       # LJ length scale
    'rc': 2.5,          # Cutoff radius (in sigma units)
    'Q': 1.0,           # MUCH STRONGER thermostat coupling (was 10.0)
    'tau': 0.1,         # Alternative: relaxation time for thermostat
}

class MDSystem:
    """Molecular Dynamics System with Lennard-Jones potential and Nosé-Hoover thermostat"""
    
    def __init__(self, config):
        self.N = config['N']
        self.rho = config['rho']
        self.L = (self.N / self.rho)**(1/3)
        self.mass = 1.0

        self.eps = config['epsilon']
        self.sig = config['sigma']
        self.rc = config['rc'] * self.sig
        self.rc2 = self.rc**2

        inv_rc6 = (self.sig / self.rc)**6
        self.U_shift = 4 * self.eps * (inv_rc6**2 - inv_rc6)

        self.T_target = config['T_target']
        # Stronger thermostat: Q relates to tau as Q = dof * kB * T * tau^2
        self.tau = config.get('tau', 0.1)
        dof = 3 * self.N
        self.Q = dof * kB * self.T_target * self.tau**2
        self.xi = 0.0
        self.xi_dot = 0.0  # Track thermostat velocity

        self.pos = self._init_fcc_lattice()
        self.vel = self._init_velocities()
        self._remove_com_motion()
        
    def _init_fcc_lattice(self):
        """Initialize simple cubic lattice with optimal spacing"""
        n = int(np.ceil(self.N**(1/3)))
        
        # Ensure minimum distance of ~1.2 sigma (safe distance)
        min_spacing = 1.3 * self.sig  # Increased safety margin
        required_L = n * min_spacing
        
        if self.L < required_L:
            print(f"Warning: Box too small for density {self.rho}. Adjusting...")
            self.L = required_L
            self.rho = self.N / self.L**3
        
        a = self.L / n  # Lattice constant
        
        positions = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    pos = np.array([i, j, k]) * a + a/2  # Center in cells
                    # Smaller random displacement
                    pos += (np.random.random(3) - 0.5) * 0.005 * a
                    positions.append(pos)
                    if len(positions) >= self.N:
                        result = np.array(positions[:self.N])
                        result = result % self.L
                        return result
        
        return np.array(positions[:self.N]) % self.L
    
    def _init_velocities(self):
        """Initialize velocities from Maxwell-Boltzmann distribution"""
        # Start at even LOWER temperature
        T_init = self.T_target * 0.1
        v = np.random.normal(0, np.sqrt(T_init / self.mass), (self.N, 3))
        return v
    
    def _remove_com_motion(self):
        """Remove center-of-mass motion"""
        self.vel -= np.mean(self.vel, axis=0)
    
    def _apply_pbc(self, r):
        """Apply periodic boundary conditions with minimum image convention"""
        return r - self.L * np.round(r / self.L)
    
    def compute_forces(self):
        """Compute Lennard-Jones forces with cutoff"""
        forces = np.zeros_like(self.pos)
        U_pot = 0.0
        virial = 0.0
        
        # Stricter minimum distance to prevent singularities
        r_min2 = (0.8 * self.sig)**2
        
        for i in range(self.N - 1):
            dr = self.pos[i+1:] - self.pos[i]
            dr = self._apply_pbc(dr)
            r2 = np.sum(dr**2, axis=1)
            
            # Apply cutoff and minimum distance
            mask = (r2 < self.rc2) & (r2 > r_min2)
            
            if not np.any(mask):
                continue
            
            r2_masked = r2[mask]
            dr_masked = dr[mask]
            
            inv_r2 = self.sig**2 / r2_masked
            inv_r6 = inv_r2**3
            inv_r12 = inv_r6**2
            
            # Standard LJ force
            f_mag = 24 * self.eps * inv_r2 * (2*inv_r12 - inv_r6)
            
            # More aggressive force capping
            F_MAX = 50.0  # Reduced from 100
            f_mag = np.clip(f_mag, -F_MAX, F_MAX)
            
            f_vec = f_mag[:, np.newaxis] * dr_masked
            
            forces[i] += np.sum(f_vec, axis=0)
            j_indices = np.arange(i+1, self.N)[mask]
            for idx, f in zip(j_indices, f_vec):
                forces[idx] -= f
            
            # Potential energy
            U_pot += np.sum(4*self.eps*(inv_r12 - inv_r6) - self.U_shift)
            
            # Virial for pressure
            virial += np.sum(f_mag * np.sqrt(r2_masked))
            
        return forces, U_pot, virial
    
    def step(self, dt):
        """Velocity Verlet with improved Nosé-Hoover thermostat"""
        # Compute initial forces
        forces, U, virial = self.compute_forces()
        
        # Current kinetic energy and temperature
        K = 0.5 * self.mass * np.sum(self.vel**2)
        dof = 3 * self.N
        T_inst = 2 * K / (dof * kB)
        
        # Update thermostat variable with proper coupling
        # xi_dot = (T_inst - T_target) / (tau^2 * T_target)
        self.xi += dt * (T_inst - self.T_target) / (self.tau**2 * self.T_target)
        
        # Dampen thermostat oscillations
        damping = 0.95
        self.xi *= damping
        
        # Velocity Verlet half-step
        acc = forces / self.mass - self.xi * self.vel
        self.vel += 0.5 * dt * acc
        
        # Position update
        self.pos += self.vel * dt
        self.pos %= self.L  # PBC
        
        # Recompute forces at new positions
        forces, U, virial = self.compute_forces()
        
        # Update temperature
        K = 0.5 * self.mass * np.sum(self.vel**2)
        T_inst = 2 * K / (dof * kB)
        
        # Update thermostat again
        self.xi += dt * (T_inst - self.T_target) / (self.tau**2 * self.T_target)
        self.xi *= damping
        
        # Velocity Verlet second half-step
        acc = forces / self.mass - self.xi * self.vel
        self.vel += 0.5 * dt * acc
        
        # Velocity capping as last resort
        V_MAX = 5.0  # Reduced from 10.0
        speed = np.linalg.norm(self.vel, axis=1)
        speed_max = np.max(speed)
        if speed_max > V_MAX:
            print(f"Warning: Capping velocities (max speed: {speed_max:.2f})")
            self.vel *= V_MAX / speed_max
        
        # Check for NaN
        if np.any(np.isnan(self.pos)) or np.any(np.isnan(self.vel)):
            raise ValueError("NaN detected in simulation - numerical instability!")
        
        # Final temperature and pressure
        T = self.temperature()
        P = self.pressure(T, virial)
        
        # Recompute kinetic energy after velocity rescaling
        K = 0.5 * self.mass * np.sum(self.vel**2)
        
        return K + U, T, P, K, U
    
    def temperature(self):
        """Instantaneous temperature"""
        dof = 3 * self.N
        K = 0.5 * self.mass * np.sum(self.vel**2)
        return 2 * K / (dof * kB)
    
    def pressure(self, T, virial):
        """Instantaneous pressure using virial theorem"""
        V = self.L**3
        return self.N * kB * T / V + virial / (3 * V)

class Statistics:
    """Track and analyze simulation statistics"""
    
    def __init__(self, T_target):
        self.T_target = T_target
        self.reset()
    
    def reset(self):
        self.energies = []
        self.temps = []
        self.pressures = []
        self.kinetic = []
        self.potential = []
        
    def add(self, E, T, P, K, U):
        self.energies.append(E)
        self.temps.append(T)
        self.pressures.append(P)
        self.kinetic.append(K)
        self.potential.append(U)
    
    def get_stats(self, equilibration=0.3):
        """Compute statistics after equilibration"""
        start = int(len(self.temps) * equilibration)
        
        if len(self.temps) <= start:
            return None
        
        T_eq = self.temps[start:]
        E_eq = self.energies[start:]
        P_eq = self.pressures[start:]
        
        stats_dict = {
            'T_mean': np.mean(T_eq),
            'T_std': np.std(T_eq),
            'T_error': abs(np.mean(T_eq) - self.T_target) / self.T_target * 100,
            'E_mean': np.mean(E_eq),
            'E_std': np.std(E_eq),
            'E_drift': (E_eq[-1] - E_eq[0]) / abs(E_eq[0]) * 100 if E_eq[0] != 0 else 0,
            'P_mean': np.mean(P_eq),
            'P_std': np.std(P_eq),
        }
        
        return stats_dict

class Visualizer:
    """Real-time visualization with Maxwell-Boltzmann validation"""
    
    def __init__(self, system, config):
        self.system = system
        self.dt = config['dt']
        self.stats = Statistics(config['T_target'])
        
        # Setup figure
        self.fig = plt.figure(figsize=(18, 10))
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 3D particle view
        self.ax_3d = self.fig.add_subplot(gs[:, 0], projection='3d')
        
        # Energy plot
        self.ax_energy = self.fig.add_subplot(gs[0, 1])
        
        # Temperature & Pressure
        self.ax_temp = self.fig.add_subplot(gs[1, 1])
        self.ax_press = self.ax_temp.twinx()
        
        # Maxwell-Boltzmann distribution
        self.ax_mb = self.fig.add_subplot(gs[0, 2])
        
        # Statistics text
        self.ax_stats = self.fig.add_subplot(gs[1, 2])
        self.ax_stats.axis('off')
        
        self._setup_plots()
        
    def _setup_plots(self):
        """Initialize all plot elements"""
        # 3D scatter
        pos = self.system.pos
        speeds = np.linalg.norm(self.system.vel, axis=1)
        self.scat = self.ax_3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                                       c=speeds, cmap='plasma', s=30, alpha=0.8)
        self.ax_3d.set_xlim(0, self.system.L)
        self.ax_3d.set_ylim(0, self.system.L)
        self.ax_3d.set_zlim(0, self.system.L)
        self.ax_3d.set_title('Particle Positions')
        self.fig.colorbar(self.scat, ax=self.ax_3d, label='Speed', shrink=0.7)
        self.ax_3d.set_xticklabels([])
        self.ax_3d.set_yticklabels([])
        self.ax_3d.set_zticklabels([])
        
        # Energy lines
        self.line_E, = self.ax_energy.plot([], [], 'r-', lw=2, label='Total')
        self.line_K, = self.ax_energy.plot([], [], 'b--', lw=1, alpha=0.7, label='Kinetic')
        self.line_U, = self.ax_energy.plot([], [], 'g--', lw=1, alpha=0.7, label='Potential')
        self.ax_energy.set_xlabel('Time')
        self.ax_energy.set_ylabel('Energy')
        self.ax_energy.set_title('Energy Conservation')
        self.ax_energy.legend(loc='upper right')
        self.ax_energy.grid(True, alpha=0.3)
        
        # Temperature & Pressure
        self.line_T, = self.ax_temp.plot([], [], 'b-', lw=2, label='Temperature')
        self.ax_temp.axhline(y=self.system.T_target, color='b', ls='--', 
                            alpha=0.5, label='Target T')
        self.ax_temp.set_xlabel('Time')
        self.ax_temp.set_ylabel('Temperature', color='b')
        self.ax_temp.tick_params(axis='y', labelcolor='b')
        self.ax_temp.grid(True, alpha=0.3)
        
        self.line_P, = self.ax_press.plot([], [], 'orange', lw=2, label='Pressure')
        self.ax_press.set_ylabel('Pressure', color='orange')
        self.ax_press.tick_params(axis='y', labelcolor='orange')
        
        # MB distribution
        self.ax_mb.set_xlabel('Speed')
        self.ax_mb.set_ylabel('Probability Density')
        self.ax_mb.set_title('Maxwell-Boltzmann Distribution')
        self.ax_mb.grid(True, alpha=0.3)
        
        self.time_data = []
        
    def update(self, frame):
        """Update animation frame"""
        try:
            E, T, P, K, U = self.system.step(self.dt)
        except ValueError as e:
            print(f"Simulation failed: {e}")
            return
            
        self.stats.add(E, T, P, K, U)
        
        # Update 3D positions
        pos = self.system.pos
        speeds = np.linalg.norm(self.system.vel, axis=1)
        self.scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        self.scat.set_array(speeds)
        self.ax_3d.view_init(elev=20, azim=frame * 0.5)
        
        # Update time series
        time = frame * self.dt
        self.time_data.append(time)
        
        self.line_E.set_data(self.time_data, self.stats.energies)
        self.line_K.set_data(self.time_data, self.stats.kinetic)
        self.line_U.set_data(self.time_data, self.stats.potential)
        self.line_T.set_data(self.time_data, self.stats.temps)
        self.line_P.set_data(self.time_data, self.stats.pressures)
        
        # Auto-scale
        self.ax_energy.relim()
        self.ax_energy.autoscale_view()
        self.ax_temp.relim()
        self.ax_temp.autoscale_view()
        self.ax_press.relim()
        self.ax_press.autoscale_view()
        
        # Update Maxwell-Boltzmann every 20 frames
        if frame % 20 == 0:
            self._update_mb_distribution(T)
            self._update_statistics(frame)
        
        return (self.scat, self.line_E, self.line_K, self.line_U, 
                self.line_T, self.line_P)
    
    def _update_mb_distribution(self, T):
        """Update Maxwell-Boltzmann distribution comparison"""
        self.ax_mb.clear()
        
        speeds = np.linalg.norm(self.system.vel, axis=1)
        
        # Check for NaN or invalid speeds
        if np.any(np.isnan(speeds)) or np.any(np.isinf(speeds)):
            self.ax_mb.text(0.5, 0.5, 'Invalid data - simulation unstable',
                           transform=self.ax_mb.transAxes, ha='center', va='center',
                           fontsize=12, color='red')
            return
        
        max_speed = np.max(speeds)
        if max_speed == 0 or not np.isfinite(max_speed):
            return
        
        # Histogram of actual speeds
        counts, bins, _ = self.ax_mb.hist(speeds, bins=30, density=True, 
                                          alpha=0.6, color='blue', 
                                          label='MD Simulation')
        
        # Theoretical MB distribution
        v = np.linspace(0, max_speed*1.1, 200)
        m, kB = self.system.mass, 1.0
        if T > 0:
            f_theory = 4*np.pi * (m/(2*np.pi*kB*T))**1.5 * v**2 * np.exp(-m*v**2/(2*kB*T))
        else:
            f_theory = np.zeros_like(v)
        
        self.ax_mb.plot(v, f_theory, 'r-', lw=2, label='MB Theory')
        
        # Chi-squared test
        if T > 0:
            bin_centers = (bins[:-1] + bins[1:]) / 2
            f_expected = 4*np.pi * (m/(2*np.pi*kB*T))**1.5 * bin_centers**2 * \
                         np.exp(-m*bin_centers**2/(2*kB*T))
            f_expected *= np.sum(counts) / np.sum(f_expected)  # Normalize
            
            # Avoid division by zero
            mask = f_expected > 0.01
            if np.sum(mask) > 1:
                chi2 = np.sum((counts[mask] - f_expected[mask])**2 / f_expected[mask])
                dof = np.sum(mask) - 1
                p_value = 1 - stats.chi2.cdf(chi2, dof)
                self.ax_mb.text(0.95, 0.95, f'χ²/dof = {chi2/dof:.2f}\np = {p_value:.3f}',
                               transform=self.ax_mb.transAxes, ha='right', va='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.ax_mb.set_xlabel('Speed')
        self.ax_mb.set_ylabel('Probability Density')
        self.ax_mb.set_title('Maxwell-Boltzmann Validation')
        self.ax_mb.legend()
        self.ax_mb.grid(True, alpha=0.3)
    
    def _update_statistics(self, frame):
        """Update statistics text"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        if frame < 50:  # Wait for some data
            return
        
        stats = self.stats.get_stats()
        if stats is None:
            return
        
        text = f"""SIMULATION STATISTICS (Equilibrated)
Current Step: {frame}
Temperature:
  Mean: {stats['T_mean']:.4f} ± {stats['T_std']:.4f}
  Target: {self.system.T_target:.4f}
  Error: {stats['T_error']:.2f}%
Energy:
  Mean: {stats['E_mean']:.4f} ± {stats['E_std']:.4f}
  Drift: {stats['E_drift']:.3f}%
Pressure:
  Mean: {stats['P_mean']:.4f} ± {stats['P_std']:.4f}
System:
  N = {self.system.N}
  ρ = {self.system.rho:.2f}
  L = {self.system.L:.2f}
  τ = {self.system.tau:.2f}
"""
        self.ax_stats.text(0.1, 0.9, text, transform=self.ax_stats.transAxes,
                          fontfamily='monospace', fontsize=9, va='top',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

if __name__ == '__main__':
    print("Initializing MD Simulation...")
    print(f"Particles: {CONFIG['N']}, Density: {CONFIG['rho']}, Target T: {CONFIG['T_target']}")
    
    system = MDSystem(CONFIG)
    print(f"Box size: {system.L:.3f}")
    print(f"Initial temperature: {system.temperature():.3f}")
    print(f"Thermostat time constant: {system.tau:.3f}")
    print(f"Thermostat mass Q: {system.Q:.3f}")
    
    forces, U, virial = system.compute_forces()
    K = 0.5 * system.mass * np.sum(system.vel**2)
    print(f"Initial energy: K={K:.3f}, U={U:.3f}, Total={K+U:.3f}")
    
    viz = Visualizer(system, CONFIG)
    
    print("\nStarting animation...")
    print("This should maintain T ≈ 1.0 with <5% error\n")
    
    ani = FuncAnimation(viz.fig, viz.update, frames=CONFIG['steps'],
                       interval=30, blit=False)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    final_stats = viz.stats.get_stats()
    if final_stats:
        for key, value in final_stats.items():
            print(f"{key:12s}: {value:10.4f}")