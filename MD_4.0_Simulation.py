"""
MD Simulation with Trained PINN
Place trained model files in the same directory and run this
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os

# Disable GPU warnings for inference
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

kB = 1.0  # Boltzmann constant

CONFIG = {
    'N': 108,
    'rho': 0.5,
    'T_target': 1.0,
    'dt': 0.005,
    'steps': 3000,
    'epsilon': 1.0,
    'sigma': 1.0,
    'rc': 2.5,
    'tau': 0.1,
    
    # ML transition parameters
    'ml_transition_start': 1000,  # Start blending at step 1000
    'ml_transition_end': 1500,    # Full PINN by step 1500
}


class EnhancedPINN(keras.Model):
    """
    Enhanced Physics-Informed Neural Network
    MUST MATCH TRAINING ARCHITECTURE EXACTLY
    """
    
    def __init__(self, hidden_dims=[256, 512, 512, 256], dropout_rate=0.1):
        super(EnhancedPINN, self).__init__()
        
        self.dropout_rate = dropout_rate
        
        # Distance encoder
        self.distance_encoder = keras.Sequential([
            layers.Dense(hidden_dims[0], activation='tanh', dtype='float32'),
            layers.LayerNormalization(dtype='float32'),
            layers.Dropout(dropout_rate),
            layers.Dense(hidden_dims[0], activation='tanh', dtype='float32'),
            layers.LayerNormalization(dtype='float32'),
        ], name='distance_encoder')
        
        # Force predictor
        self.force_net = keras.Sequential([
            layers.Dense(hidden_dims[1], activation='tanh', dtype='float32'),
            layers.LayerNormalization(dtype='float32'),
            layers.Dropout(dropout_rate),
            layers.Dense(hidden_dims[2], activation='tanh', dtype='float32'),
            layers.LayerNormalization(dtype='float32'),
            layers.Dropout(dropout_rate),
            layers.Dense(hidden_dims[3], activation='tanh', dtype='float32'),
            layers.LayerNormalization(dtype='float32'),
            layers.Dense(1, dtype='float32'),
        ], name='force_net')
        
        # Energy predictor
        self.energy_net = keras.Sequential([
            layers.Dense(hidden_dims[1], activation='tanh', dtype='float32'),
            layers.LayerNormalization(dtype='float32'),
            layers.Dropout(dropout_rate),
            layers.Dense(hidden_dims[2], activation='tanh', dtype='float32'),
            layers.LayerNormalization(dtype='float32'),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation='tanh', dtype='float32'),
            layers.LayerNormalization(dtype='float32'),
            layers.Dense(1, dtype='float32'),
        ], name='energy_net')
    
    def compute_forces_single(self, positions, L, rc, training=False):
        """
        Compute forces for a single configuration
        """
        positions = tf.cast(positions, tf.float32)
        L = float(L)
        rc = float(rc)
        
        N = positions.shape[0]
        
        # Compute pairwise displacements
        pos_i = tf.expand_dims(positions, 1)  # (N, 1, 3)
        pos_j = tf.expand_dims(positions, 0)  # (1, N, 3)
        dr = pos_j - pos_i  # (N, N, 3)
        
        # Apply PBC
        dr = dr - L * tf.round(dr / L)
        
        # Compute distances
        r = tf.norm(dr, axis=2)  # (N, N)
        
        # Create mask: upper triangular and within cutoff
        indices = np.arange(N)
        i_idx, j_idx = np.meshgrid(indices, indices, indexing='ij')
        upper_tri = i_idx < j_idx
        
        r_np = r.numpy()
        cutoff_mask = (r_np < rc) & (r_np > 0.5)
        mask = upper_tri & cutoff_mask
        
        # Get valid pairs
        pairs = np.where(mask)
        i_indices = pairs[0]
        j_indices = pairs[1]
        
        if len(i_indices) == 0:
            return tf.zeros_like(positions), tf.constant(0.0, dtype=tf.float32)
        
        # Extract pair data
        dr_pairs = tf.gather_nd(dr, list(zip(i_indices, j_indices)))
        r_pairs = tf.gather_nd(r, list(zip(i_indices, j_indices)))
        
        # Network forward pass
        r_input = tf.expand_dims(r_pairs, 1)
        dist_features = self.distance_encoder(r_input, training=training)
        f_mag = tf.cast(self.force_net(dist_features, training=training), tf.float32)
        u_pair = tf.cast(self.energy_net(dist_features, training=training), tf.float32)
        
        # Compute forces
        f_vec = f_mag * (dr_pairs / tf.expand_dims(r_pairs, 1))
        
        # Accumulate forces
        forces = tf.zeros_like(positions, dtype=tf.float32)
        
        # Add forces to i particles
        indices_i = [[int(i), j] for i in i_indices for j in range(3)]
        updates_i = tf.reshape(f_vec, [-1])
        forces = tf.tensor_scatter_nd_add(forces, indices_i, updates_i)
        
        # Subtract forces from j particles
        indices_j = [[int(j), k] for j in j_indices for k in range(3)]
        updates_j = -tf.reshape(f_vec, [-1])
        forces = tf.tensor_scatter_nd_add(forces, indices_j, updates_j)
        
        # Total energy
        energy = tf.reduce_sum(u_pair)
        
        return forces, energy
    
    def compute_forces_batch(self, positions, L, rc, training=False):
        """
        Compute forces for a batch
        """
        # Handle single configuration
        single_config = len(positions.shape) == 2
        if single_config:
            return self.compute_forces_single(positions, L, rc, training)
        
        # Process batch one by one
        forces_list = []
        energies_list = []
        
        for i in range(positions.shape[0]):
            f, e = self.compute_forces_single(positions[i], L, rc, training)
            forces_list.append(f)
            energies_list.append(e)
        
        forces = tf.stack(forces_list)
        energies = tf.stack(energies_list)
        
        return forces, energies


class MDSystem:
    """Molecular Dynamics System with ML-Hybrid capability"""
    
    def __init__(self, config, pinn_model=None, model_config=None):
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
        self.tau = config.get('tau', 0.1)
        dof = 3 * self.N
        self.Q = dof * kB * self.T_target * self.tau**2
        self.xi = 0.0

        self.pos = self._init_fcc_lattice()
        self.vel = self._init_velocities()
        self._remove_com_motion()
        
        # ML components
        self.pinn = pinn_model
        self.model_config = model_config
        self.ml_blend_factor = 0.0
        self.ml_loaded = pinn_model is not None
        
        if self.ml_loaded:
            print("✓ PINN model loaded successfully")
            # Warm up the model
            dummy_input = tf.constant(self.pos, dtype=tf.float32)
            _ = self.pinn.compute_forces_batch(dummy_input, self.L, self.rc, training=False)
            print("✓ PINN model warmed up")
        else:
            print("⚠ Running in pure MD mode (no PINN)")
        
    def _init_fcc_lattice(self):
        """Initialize cubic lattice"""
        n = int(np.ceil(self.N**(1/3)))
        min_spacing = 1.3 * self.sig
        required_L = n * min_spacing
        
        if self.L < required_L:
            self.L = required_L
            self.rho = self.N / self.L**3
        
        a = self.L / n
        positions = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    pos = np.array([i, j, k]) * a + a/2
                    pos += (np.random.random(3) - 0.5) * 0.005 * a
                    positions.append(pos)
                    if len(positions) >= self.N:
                        result = np.array(positions[:self.N])
                        return result % self.L
        return np.array(positions[:self.N]) % self.L
    
    def _init_velocities(self):
        """Initialize velocities"""
        T_init = self.T_target * 0.1
        v = np.random.normal(0, np.sqrt(T_init / self.mass), (self.N, 3))
        return v
    
    def _remove_com_motion(self):
        """Remove center-of-mass motion"""
        self.vel -= np.mean(self.vel, axis=0)
    
    def _apply_pbc(self, r):
        """Apply periodic boundary conditions"""
        return r - self.L * np.round(r / self.L)
    
    def compute_forces_md(self):
        """Compute traditional Lennard-Jones forces"""
        forces = np.zeros_like(self.pos)
        U_pot = 0.0
        virial = 0.0
        F_MAX = 50.0
        r_min2 = (0.8 * self.sig)**2
        
        for i in range(self.N - 1):
            dr = self.pos[i+1:] - self.pos[i]
            dr = self._apply_pbc(dr)
            r2 = np.sum(dr**2, axis=1)
            
            mask = (r2 < self.rc2) & (r2 > r_min2)
            
            if not np.any(mask):
                continue
            
            r2_masked = r2[mask]
            dr_masked = dr[mask]
            
            inv_r2 = self.sig**2 / r2_masked
            inv_r6 = inv_r2**3
            inv_r12 = inv_r6**2
            
            f_mag = 24 * self.eps * inv_r2 * (2*inv_r12 - inv_r6)
            f_mag = np.clip(f_mag, -F_MAX, F_MAX)
            
            f_vec = f_mag[:, np.newaxis] * dr_masked
            
            forces[i] += np.sum(f_vec, axis=0)
            j_indices = np.arange(i+1, self.N)[mask]
            for idx, f in zip(j_indices, f_vec):
                forces[idx] -= f
            
            U_pot += np.sum(4*self.eps*(inv_r12 - inv_r6) - self.U_shift)
            virial += np.sum(f_mag * np.sqrt(r2_masked))
        
        return forces, U_pot, virial
    
    def compute_forces_ml(self):
        """Compute forces using PINN"""
        pos_tensor = tf.constant(self.pos, dtype=tf.float32)
        forces_tensor, energy_tensor = self.pinn.compute_forces_batch(
            pos_tensor, self.L, self.rc, training=False
        )
        
        forces = forces_tensor.numpy()
        energy = energy_tensor.numpy()
        
        # Estimate virial from forces
        virial = 0.0
        for i in range(self.N - 1):
            dr = self.pos[i+1:] - self.pos[i]
            dr = self._apply_pbc(dr)
            r = np.linalg.norm(dr, axis=1)
            mask = r < self.rc
            if np.any(mask):
                # Approximate virial contribution
                f_i = np.linalg.norm(forces[i])
                virial += f_i * np.sum(r[mask])
        
        return forces, energy, virial
    
    def compute_forces(self):
        """Hybrid force computation with smooth blending"""
        if not self.ml_loaded or self.ml_blend_factor == 0.0:
            # Pure MD
            return self.compute_forces_md()
        elif self.ml_blend_factor == 1.0:
            # Pure ML
            return self.compute_forces_ml()
        else:
            # Smooth blend
            f_md, u_md, v_md = self.compute_forces_md()
            f_ml, u_ml, v_ml = self.compute_forces_ml()
            
            alpha = self.ml_blend_factor
            forces = (1 - alpha) * f_md + alpha * f_ml
            energy = (1 - alpha) * u_md + alpha * u_ml
            virial = (1 - alpha) * v_md + alpha * v_ml
            
            return forces, energy, virial
    
    def step(self, dt, current_step, config):
        """Velocity Verlet with Nosé-Hoover thermostat"""
        
        # Update ML blend factor with smooth transition
        if self.ml_loaded:
            if current_step < config['ml_transition_start']:
                self.ml_blend_factor = 0.0
            elif current_step >= config['ml_transition_end']:
                self.ml_blend_factor = 1.0
            else:
                # Smooth sigmoid transition
                progress = (current_step - config['ml_transition_start']) / \
                          (config['ml_transition_end'] - config['ml_transition_start'])
                # Use smooth sigmoid instead of linear
                self.ml_blend_factor = 1 / (1 + np.exp(-10 * (progress - 0.5)))
        
        # Velocity Verlet integration
        forces, U, virial = self.compute_forces()
        
        K = 0.5 * self.mass * np.sum(self.vel**2)
        dof = 3 * self.N
        T_inst = 2 * K / (dof * kB)
        
        # Thermostat update
        self.xi += dt * (T_inst - self.T_target) / (self.tau**2 * self.T_target)
        damping = 0.95
        self.xi *= damping
        
        # First half-step
        acc = forces / self.mass - self.xi * self.vel
        self.vel += 0.5 * dt * acc
        
        # Position update
        self.pos += self.vel * dt
        self.pos %= self.L
        
        # Recompute forces
        forces, U, virial = self.compute_forces()
        K = 0.5 * self.mass * np.sum(self.vel**2)
        T_inst = 2 * K / (dof * kB)
        
        # Thermostat update
        self.xi += dt * (T_inst - self.T_target) / (self.tau**2 * self.T_target)
        self.xi *= damping
        
        # Second half-step
        acc = forces / self.mass - self.xi * self.vel
        self.vel += 0.5 * dt * acc
        
        # Velocity capping (safety)
        V_MAX = 5.0
        speed = np.linalg.norm(self.vel, axis=1)
        speed_max = np.max(speed)
        if speed_max > V_MAX:
            self.vel *= V_MAX / speed_max
        
        # Check for NaN
        if np.any(np.isnan(self.pos)) or np.any(np.isnan(self.vel)):
            raise ValueError("NaN detected in simulation!")
        
        T = self.temperature()
        P = self.pressure(T, virial)
        K = 0.5 * self.mass * np.sum(self.vel**2)
        
        return K + U, T, P, K, U
    
    def temperature(self):
        """Instantaneous temperature"""
        dof = 3 * self.N
        K = 0.5 * self.mass * np.sum(self.vel**2)
        return 2 * K / (dof * kB)
    
    def pressure(self, T, virial):
        """Instantaneous pressure"""
        V = self.L**3
        return self.N * kB * T / V + virial / (3 * V)


class Statistics:
    """Track simulation statistics"""
    
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
        
        return {
            'T_mean': np.mean(T_eq),
            'T_std': np.std(T_eq),
            'T_error': abs(np.mean(T_eq) - self.T_target) / self.T_target * 100,
            'E_mean': np.mean(E_eq),
            'E_std': np.std(E_eq),
            'E_drift': (E_eq[-1] - E_eq[0]) / abs(E_eq[0]) * 100 if abs(E_eq[0]) > 0 else 0,
            'P_mean': np.mean(P_eq),
            'P_std': np.std(P_eq),
        }


class Visualizer:
    """Real-time visualization with ML transition monitoring"""
    
    def __init__(self, system, config):
        self.system = system
        self.dt = config['dt']
        self.config = config
        self.stats = Statistics(config['T_target'])
        
        # Setup figure
        self.fig = plt.figure(figsize=(20, 12))
        gs = self.fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        # 3D particle view
        self.ax_3d = self.fig.add_subplot(gs[0:2, 0], projection='3d')
        
        # Energy plot
        self.ax_energy = self.fig.add_subplot(gs[0, 1])
        
        # Temperature & Pressure
        self.ax_temp = self.fig.add_subplot(gs[1, 1])
        self.ax_press = self.ax_temp.twinx()
        
        # ML Blend Factor
        self.ax_blend = self.fig.add_subplot(gs[0, 2])
        
        # Maxwell-Boltzmann
        self.ax_mb = self.fig.add_subplot(gs[1, 2])
        
        # Statistics
        self.ax_stats = self.fig.add_subplot(gs[2, 0])
        self.ax_stats.axis('off')
        
        # ML Info
        self.ax_ml_info = self.fig.add_subplot(gs[2, 1])
        self.ax_ml_info.axis('off')
        
        # Phase indicator
        self.ax_phase = self.fig.add_subplot(gs[2, 2])
        self.ax_phase.axis('off')
        
        self._setup_plots()
        self.time_data = []
        self.blend_data = []
        
    def _setup_plots(self):
        """Initialize plot elements"""
        # 3D scatter
        pos = self.system.pos
        speeds = np.linalg.norm(self.system.vel, axis=1)
        self.scat = self.ax_3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                                       c=speeds, cmap='plasma', s=30, alpha=0.8)
        self.ax_3d.set_xlim(0, self.system.L)
        self.ax_3d.set_ylim(0, self.system.L)
        self.ax_3d.set_zlim(0, self.system.L)
        self.ax_3d.set_title('Particle Positions (Color = Speed)')
        self.fig.colorbar(self.scat, ax=self.ax_3d, label='Speed', shrink=0.5)
        self.ax_3d.set_xticklabels([])
        self.ax_3d.set_yticklabels([])
        self.ax_3d.set_zticklabels([])
        
        # Energy
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
        
        # Blend Factor
        if self.system.ml_loaded:
            self.line_blend, = self.ax_blend.plot([], [], 'purple', lw=3)
            self.ax_blend.axhline(y=0.0, color='blue', ls='--', alpha=0.5, label='Pure MD')
            self.ax_blend.axhline(y=1.0, color='red', ls='--', alpha=0.5, label='Pure PINN')
            self.ax_blend.axvline(x=self.config['ml_transition_start'] * self.dt,
                                 color='green', ls=':', alpha=0.5)
            self.ax_blend.axvline(x=self.config['ml_transition_end'] * self.dt,
                                 color='green', ls=':', alpha=0.5)
            self.ax_blend.fill_between(
                [self.config['ml_transition_start'] * self.dt,
                 self.config['ml_transition_end'] * self.dt],
                0, 1, alpha=0.2, color='green', label='Transition'
            )
            self.ax_blend.set_xlabel('Time')
            self.ax_blend.set_ylabel('ML Blend Factor')
            self.ax_blend.set_title('MD → PINN Transition (Sigmoid)')
            self.ax_blend.set_ylim(-0.1, 1.1)
            self.ax_blend.legend(loc='center right')
            self.ax_blend.grid(True, alpha=0.3)
        else:
            self.ax_blend.text(0.5, 0.5, 'Pure MD Mode\n(No PINN)',
                              transform=self.ax_blend.transAxes,
                              ha='center', va='center', fontsize=14)
            self.ax_blend.axis('off')
        
        # MB distribution
        self.ax_mb.set_xlabel('Speed')
        self.ax_mb.set_ylabel('Probability Density')
        self.ax_mb.set_title('Maxwell-Boltzmann Distribution')
        self.ax_mb.grid(True, alpha=0.3)
        
    def update(self, frame):
        """Update animation frame"""
        try:
            E, T, P, K, U = self.system.step(self.dt, frame, self.config)
        except ValueError as e:
            print(f"Simulation failed: {e}")
            return
        
        self.stats.add(E, T, P, K, U)
        
        # Update 3D particles
        pos = self.system.pos
        speeds = np.linalg.norm(self.system.vel, axis=1)
        self.scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        self.scat.set_array(speeds)
        self.ax_3d.view_init(elev=20, azim=frame * 0.3)
        
        # Update time series
        time = frame * self.dt
        self.time_data.append(time)
        if self.system.ml_loaded:
            self.blend_data.append(self.system.ml_blend_factor)
        
        self.line_E.set_data(self.time_data, self.stats.energies)
        self.line_K.set_data(self.time_data, self.stats.kinetic)
        self.line_U.set_data(self.time_data, self.stats.potential)
        self.line_T.set_data(self.time_data, self.stats.temps)
        self.line_P.set_data(self.time_data, self.stats.pressures)
        
        if self.system.ml_loaded:
            self.line_blend.set_data(self.time_data, self.blend_data)
        
        # Auto-scale
        for ax in [self.ax_energy, self.ax_temp, self.ax_press]:
            ax.relim()
            ax.autoscale_view()
        
        if self.system.ml_loaded:
            self.ax_blend.relim()
            self.ax_blend.autoscale_view()
        
        # Update MB and stats every 20 frames
        if frame % 20 == 0:
            self._update_mb_distribution(T)
            self._update_statistics(frame)
            if self.system.ml_loaded:
                self._update_ml_info(frame)
                self._update_phase(frame)
        
        return (self.scat, self.line_E, self.line_K, self.line_U,
                self.line_T, self.line_P)
    
    def _update_mb_distribution(self, T):
        """Update Maxwell-Boltzmann distribution"""
        self.ax_mb.clear()
        
        speeds = np.linalg.norm(self.system.vel, axis=1)
        
        if np.any(np.isnan(speeds)) or np.any(np.isinf(speeds)):
            self.ax_mb.text(0.5, 0.5, 'Invalid data',
                           transform=self.ax_mb.transAxes,
                           ha='center', va='center', fontsize=12, color='red')
            return
        
        max_speed = np.max(speeds)
        if max_speed == 0 or not np.isfinite(max_speed):
            return
        
        counts, bins, _ = self.ax_mb.hist(speeds, bins=30, density=True,
                                          alpha=0.6, color='blue',
                                          label='MD Simulation')
        
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
            f_expected *= np.sum(counts) / np.sum(f_expected)
            
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
        
        if frame < 50:
            return
        
        stats = self.stats.get_stats()
        if stats is None:
            return
        
        text = f"""SIMULATION STATISTICS
Step: {frame}/{self.config['steps']}
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
"""
        self.ax_stats.text(0.1, 0.9, text, transform=self.ax_stats.transAxes,
                          fontfamily='monospace', fontsize=9, va='top',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    def _update_ml_info(self, frame):
        """Update ML information"""
        self.ax_ml_info.clear()
        self.ax_ml_info.axis('off')
        
        text = f"""ML-HYBRID STATUS
Current Blend: {self.system.ml_blend_factor:.4f}
  (0 = Pure MD, 1 = Pure PINN)
Method: Sigmoid Transition
  Start: Step {self.config['ml_transition_start']}
  End: Step {self.config['ml_transition_end']}
Model: Enhanced PINN
  Architecture: [256, 512, 512, 256]
  Status: {'Active' if self.system.ml_blend_factor > 0 else 'Inactive'}
"""
        
        color = 'lightgreen' if self.system.ml_blend_factor > 0 else 'lightblue'
        self.ax_ml_info.text(0.1, 0.9, text, transform=self.ax_ml_info.transAxes,
                            fontfamily='monospace', fontsize=9, va='top',
                            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    def _update_phase(self, frame):
        """Update phase indicator"""
        self.ax_phase.clear()
        self.ax_phase.axis('off')
        
        if frame < self.config['ml_transition_start']:
            phase = "PURE MD"
            color = 'blue'
            emoji = "🔬"
        elif frame < self.config['ml_transition_end']:
            phase = "TRANSITION"
            color = 'orange'
            emoji = "⚡"
        else:
            phase = "PURE PINN"
            color = 'red'
            emoji = "🤖"
        
        self.ax_phase.text(0.5, 0.5, f"{emoji}\n{phase}\n{emoji}",
                          transform=self.ax_phase.transAxes,
                          ha='center', va='center', fontsize=24, weight='bold',
                          bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))


def load_pinn_model(model_path='pinn_model.h5', config_path='model_config.pkl'):
    """Load pre-trained PINN model"""
    
    if not os.path.exists(model_path):
        print(f"⚠ Model file '{model_path}' not found")
        return None, None
    
    if not os.path.exists(config_path):
        print(f"⚠ Config file '{config_path}' not found")
        return None, None
    
    print("Loading PINN model...")
    
    # Load config
    with open(config_path, 'rb') as f:
        model_config = pickle.load(f)
    
    # Create model
    pinn = EnhancedPINN(
        hidden_dims=model_config['hidden_dims'],
        dropout_rate=model_config['dropout_rate']
    )
    
    # Build model with dummy input
    dummy_pos = np.random.rand(CONFIG['N'], 3).astype(np.float32) * model_config['L']
    dummy_input = tf.constant(dummy_pos, dtype=tf.float32)
    _ = pinn.compute_forces_batch(dummy_input, model_config['L'], model_config['rc'])
    
    # Load weights
    pinn.load_weights(model_path)
    
    print("✓ Model loaded successfully")
    print(f"  Architecture: {model_config['hidden_dims']}")
    print(f"  Box size: {model_config['L']:.3f}")
    print(f"  Cutoff: {model_config['rc']:.3f}")
    
    return pinn, model_config


if __name__ == '__main__':
    print("="*70)
    print("MOLECULAR DYNAMICS SIMULATION WITH PINN")
    print("="*70)
    
    # Try to load PINN model
    pinn, model_config = load_pinn_model()
    
    print(f"\nSimulation Parameters:")
    print(f"  Particles: {CONFIG['N']}")
    print(f"  Density: {CONFIG['rho']}")
    print(f"  Target Temperature: {CONFIG['T_target']}")
    print(f"  Time step: {CONFIG['dt']}")
    print(f"  Total steps: {CONFIG['steps']}")
    
    if pinn is not None:
        print(f"\nML Transition:")
        print(f"  Start: Step {CONFIG['ml_transition_start']} (t={CONFIG['ml_transition_start']*CONFIG['dt']:.2f})")
        print(f"  End: Step {CONFIG['ml_transition_end']} (t={CONFIG['ml_transition_end']*CONFIG['dt']:.2f})")
        print(f"  Pure PINN from: Step {CONFIG['ml_transition_end']}")
    
    # Create system
    print("\nInitializing MD system...")
    system = MDSystem(CONFIG, pinn_model=pinn, model_config=model_config)
    
    print(f"\nSystem Info:")
    print(f"  Box size: {system.L:.3f}")
    print(f"  Initial temperature: {system.temperature():.3f}")
    print(f"  Thermostat τ: {system.tau:.3f}")
    
    forces, U, virial = system.compute_forces_md()
    K = 0.5 * system.mass * np.sum(system.vel**2)
    print(f"  Initial energy: K={K:.3f}, U={U:.3f}, Total={K+U:.3f}")
    
    # Create visualizer
    print("\nSetting up visualization...")
    viz = Visualizer(system, CONFIG)
    
    print("\n" + "="*70)
    print("STARTING SIMULATION")
    print("="*70)
    if pinn is not None:
        print("Watch the ML Blend Factor plot to see the MD→PINN transition!")
    print()
    
    # Run animation
    ani = FuncAnimation(viz.fig, viz.update, frames=CONFIG['steps'],
                       interval=30, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()
    
    # Final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    final_stats = viz.stats.get_stats()
    if final_stats:
        for key, value in final_stats.items():
            print(f"{key:12s}: {value:10.4f}")
    
    if pinn is not None:
        print(f"\nFinal ML Blend Factor: {system.ml_blend_factor:.4f}")
        print("Simulation completed successfully with PINN integration!")
    else:
        print("\nSimulation completed in pure MD mode.")