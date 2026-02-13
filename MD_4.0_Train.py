"""
PINN Training Script
Download the trained model (pinn_model.h5)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# Check GPU
print("TensorFlow version:", tf.__version__)
print("GPU Available:", len(tf.config.list_physical_devices('GPU')) > 0)
if tf.config.list_physical_devices('GPU'):
    print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Simulation parameters
CONFIG = {
    'N': 108,
    'rho': 0.5,
    'T_target': 1.0,
    'dt': 0.005,
    'epsilon': 1.0,
    'sigma': 1.0,
    'rc': 2.5,
    'tau': 0.1,
}

kB = 1.0


class MDDataGenerator:
    """Generate training data from classical MD simulation"""
    
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
        self.tau = config.get('tau', 0.1)
        dof = 3 * self.N
        self.Q = dof * kB * self.T_target * self.tau**2
        self.xi = 0.0
        
        self.pos = self._init_fcc_lattice()
        self.vel = self._init_velocities()
        self._remove_com_motion()
        
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
    
    def compute_forces(self):
        """Compute Lennard-Jones forces"""
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
    
    def step(self, dt):
        """Velocity Verlet with Nosé-Hoover thermostat"""
        forces, U, virial = self.compute_forces()
        
        K = 0.5 * self.mass * np.sum(self.vel**2)
        dof = 3 * self.N
        T_inst = 2 * K / (dof * kB)
        
        self.xi += dt * (T_inst - self.T_target) / (self.tau**2 * self.T_target)
        damping = 0.95
        self.xi *= damping
        
        acc = forces / self.mass - self.xi * self.vel
        self.vel += 0.5 * dt * acc
        self.pos += self.vel * dt
        self.pos %= self.L
        
        forces, U, virial = self.compute_forces()
        K = 0.5 * self.mass * np.sum(self.vel**2)
        T_inst = 2 * K / (dof * kB)
        
        self.xi += dt * (T_inst - self.T_target) / (self.tau**2 * self.T_target)
        self.xi *= damping
        
        acc = forces / self.mass - self.xi * self.vel
        self.vel += 0.5 * dt * acc
        
        V_MAX = 5.0
        speed = np.linalg.norm(self.vel, axis=1)
        speed_max = np.max(speed)
        if speed_max > V_MAX:
            self.vel *= V_MAX / speed_max
        
        T = 2 * (0.5 * self.mass * np.sum(self.vel**2)) / (dof * kB)
        
        return forces, U, T
    
    def temperature(self):
        """Current temperature"""
        dof = 3 * self.N
        K = 0.5 * self.mass * np.sum(self.vel**2)
        return 2 * K / (dof * kB)


def generate_training_data(n_trajectories=5, steps_per_traj=2000, equilibration=400):
    """
    Generate diverse training data from multiple MD trajectories
    Args:
        n_trajectories: Number of independent trajectories
        steps_per_traj: Steps per trajectory
        equilibration: Equilibration steps before collecting data
    Returns:
        positions, forces, energies, temperatures, box_size
    """
    print(f"\nGenerating training data...")
    print(f"Trajectories: {n_trajectories}")
    print(f"Steps per trajectory: {steps_per_traj}")
    print(f"Equilibration: {equilibration}")
    
    all_positions = []
    all_forces = []
    all_energies = []
    all_temps = []
    
    for traj in range(n_trajectories):
        print(f"\nTrajectory {traj+1}/{n_trajectories}")
        
        # Vary initial conditions slightly
        config = CONFIG.copy()
        config['T_target'] = 1.0 + (np.random.rand() - 0.5) * 0.2  # T in [0.9, 1.1]
        
        system = MDDataGenerator(config)
        L = system.L
        
        # Equilibration
        print("  Equilibrating...")
        for step in tqdm(range(equilibration), desc="  Equilibration"):
            system.step(CONFIG['dt'])
        
        # Data collection
        print("  Collecting data...")
        for step in tqdm(range(steps_per_traj), desc="  Collection"):
            forces, energy, temp = system.step(CONFIG['dt'])
            
            # Store every 5th step to reduce correlation
            if step % 5 == 0:
                all_positions.append(system.pos.copy())
                all_forces.append(forces.copy())
                all_energies.append(energy)
                all_temps.append(temp)
    
    positions = np.array(all_positions, dtype=np.float32)
    forces = np.array(all_forces, dtype=np.float32)
    energies = np.array(all_energies, dtype=np.float32)
    temps = np.array(all_temps, dtype=np.float32)
    
    print(f"\nGenerated {len(positions)} snapshots")
    print(f"Temperature range: [{temps.min():.3f}, {temps.max():.3f}]")
    print(f"Energy range: [{energies.min():.3f}, {energies.max():.3f}]")
    
    return positions, forces, energies, temps, L


class EnhancedPINN(keras.Model):
    """
    Physics-Informed Neural Network for MD force prediction
    """
    
    def __init__(self, hidden_dims=[256, 512, 512, 256], dropout_rate=0.1):
        super(EnhancedPINN, self).__init__()
        
        self.dropout_rate = dropout_rate
        
        # Distance feature encoder
        self.distance_encoder = keras.Sequential([
            layers.Dense(hidden_dims[0], activation='tanh', dtype='float32'),
            layers.LayerNormalization(dtype='float32'),
            layers.Dropout(dropout_rate),
            layers.Dense(hidden_dims[0], activation='tanh', dtype='float32'),
            layers.LayerNormalization(dtype='float32'),
        ], name='distance_encoder')
        
        # Force magnitude predictor
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
        """Compute forces for a single configuration - NO @tf.function"""
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
        """Compute forces for a batch - processes each config separately"""
        # Handle single configuration
        single_config = len(positions.shape) == 2
        if single_config:
            return self.compute_forces_single(positions, L, rc, training)
        
        # Process batch one by one (not ideal but works)
        forces_list = []
        energies_list = []
        
        for i in range(positions.shape[0]):
            f, e = self.compute_forces_single(positions[i], L, rc, training)
            forces_list.append(f)
            energies_list.append(e)
        
        forces = tf.stack(forces_list)
        energies = tf.stack(energies_list)
        
        return forces, energies


def create_training_dataset(positions, forces, energies, batch_size=8, validation_split=0.15):
    """Create TensorFlow datasets for training and validation"""
    
    n_samples = len(positions)
    n_val = int(n_samples * validation_split)
    n_train = n_samples - n_val
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        positions[train_indices],
        forces[train_indices],
        energies[train_indices]
    ))
    
    val_dataset = tf.data.Dataset.from_tensor_slices((
        positions[val_indices],
        forces[val_indices],
        energies[val_indices]
    ))
    
    # Batch and prefetch
    train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset


class CustomTrainer:
    """Custom training loop with physics-informed loss"""
    
    def __init__(self, model, L, rc, learning_rate=1e-4):
        self.model = model
        self.L = L
        self.rc = rc
        
        # Use learning rate schedule
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=1000,
            decay_rate=0.95,
            staircase=True
        )
        
        self.optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Metrics
        self.train_loss_metric = keras.metrics.Mean(name='train_loss')
        self.val_loss_metric = keras.metrics.Mean(name='val_loss')
        self.force_mae_metric = keras.metrics.MeanAbsoluteError(name='force_mae')
        self.energy_mae_metric = keras.metrics.MeanAbsoluteError(name='energy_mae')
        
    def train_step(self, pos_batch, force_batch, energy_batch):
        """Single training step - NO @tf.function decorator"""
        with tf.GradientTape() as tape:
            # Forward pass
            forces_pred, energies_pred = self.model.compute_forces_batch(
                pos_batch, self.L, self.rc, training=True
            )
            
            # Force loss (MSE)
            loss_force = tf.reduce_mean(tf.square(forces_pred - force_batch))
            
            # Energy loss (MSE)
            loss_energy = tf.reduce_mean(tf.square(energies_pred - energy_batch))
            
            # Total loss with weights
            loss = loss_force + 0.1 * loss_energy
            
            # Add L2 regularization
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables 
                               if 'bias' not in v.name]) * 1e-5
            loss = loss + l2_loss
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss_metric.update_state(loss)
        self.force_mae_metric.update_state(force_batch, forces_pred)
        self.energy_mae_metric.update_state(energy_batch, energies_pred)
        
        return loss
    
    def val_step(self, pos_batch, force_batch, energy_batch):
        """Single validation step - NO @tf.function decorator"""
        # Forward pass
        forces_pred, energies_pred = self.model.compute_forces_batch(
            pos_batch, self.L, self.rc, training=False
        )
        
        # Losses
        loss_force = tf.reduce_mean(tf.square(forces_pred - force_batch))
        loss_energy = tf.reduce_mean(tf.square(energies_pred - energy_batch))
        loss = loss_force + 0.1 * loss_energy
        
        # Update metrics
        self.val_loss_metric.update_state(loss)
        
        return loss
    
    def train(self, train_dataset, val_dataset, epochs=100):
        """Train the model"""
        history = {
            'train_loss': [],
            'val_loss': [],
            'force_mae': [],
            'energy_mae': []
        }
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Reset metrics - FIXED: reset_state() not reset_states()
            self.train_loss_metric.reset_state()
            self.val_loss_metric.reset_state()
            self.force_mae_metric.reset_state()
            self.energy_mae_metric.reset_state()
            
            # Training
            for pos_batch, force_batch, energy_batch in tqdm(train_dataset, desc="Training"):
                self.train_step(pos_batch, force_batch, energy_batch)
            
            # Validation
            for pos_batch, force_batch, energy_batch in tqdm(val_dataset, desc="Validation"):
                self.val_step(pos_batch, force_batch, energy_batch)
            
            # Get metrics
            train_loss = self.train_loss_metric.result().numpy()
            val_loss = self.val_loss_metric.result().numpy()
            force_mae = self.force_mae_metric.result().numpy()
            energy_mae = self.energy_mae_metric.result().numpy()
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['force_mae'].append(force_mae)
            history['energy_mae'].append(energy_mae)
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"Force MAE: {force_mae:.6f}, Energy MAE: {energy_mae:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.model.save_weights('best_pinn_weights.h5')
                print("✓ Saved best model")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping after {epoch+1} epochs")
                    break
        
        # Load best weights
        self.model.load_weights('best_pinn_weights.h5')
        
        return history


def visualize_training(history):
    """Visualize training history"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Force MAE
    axes[0, 1].plot(history['force_mae'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Force MAE')
    axes[0, 1].set_title('Force Prediction Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Energy MAE
    axes[1, 0].plot(history['energy_mae'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Energy MAE')
    axes[1, 0].set_title('Energy Prediction Error')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss ratio
    axes[1, 1].plot(np.array(history['val_loss']) / np.array(history['train_loss']))
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Val Loss / Train Loss')
    axes[1, 1].set_title('Overfitting Monitor')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()


def validate_model(model, positions, forces, energies, L, rc, n_samples=100):
    """Validate model accuracy on test data"""
    print("\n" + "="*70)
    print("MODEL VALIDATION")
    print("="*70)
    
    indices = np.random.choice(len(positions), min(n_samples, len(positions)), replace=False)
    
    force_errors = []
    energy_errors = []
    
    for idx in tqdm(indices, desc="Validating"):
        pos = tf.constant(positions[idx], dtype=tf.float32)
        force_true = forces[idx]
        energy_true = energies[idx]
        
        force_pred, energy_pred = model.compute_forces_batch(pos, L, rc, training=False)
        
        force_pred = force_pred.numpy()
        energy_pred = energy_pred.numpy()
        
        force_error = np.mean(np.abs(force_pred - force_true))
        energy_error = np.abs(energy_pred - energy_true)
        
        force_errors.append(force_error)
        energy_errors.append(energy_error)
    
    force_errors = np.array(force_errors)
    energy_errors = np.array(energy_errors)
    
    print(f"\nForce MAE: {np.mean(force_errors):.6f} ± {np.std(force_errors):.6f}")
    print(f"Energy MAE: {np.mean(energy_errors):.6f} ± {np.std(energy_errors):.6f}")
    print(f"Force Max Error: {np.max(force_errors):.6f}")
    print(f"Energy Max Error: {np.max(energy_errors):.6f}")
    
    # Visualize error distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(force_errors, bins=30, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Force MAE')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Force Error Distribution')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(energy_errors, bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Energy MAE')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Energy Error Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('validation_errors.png', dpi=150)
    plt.show()


def main():
    """Main training pipeline"""
    print("="*70)
    print("PINN TRAINING FOR MOLECULAR DYNAMICS")
    print("="*70)
    
    # Generate training data
    positions, forces, energies, temps, L = generate_training_data(
        n_trajectories=10,
        steps_per_traj=2000,
        equilibration=400
    )
    
    # Save raw data
    print("\nSaving training data...")
    np.savez_compressed(
        'training_data.npz',
        positions=positions,
        forces=forces,
        energies=energies,
        temps=temps,
        L=L,
        config=CONFIG
    )
    
    # Create model
    print("\nCreating PINN model...")
    model = EnhancedPINN(
        hidden_dims=[256, 512, 512, 256],
        dropout_rate=0.1
    )
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset, val_dataset = create_training_dataset(
        positions, forces, energies,
        batch_size=8,  # Smaller batch size for CPU
        validation_split=0.15
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = CustomTrainer(model, L, CONFIG['rc'] * CONFIG['sigma'], learning_rate=1e-3)
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_dataset,
        val_dataset,
        epochs=150
    )
    
    # Visualize training
    visualize_training(history)
    
    # Validate model
    validate_model(model, positions, forces, energies, L, CONFIG['rc'] * CONFIG['sigma'])
    
    # Save final model
    print("\nSaving final model...")
    model.save_weights('pinn_model.h5')
    
    # Save config
    with open('model_config.pkl', 'wb') as f:
        pickle.dump({
            'L': L,
            'rc': CONFIG['rc'] * CONFIG['sigma'],
            'sigma': CONFIG['sigma'],
            'epsilon': CONFIG['epsilon'],
            'hidden_dims': [256, 512, 512, 256],
            'dropout_rate': 0.1,
        }, f)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nFiles saved:")
    print("  - pinn_model.h5 (trained weights)")
    print("  - best_pinn_weights.h5 (best weights)")
    print("  - model_config.pkl (configuration)")
    print("  - training_data.npz (training data)")
    print("  - training_history.png (training curves)")
    print("  - validation_errors.png (error distribution)")
    print("\nDownload these files to use with the simulation script!")

if __name__ == '__main__':
    main()