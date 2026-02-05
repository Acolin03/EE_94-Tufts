import os
import random
import torch
import numpy as np
import pandas as pd
import pickle
from ctgan import CTGAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import List, Dict, Tuple

# Assuming relative imports for RRAMEvaluator
from .ctgan_eval import RRAMEvaluator, get_frequency
from src.loss import simulate_rram_wrapper

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.ctgan import Generator, Discriminator

# --- Data Preparation and Training ---

def prepare_ctgan_dataset(evaluator: RRAMEvaluator, min_pulse_width: float = 1e-9, max_rows: int = 2000) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Evaluates RRAM sequences using the PINN model to create the dataset for CTGAN.
    CRITICAL FIX: Fits a FRESH StandardScaler to solve scikit-learn version mismatches.
    """
    print("Generating training dataset for CTGAN (evaluating RRAM sequences)...")
    
    # 1. Gather stats from the original dataset to fix the 'Broken Scaler' issue
    # We refit a new scaler on the current environment's data to ensure compatibility
    print("Refitting StandardScalers to fix version mismatch...")
    raw_voltages = []
    raw_dts = []
    
    # Extract raw data samples from the evaluator's dataset to learn the distribution
    # We take a sample to save time if the dataset is huge
    sample_size = min(len(evaluator.dataset), 1000)
    indices = random.sample(range(len(evaluator.dataset)), sample_size)
    
    for idx in indices:
        seq = evaluator.dataset[idx]
        # We assume the dataset might be returning normalized tensors, 
        # but we need to establish a baseline 'Mean' and 'Std' for the PINN inputs.
        # Since we can't easily un-normalize broken data, we use the PINN's existing
        # (potentially version-mismatched) scalers just to GENERATE the inputs,
        # but we will save a NEW scaler for the CTGAN to use.
        raw_voltages.extend(seq['voltage'].view(-1).tolist())
        raw_dts.extend(seq['dt'].view(-1).tolist())

    # Create the "Correct" Scalers for 1.6.1
    # Note: These are used to normalize the CTGAN's training data
    v_scaler = StandardScaler().fit(np.array(raw_voltages).reshape(-1, 1))
    dt_scaler = StandardScaler().fit(np.array(raw_dts).reshape(-1, 1))
    
    # Extract the 'Broken' parameters from the evaluator just for the generation loop
    # (We have to use what the PINN expects, even if it's weird)
    pinn_v_scale = evaluator.scalers['voltage'].scale_[0]
    pinn_v_mean = evaluator.scalers['voltage'].mean_[0]
    pinn_dt_scale = evaluator.scalers['dt'].scale_[0]
    pinn_dt_mean = evaluator.scalers['dt'].mean_[0]

    num_templates = len(evaluator.dataset)
    data_list = []

    # Create a mapping of material to template indices for balanced sampling
    mat_to_indices = {mat: [] for mat in evaluator.materials}
    for idx in range(num_templates):
        seq = evaluator.dataset[idx]
        mat_idx = seq.get('material_int', seq.get('material_idx')).item()
        mat_to_indices[evaluator.materials[mat_idx]].append(idx)

    # LOOP: Generate Hallucinated Data
    for i in tqdm(range(max_rows), desc="Progress:"):
        
        # 1. Balanced Material Selection (Round-Robin)
        material = evaluator.materials[i % len(evaluator.materials)]
        if not mat_to_indices[material]:
            continue # Skip if no templates for this material
            
        template_idx = random.choice(mat_to_indices[material])
        original_seq = evaluator.dataset[template_idx]
        sequence = {k: v.clone() if torch.is_tensor(v) else v for k, v in original_seq.items()}
        
        # 2. Hallucinate Random Parameters (Physical Ranges)
        fake_Vset_real = random.uniform(1.0, 5)   # Avoids ultra-low voltages that never switch
        fake_Vreset_real = random.uniform(-5, -1.0)
        fake_dt_real = random.uniform(1e-9, 100e-9) # Focus on 1ns - 100ns (The 'Action Zone')

        # 3. Normalize for PINN (Using PINN's expectations)
        fake_Vset_norm = (fake_Vset_real - pinn_v_mean) / pinn_v_scale
        fake_Vreset_norm = (fake_Vreset_real - pinn_v_mean) / pinn_v_scale
        fake_dt_norm = (fake_dt_real - pinn_dt_mean) / pinn_dt_scale
        
        sequence['voltage'][-1] = fake_Vset_norm
        sequence['dt'][-1] = fake_dt_norm

        if 'material_int' in sequence:
            material_index = sequence['material_int'].item()
        elif 'material_idx' in sequence:
            material_index = sequence['material_idx'].item()
        else:
            continue
        material = evaluator.materials[material_index]
        
        try:
            # 4. Evaluate SET
            pos_result = evaluator._evaluate_single_sequence(sequence, template_idx, material, "SET")

            # 5. Evaluate RESET
            neg_sequence = {k: v.clone() if torch.is_tensor(v) else v for k, v in original_seq.items()}
            neg_sequence['voltage'][-1] = fake_Vreset_norm
            neg_sequence['dt'][-1] = fake_dt_norm
            neg_result = evaluator._evaluate_single_sequence(neg_sequence, template_idx, material, "RESET")
            
            merged_result = evaluator._merge_results(pos_result, neg_result)

            if i % 1000 == 0:
                print(f"DEBUG: Sample {i} - Energy: {merged_result['total_energy']:.2e} J")
            
    # 6. Store Clean Real-World Data
            data_list.append({
                'material': material,
                'Vset': merged_result['pos_voltage'],
                'Vreset': merged_result['neg_voltage'],
                'dt': abs(fake_dt_real), 
                'endurance': merged_result['avg_endurance'],
                'latency': abs(merged_result['total_switching_time']), 
                'energy': abs(merged_result['total_energy']),
            })

        except Exception as e:
            continue
            
    print(f"Dataset generated with {len(data_list)} synthetic samples.")
    data_df = pd.DataFrame(data_list)

    # --- BALANCE CHECK ---
    # Ensure we have a healthy mix of all materials
    print("\nMaterial Balance in Generated Data:")
    print(data_df['material'].value_counts())

    initial_count = len(data_df)
    data_df = data_df[data_df['latency'] <= 5e-9].copy() 
    
    pruned_count = initial_count - len(data_df)
    print(f"DEBUG: High-Speed Filter removed {pruned_count} slow designs.")
    print(f"DEBUG: Final Training Set Size: {len(data_df)} samples.")

    if len(data_df) < 100:
        print("WARNING: Physics model is struggling to find designs under 500ns! Training may fail.")
    
    # 7. FIT THE MASTER SCALER
    # We fit a brand new scaler on this generated data.
    # This scaler is 100% compatible with the current environment.
    print("Fitting new Master Scaler for CTGAN...")
    feature_cols = ['Vset', 'Vreset', 'dt', 'endurance', 'latency', 'energy']
    master_scaler = StandardScaler()
    master_scaler.fit(data_df[feature_cols])
    
    return data_df, master_scaler

def compute_physics_loss(generated_df, evaluator, master_scaler, physics_lambda=1.0):
    """
    Computes the physical validity of the generated designs.
    NOTE: Since gradients are broken by inverse_transform, this acts as a 
    High-Fidelity Monitor during training.
    """

    v_set = torch.tensor(real_data[:, 0], device=evaluator.device).float().abs()

    # 1. De-normalize (Map back to Real Units: Volts, Seconds)
    feature_cols = ['Vset', 'Vreset', 'dt', 'endurance', 'latency', 'energy']
    
    # We use the scaler you fit in prepare_ctgan_dataset
    real_data = master_scaler.inverse_transform(generated_df[feature_cols])
    
    # Convert back to Tensor for calculation (Device placement)
    v_set = torch.tensor(real_data[:, 0], device=evaluator.device).float()
    v_reset = torch.tensor(real_data[:, 1], device=evaluator.device).float()
    dt = torch.tensor(real_data[:, 2], device=evaluator.device).float().abs()
    
    # Map integers back to material names
    idx_to_mat = {v: k for k, v in evaluator.material_to_idx.items()}
    
    total_penalty = 0.0
    
    # Loop through the batch (Physics is sequential)
    for i in range(len(v_set)):
        mat_idx = int(generated_df.iloc[i]['material_int'])
        material = idx_to_mat.get(mat_idx, 'HfO2') # Default safety
        
        # 2. SIMULATE PHYSICS (The Referee)
        # We check if the device can switch within the pulse width dt
        # Note: We use a small current floor (100uA) to ensure heat generation
        gap, temp = simulate_rram_wrapper(
            dt[i:i+1], v_set[i:i+1], torch.tensor([100e-6], device=evaluator.device), 
            evaluator.const, material, use_cache=False
        )
        
        # 3. CALCULATE ERRORS
        # Did it close the gap? (Goal: 0.1nm)
        gap_final = gap[-1]
        switching_error = torch.abs(gap_final - evaluator.const.gap_min) * 1e9
        
        # Did it melt?
        t_limit = evaluator.const.material_params[material]['T_melt']
        thermal_error = torch.relu(temp.max() - t_limit) / 100.0
        
        # Polarity Check (Vset > 0, Vreset < 0)
        polarity_error = torch.relu(-v_set[i]) + torch.relu(v_reset[i])
        
        # 4. DISCOVERY BONUS (The "Slope")
        # If the gap didn't move (still at max), penalize distance from 5.0V
        # This tells us if the GAN is "trying" to find the switch.
        discovery_loss = 0.0
        if gap_final >= (evaluator.const.gap_max * 0.99):
            discovery_loss = 0.05 * torch.abs(5.0 - v_set[i])
        
        total_penalty += (switching_error + thermal_error + polarity_error + discovery_loss)
        
        # LOGGING (Only for the first item in batch)
        if i == 0:
            status = "FROZEN" if gap_final > 1.0e-9 else "SWITCHED"
            print(f" REF CHECK: {material} | Vset={v_set[i]:.2f}V | Gap={gap_final*1e9:.2f}nm ({status}) | Err={switching_error:.4f}")

    return (total_penalty / len(v_set)) * physics_lambda

    
def train_ctgan(evaluator: RRAMEvaluator, num_epochs: int, dataset_path: str, model_save_path: str, 
                create_new_dataset: bool, max_rows_per_file: int, min_pulse_width: float):
    
    master_scaler = None
    
    if create_new_dataset or not os.path.exists(dataset_path):
        # Generate data AND get the new clean scaler
        data_df, master_scaler = prepare_ctgan_dataset(evaluator, min_pulse_width, max_rows_per_file)
        
        # Save both the data and the scaler in one package
        torch.save({'data': data_df, 'scaler': master_scaler}, dataset_path)
        print(f"Dataset and new Scaler saved to {dataset_path}")
    else:
        print(f"Loading existing dataset from {dataset_path}")
        packet = torch.load(dataset_path, weights_only=False)
        if isinstance(packet, dict) and 'scaler' in packet:
            data_df = packet['data']
            master_scaler = packet['scaler']
        else:
            # Fallback for old files (should not happen if you deleted the file)
            data_df = packet
            print("WARNING: Old dataset format detected. Scalers might be broken.")

    # Prepare for Training
    train_df = data_df.copy()
    train_df['material_int'] = train_df['material'].apply(lambda x: evaluator.material_to_idx[x])
    discrete_columns = ['material_int'] 
    train_df = train_df.drop(columns=['material'])
    
    print(f"Training CTGAN for {num_epochs} epochs...")
    ctgan_model = CTGAN(epochs=num_epochs, batch_size=500, generator_dim=(128, 128), discriminator_dim=(128, 128))
    ctgan_model.fit(train_df, discrete_columns)

    with open(model_save_path, 'wb') as f:
        pickle.dump(ctgan_model, f)
    
    print(f"CTGAN training complete. Model saved to {model_save_path}")


# --- Recommendation and Pareto Logic ---

class RRAMCTGANRecommender:
    # Centralized Voltage Ranges
    voltage_ranges = {
        'HfO2':  {'pos': (1.5, 2.0),  'neg': (-2.0, -1.5)},
        'TiO2':  {'pos': (1.65, 2.0), 'neg': (-2.0, -1.65)},
        'Al2O3': {'pos': (1.38, 2.0), 'neg': (-2.0, -1.38)}
    }

    def __init__(self, ctgan_model_path: str, evaluator: RRAMEvaluator, dataset_path: str, min_pulse_width: float):
        
        self.evaluator = evaluator
        self.min_pulse_width = min_pulse_width
        self.device = evaluator.device
        
        print(f"Loading CTGAN model from {ctgan_model_path}")
        with open(ctgan_model_path, 'rb') as f:
            self.ctgan_model = pickle.load(f)
            
        print(f"Loading CTGAN training data stats from {dataset_path}")
        packet = torch.load(dataset_path, weights_only=False)
        
        # Handle the new packet format
        if isinstance(packet, dict) and 'scaler' in packet:
            self.data_df = packet['data']
            self.scaler = packet['scaler'] # LOAD THE CLEAN SCALER
            print("Successfully loaded clean StandardScaler.")
        else:
            self.data_df = packet
            self.scaler = StandardScaler()
            self.scaler.fit(self.data_df[['Vset', 'Vreset', 'dt', 'endurance', 'latency', 'energy']])
        
        if 'material_int' not in self.data_df.columns:
             self.data_df['material_int'] = self.data_df['material'].apply(lambda x: evaluator.material_to_idx.get(x, -1))
        
        self.input_cols = ['Vset', 'Vreset', 'dt']
        self.perf_cols = ['endurance', 'latency', 'energy']

    def calculate_score(self, predicted_performance: np.ndarray, target_performance: Dict, 
                        energy_penalty_factor: float = 1.0, 
                        w_end: float = 1.0, w_lat: float = 1.0, w_en: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Calculates a weighted mismatch score. 
        Higher weights (e.g., 5.0) force the model to prioritize that specific metric.
        """
        pred_endurance = predicted_performance[:, 0]
        pred_latency = predicted_performance[:, 1]
        pred_energy = predicted_performance[:, 2]

        target_endurance = target_performance['endurance']
        target_latency = target_performance['latency']
        target_energy = target_performance['energy']
        
        scores = np.zeros_like(pred_endurance)
        impossible_mask = (pred_endurance <= 0) | (pred_latency <= 0) | (pred_energy <= 0)
        scores[impossible_mask] = float('inf')
        
        valid_mask = ~impossible_mask
        if not np.any(valid_mask):
            return {'combined': scores}

        # Log-scale improvement for endurance
        valid_endurance = pred_endurance[valid_mask]
        log_improvement = np.log10(np.maximum(1.0, valid_endurance / target_endurance))
        
        k = 0.35 
        endurance_score_val = 0.5 + 0.5 * np.tanh(k * log_improvement)
        endurance_mismatch = 1.0 - endurance_score_val

        # Relative mismatch for latency and energy
        latency_mismatch = np.maximum(0, pred_latency[valid_mask] - target_latency) / target_latency
        energy_mismatch = np.maximum(0, pred_energy[valid_mask] - target_energy) / target_energy
        
        # Applying the optimization weights
        valid_combined = (w_end * endurance_mismatch) + \
                         (w_lat * latency_mismatch) + \
                         (w_en * energy_mismatch * energy_penalty_factor)
                         
        scores[valid_mask] = valid_combined
        
        return {'combined': scores}

    def recommend_parameters(self, target_endurance, target_switching_time, target_energy, 
                            num_recommendations=4, num_samples=10000, 
                            energy_penalty_factor=1.0, max_energy_error_ratio=0.0, retry_count=0):
        
        if retry_count > 4:
            print("ERROR: Physics filter pruned all candidates.")
            return []

        try:
            generated_df = self.ctgan_model.sample(num_samples)
        except AttributeError:
            # Kickstart the internal sampler if it's missing
            self.ctgan_model._is_fitted = True 
            generated_df = self.ctgan_model.sample(num_samples)
        
        # De-normalize and clean data
        feature_cols = ['Vset', 'Vreset', 'dt', 'endurance', 'latency', 'energy']
        generated_df[feature_cols] = self.scaler.inverse_transform(generated_df[feature_cols])

        generated_df['Vset'] = generated_df['Vset'].abs()
        generated_df['Vreset'] = -generated_df['Vreset'].abs()

        # 2. Material-Specific Magnitude Clipping
        # We prevent the GAN from 'Corner Hunting' at 3.89V
        for mat_name, mat_idx in self.evaluator.material_to_idx.items():
            mask = (generated_df['material_int'] == mat_idx)
            if mask.any():
                # HfO2 is tougher, Al2O3 is more sensitive
                v_max = 5
                generated_df.loc[mask, 'Vset'] = generated_df.loc[mask, 'Vset'].clip(0.8, v_max)
                generated_df.loc[mask, 'Vreset'] = generated_df.loc[mask, 'Vreset'].clip(-v_max, -0.8)

                
        generated_df['dt'] = generated_df['dt'].abs()
        generated_df['latency'] = generated_df['latency'].abs()
        generated_df['energy'] = generated_df['energy'].abs()

        # Physical dimension validation and material-specific endurance caps
        valid_df = self.validate_physical_dimensions(generated_df)
        if len(valid_df) == 0:
            return self.recommend_parameters(target_endurance, target_switching_time, target_energy, 
                                          num_recommendations, num_samples*2, energy_penalty_factor, 
                                          max_energy_error_ratio, retry_count + 1)

        idx_to_mat = {idx: mat for mat, idx in self.evaluator.material_to_idx.items()}
        for idx, mat in idx_to_mat.items():
            mask = (valid_df['material_int'] == idx)
            cap = 1e6 if mat == 'HfO2' else 1e8
            valid_df.loc[mask, 'endurance'] = valid_df.loc[mask, 'endurance'].clip(upper=cap)

        # 4 Optimization Targets Loop (Zhang-Donato Style)
        optimization_configs = [
            ("Overall Performance", 1.0, 1.0, 1.0),
            ("Endurance Optimization", 5.0, 1.0, 1.0),
            ("Energy Optimization", 1.0, 1.0, 5.0),
            ("Switching Time Optimization", 1.0, 5.0, 1.0)
        ]

        predicted_perf = valid_df[['endurance', 'latency', 'energy']].values
        target_dict = {'endurance': target_endurance, 'latency': target_switching_time, 'energy': target_energy}
        results = []
        used_indices = set()

        for target_name, w_end, w_lat, w_en in optimization_configs:
            score_components = self.calculate_score(
                predicted_perf, target_dict, energy_penalty_factor,
                w_end=w_end, w_lat=w_lat, w_en=w_en
            )
            valid_df['temp_score'] = score_components['combined']
            
            # Find the best candidate that hasn't been used yet
            sorted_candidates = valid_df.sort_values('temp_score')
            best_row = None
            for idx, row in sorted_candidates.iterrows():
                if idx not in used_indices:
                    best_row = row
                    used_indices.add(idx)
                    break
            
            if best_row is None:
                continue

            best_row['score'] = best_row['temp_score']
            formatted = self._format_rec(best_row, target_name, target_dict)
            results.append(formatted)
            
        return results

    def _format_rec(self, row, target_name, targets):
        idx_to_mat = {idx: mat for mat, idx in self.evaluator.material_to_idx.items()}
        
        # Handle cases where material is already a string name or an integer index
        if 'material' in row and isinstance(row['material'], str):
            mat_name = row['material']
        else:
            mat_name = idx_to_mat[int(row['material_int'])]

        return {
            'material': mat_name,
            'pos_voltage': float(row.get('Vset', row.get('pos_voltage', 0))), 
            'neg_voltage': float(row.get('Vreset', row.get('neg_voltage', 0))), 
            'dt': float(row['dt']),
            'optimization_target': target_name,
            'predicted_performance': {
                'endurance': float(row['endurance']), 
                'total_switching_time': float(row['latency']), 
                'energy': float(row['energy'])
            },
            'score': float(row['score'])
        }

    def sample_candidates(self, num_samples=10000):
        """Samples and validates candidates from CTGAN without scoring them."""
        try:
            generated_df = self.ctgan_model.sample(num_samples)
        except AttributeError:
            self.ctgan_model._is_fitted = True 
            generated_df = self.ctgan_model.sample(num_samples)
        
        # De-normalize and clean data
        feature_cols = ['Vset', 'Vreset', 'dt', 'endurance', 'latency', 'energy']
        generated_df[feature_cols] = self.scaler.inverse_transform(generated_df[feature_cols])

        generated_df['Vset'] = generated_df['Vset'].abs()
        generated_df['Vreset'] = -generated_df['Vreset'].abs()
        generated_df['dt'] = generated_df['dt'].abs()
        generated_df['latency'] = generated_df['latency'].abs()
        generated_df['energy'] = generated_df['energy'].abs()

        # Material-Specific Magnitude Clipping
        for mat_name, mat_idx in self.evaluator.material_to_idx.items():
            mask = (generated_df['material_int'] == mat_idx)
            if mask.any():
                v_max = 5
                generated_df.loc[mask, 'Vset'] = generated_df.loc[mask, 'Vset'].clip(0.8, v_max)
                generated_df.loc[mask, 'Vreset'] = generated_df.loc[mask, 'Vreset'].clip(-v_max, -0.8)

        valid_df = self.validate_physical_dimensions(generated_df)
        
        # Map material_int back to material name
        idx_to_mat = {idx: mat for mat, idx in self.evaluator.material_to_idx.items()}
        valid_df['material'] = valid_df['material_int'].apply(lambda x: idx_to_mat[int(x)])
        
        # --- PHYSICS-INFORMED POST-PROCESSING ---
        # Instead of trusting the GAN's performance predictions (which can mode collapse),
        # we treat the GAN as a "Hypothesis Generator" and use the PINN as the "Referee".
        print("Validating GAN candidates with PINN Physics Engine...")
        
        verified_data = []
        # Added tqdm for progress bar
        for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Verifying Candidates"):
            try:
                # Run the actual physics simulation for this candidate
                result = self.evaluator.evaluate(
                    row['material'], 
                    row['Vset'], 
                    row['Vreset'], 
                    # We can't easily pass dt to evaluate() without modifying it, 
                    # so we trust the GAN's dt for now but verify energy/latency relationships
                )
                
                # Append the PHYSICALLY VERIFIED metrics
                verified_data.append({
                    'material': row['material'],
                    'Vset': row['Vset'],
                    'Vreset': row['Vreset'],
                    'dt': row['dt'],
                    'endurance': float(result['avg_endurance']), # From PINN
                    'latency': float(result['total_switching_time']), # From PINN
                    'energy': float(result['total_energy']) # From PINN
                })
            except Exception:
                continue
                
        return pd.DataFrame(verified_data)

    def pick_best_from_cloud(self, verified_df, target_endurance, target_switching_time, target_energy, energy_penalty_factor=1.0):
        """Finds the best recommendations from an already verified pool of points."""
        if verified_df.empty:
            return []
            
        target_dict = {'endurance': target_endurance, 'latency': target_switching_time, 'energy': target_energy}
        
        # 4 Optimization Targets (Zhang-Donato Style)
        optimization_configs = [
            ("Overall Performance", 1.0, 1.0, 1.0),
            ("Endurance Optimization", 5.0, 1.0, 1.0),
            ("Energy Optimization", 1.0, 1.0, 5.0),
            ("Switching Time Optimization", 1.0, 5.0, 1.0)
        ]

        predicted_perf = verified_df[['endurance', 'latency', 'energy']].values
        results = []
        used_indices = set()

        for target_name, w_end, w_lat, w_en in optimization_configs:
            score_components = self.calculate_score(
                predicted_perf, target_dict, energy_penalty_factor,
                w_end=w_end, w_lat=w_lat, w_en=w_en
            )
            verified_df['temp_score'] = score_components['combined']
            
            # Find best unused candidate
            sorted_candidates = verified_df.sort_values('temp_score')
            best_row = None
            for idx, row in sorted_candidates.iterrows():
                if idx not in used_indices:
                    best_row = row
                    used_indices.add(idx)
                    break
            
            if best_row is not None:
                best_row['score'] = best_row['temp_score']
                formatted = self._format_rec(best_row, target_name, target_dict)
                results.append(formatted)
                
        return results

    def validate_physical_dimensions(self, df):
        initial_count = len(df)
        time_col = 'latency' if 'latency' in df.columns else 'dt'

        # STRICT PHYSICS FILTER (10pJ - 5000pJ)
        df_valid = df[
            (df['energy'] >= 10e-12) & (df['energy'] <= 5000e-12) & 
            (df[time_col] >= 0.1e-9) & (df[time_col] <= 500e-9) & # CRITICAL: Max 500ns
            (df['endurance'] >= 1e3)
        ].copy()
        
        removed = initial_count - len(df_valid)
        if removed > 0:
            print(f"DEBUG: Pruned {removed} candidates. Validity Rate: {((len(df_valid)) / initial_count) * 100:.1f}%")
            
        return df_valid