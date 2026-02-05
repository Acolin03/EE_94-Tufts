import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, Dict
# Assuming 'src.data' and 'src.models' are correctly structured relative to where you run this.
from src.data import RRAMDataset, Constants
from src.models import RRAM_PINN, MLP_Current
from src.utils import load_checkpoint, calculate_accuracy

class RRAMEvaluator:
    def __init__(self, model_path, data_path, output_dir, device=None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.const = Constants()
        
        self.checkpoint, self.pinn_model, self.mlp_model = self.load_model(model_path)
        self.dataset = self.load_data(data_path, self.checkpoint['scalers'])
        self.scalers = self.checkpoint['scalers']
        
        self.materials = ['HfO2', 'Al2O3', 'TiO2']
        self.material_to_idx = {mat: idx for idx, mat in enumerate(self.materials)}
        
        self.endurance_calculator = EnduranceCalculator()

    def load_model(self, model_path):
        print(f"Loading model from {model_path}")
        # THIS IS THE ONLY LINE NEEDED TO LOAD THE CHECKPOINT:
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False) 
        
        # Note: hidden_size and embedding_size should match training parameters
        hidden_size = 10
        embedding_size = 3
        
        pinn_model = RRAM_PINN(
            hidden_size=hidden_size, 
            embedding_size=embedding_size,
            const=self.const
        ).to(self.device)
        
        mlp_model = MLP_Current(
            hidden_size=hidden_size,
            embedding_size=embedding_size
        ).to(self.device)
        
        pinn_model.load_state_dict(checkpoint['pinn_model_state_dict'])
        mlp_model.load_state_dict(checkpoint['mlp_model_state_dict'])
        
        pinn_model.eval()
        mlp_model.eval()
        
        print(f"Successfully loaded model - Epoch: {checkpoint['epoch']+1}")
        print(f"Validation accuracy: {checkpoint['best_valid_accuracy']:.2f}%")
        print(f"Validation mean error: {checkpoint['best_valid_mean_error']:.2f}%")
        
        return checkpoint, pinn_model, mlp_model
    
    def load_data(self, data_path, scalers):
        print(f"Loading dataset from {data_path}")
        dataset = RRAMDataset(
            data_path=data_path,
            fit_scaler=False,
            is_train=True,
            use_full_dataset=True,
            split_ratio=1.0,
            seed=122
        )
        dataset.set_scalers(scalers)

        print(f"Loaded {len(dataset)} sequences")
        material_counts = dataset.get_material_distribution()
        for material, count in material_counts.items():
            percentage = (count / len(dataset)) * 100
            print(f"{material}: {count} sequences ({percentage:.1f}%)")
            
        return dataset
    
    def predict_sequence(self, sequence):
        time_seq_full = sequence['time'].to(self.device)
        dt_seq_full = sequence['dt'].to(self.device)
        voltage_seq_full = sequence['voltage'].to(self.device)
        true_current_full = sequence['current'].to(self.device)
        material_idx = sequence['material_idx'].to(self.device)
        
        time_seq = time_seq_full[2:]
        dt_seq = dt_seq_full[2:]
        voltage_seq = voltage_seq_full[2:]
        true_current = true_current_full[2:]
        initial_I = torch.ones_like(voltage_seq) * true_current[0]
        
        time_scale = torch.tensor(self.scalers['time'].scale_[0], device=self.device)
        time_mean = torch.tensor(self.scalers['time'].mean_[0], device=self.device)
        time_real_full = time_seq_full * time_scale + time_mean
        
        voltage_scale = torch.tensor(self.scalers['voltage'].scale_[0], device=self.device)
        voltage_mean = torch.tensor(self.scalers['voltage'].mean_[0], device=self.device)
        voltage_real_full = voltage_seq_full * voltage_scale + voltage_mean
        
        dt_scale = torch.tensor(self.scalers['dt'].scale_[0], device=self.device)
        dt_mean = torch.tensor(self.scalers['dt'].mean_[0], device=self.device)
        dt_real_full = dt_seq_full * dt_scale + dt_mean
        
        current_scale = torch.tensor(self.scalers['current'].scale_[0], device=self.device)
        current_mean = torch.tensor(self.scalers['current'].mean_[0], device=self.device)
        true_current_real_full = true_current_full * current_scale + current_mean
        
        time_real = time_real_full[2:]
        voltage_real = voltage_real_full[2:]
        dt_real = dt_real_full[2:]
        true_current_real = true_current_real_full[2:]
        
        with torch.no_grad():
            gap = self.pinn_model(time_seq, dt_real, voltage_seq, material_idx)
            pred_current = self.mlp_model(gap, voltage_seq, initial_I, material_idx)
            pred_current_real = pred_current * current_scale + current_mean

        return time_real, dt_real, voltage_real, true_current_real, pred_current_real, gap
    
    def evaluate(self, material, pos_voltage, neg_voltage):
        # print(f"Evaluating {material} material at positive {pos_voltage}V and negative {neg_voltage}V performance...")
        
        if material not in self.materials:
            raise ValueError(f"Unknown material: {material}. Supported materials: {self.materials}")
            
        material_idx = self.material_to_idx[material]
        voltage_scale = self.scalers['voltage'].scale_[0]
        voltage_mean = self.scalers['voltage'].mean_[0]
        
        available_voltages = []
        for sequence in self.dataset:
            if sequence['material_idx'].item() != material_idx:
                continue
            
            seq_voltage = sequence['voltage'][-1].item()
            seq_voltage_real = seq_voltage * voltage_scale + voltage_mean
            available_voltages.append(seq_voltage_real)
        
        available_voltages = sorted(list(set([round(v, 2) for v in available_voltages])))
        pos_voltages = [v for v in available_voltages if v > 0]
        neg_voltages = [v for v in available_voltages if v < 0]
        
        # print(f"Available positive voltages: {pos_voltages}")
        # print(f"Available negative voltages: {neg_voltages}")
        
        pos_matched_sequence, pos_matched_idx, pos_actual_voltage = self._find_closest_sequence(material_idx, pos_voltage, "positive")
        neg_matched_sequence, neg_matched_idx, neg_actual_voltage = self._find_closest_sequence(material_idx, neg_voltage, "negative")
        
        pos_result = self._evaluate_single_sequence(pos_matched_sequence, pos_matched_idx, material, "SET")
        neg_result = self._evaluate_single_sequence(neg_matched_sequence, neg_matched_idx, material, "RESET")
        
        merged_result = self._merge_results(pos_result, neg_result)
        
        return merged_result
    
    def _find_closest_sequence(self, material_idx, target_voltage, voltage_type):
        voltage_scale = self.scalers['voltage'].scale_[0]
        voltage_mean = self.scalers['voltage'].mean_[0]
        
        closest_sequence = None
        closest_idx = -1
        closest_voltage = None
        min_diff = float('inf')
        
        for idx, sequence in enumerate(self.dataset):
            if sequence['material_idx'].item() != material_idx:
                continue
            seq_voltage = sequence['voltage'][-1].item()
            seq_voltage_real = seq_voltage * voltage_scale + voltage_mean
            
            if (voltage_type == "positive" and seq_voltage_real <= 0) or (voltage_type == "negative" and seq_voltage_real >= 0):
                continue
            diff = abs(seq_voltage_real - target_voltage)
            if diff < min_diff:
                min_diff = diff
                closest_sequence = sequence
                closest_idx = idx
                closest_voltage = seq_voltage_real
        
        if closest_sequence is None:
            raise ValueError(f"No {voltage_type} voltage sequence found for the material. Please check if the voltage value is reasonable.")
        
        return closest_sequence, closest_idx, closest_voltage
    
    def _evaluate_single_sequence(self, sequence, sequence_idx, material, operation_type):
        material_idx = self.material_to_idx[material]
        
        time_real, dt_real, voltage_real, true_current_real, pred_current_real, gap = self.predict_sequence(sequence)
        stable_idx = get_stable_index(pred_current_real)
        
        if stable_idx < len(time_real):
            actual_switching_time = time_real[stable_idx] - time_real[0]
            actual_switching_time = actual_switching_time.item()
        else:
            actual_switching_time = (time_real[-1] - time_real[0]).item()
        
        # Calculate endurance and metrics
        endurance, metrics = self.endurance_calculator.calculate_endurance(
            pred_current_real.mean().item(), 
            voltage_real.mean().item(), 
            dt_real,
            material
        )
        threshold_times, total_energy = get_energy_consumption(
            time_real, voltage_real, pred_current_real, material_idx
        )
        
        result = {
            'sequence_id': sequence_idx,
            'material': material,
            'voltage': voltage_real.mean().item(),
            'operation_type': operation_type,
            'endurance': endurance,
            'actual_switching_time': actual_switching_time,
            'total_energy': total_energy.item(), # Convert tensor to float
            'threshold_times': threshold_times,
            'temperature': metrics['temperature'],
            'stable_index': stable_idx,
            'metrics': metrics,
            'time_real': time_real,
            'voltage_real': voltage_real,
            'true_current_real': true_current_real,
            'pred_current_real': pred_current_real,
            'gap': gap
        }
        
        return result
    
    def _merge_results(self, pos_result, neg_result):
        """Merge positive and negative voltage evaluation results"""
        avg_endurance = min(pos_result['endurance'], neg_result['endurance'])
        total_switching_time = pos_result['actual_switching_time'] + neg_result['actual_switching_time']
        frequency = get_frequency(total_switching_time)
        total_energy = pos_result['total_energy'] + neg_result['total_energy']
        avg_temperature = (pos_result['temperature'] + neg_result['temperature']) / 2
        
        merged_result = {
            'material': pos_result['material'],
            'pos_voltage': pos_result['voltage'],
            'neg_voltage': neg_result['voltage'],
            'avg_endurance': avg_endurance,
            'total_switching_time': total_switching_time,
            'pos_switching_time': pos_result['actual_switching_time'],
            'neg_switching_time': neg_result['actual_switching_time'],
            'frequency': frequency,
            'total_energy': total_energy,
            'avg_temperature': avg_temperature,
            'pos_result': pos_result,
            'neg_result': neg_result
        }
        
        return merged_result
        
    def _plot_combined_results(self, merged_result):
        pos_result = merged_result['pos_result']
        neg_result = merged_result['neg_result']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        ax_iv = axes[0, 0]
        pos_time = pos_result['time_real'].detach().cpu().numpy()
        pos_voltage = pos_result['voltage_real'].detach().cpu().numpy()
        pos_current = pos_result['pred_current_real'].detach().cpu().numpy()
        
        neg_time = neg_result['time_real'].detach().cpu().numpy()
        neg_voltage = neg_result['voltage_real'].detach().cpu().numpy()
        neg_current = neg_result['pred_current_real'].detach().cpu().numpy()
        
        ax_iv.semilogx(pos_time, pos_voltage, 'b-', label='SET Voltage')
        ax_iv.semilogx(neg_time, neg_voltage, 'b--', label='RESET Voltage')
        
        ax_ic = ax_iv.twinx()
        ax_ic.semilogx(pos_time, pos_current, 'r-', label='SET Current')
        ax_ic.semilogx(neg_time, neg_current, 'r--', label='RESET Current')
        
        ax_iv.set_xlabel('Time (s)')
        ax_iv.set_ylabel('Voltage (V)')
        ax_ic.set_ylabel('Current (A)')
        ax_iv.set_title('Voltage and Current vs Time')
        
        lines_iv, labels_iv = ax_iv.get_legend_handles_labels()
        lines_ic, labels_ic = ax_ic.get_legend_handles_labels()
        ax_iv.legend(lines_iv + lines_ic, labels_iv + labels_ic, loc='upper right')
        
        ax_gap = axes[0, 1]
        pos_gap = pos_result['gap'].detach().cpu().numpy()
        neg_gap = neg_result['gap'].detach().cpu().numpy()
        
        ax_gap.semilogx(pos_time, pos_gap, 'g-', label='SET Gap')
        ax_gap.semilogx(neg_time, neg_gap, 'g--', label='RESET Gap')
        ax_gap.set_xlabel('Time (s)')
        ax_gap.set_ylabel('Gap (m)')
        ax_gap.set_title('Gap vs Time')
        ax_gap.legend()
        
        ax_energy = axes[1, 0]
        # Recalculate cumulative energy for plotting consistency
        pos_energy = np.cumsum([pos_result['voltage_real'][i].item() * pos_result['pred_current_real'][i].item() * (pos_result['time_real'][i].item() - pos_result['time_real'][i-1].item()) 
                                for i in range(1, len(pos_time))])
        neg_energy = np.cumsum([neg_result['voltage_real'][i].item() * neg_result['pred_current_real'][i].item() * (neg_result['time_real'][i].item() - neg_result['time_real'][i-1].item())
                                for i in range(1, len(neg_time))])
        
        ax_energy.semilogx(pos_time[1:], pos_energy, 'm-', label='SET Energy')
        ax_energy.semilogx(neg_time[1:], neg_energy, 'm--', label='RESET Energy')
        ax_energy.set_xlabel('Time (s)')
        ax_energy.set_ylabel('Energy (J)')
        ax_energy.set_title('Cumulative Energy vs Time')
        ax_energy.legend()
        
        ax_summary = axes[1, 1]
        ax_summary.axis('off')
        summary_text = (
            f"Material: {merged_result['material']}\n\n"
            f"SET Voltage: {merged_result['pos_voltage']:.2f} V\n"
            f"RESET Voltage: {merged_result['neg_voltage']:.2f} V\n\n"
            f"Average Endurance: {merged_result['avg_endurance']:.2e} cycles\n"
            f"Frequency: {merged_result['frequency']:.2e} Hz\n"
            f"Total Switching Time: {merged_result['total_switching_time']*1e9:.2f} ns\n"
            f"Total Energy: {merged_result['total_energy']*1e12:.2f} pJ\n"
            f"Average Temperature: {merged_result['avg_temperature']:.2f} K\n\n"
            f"SET Switching Time: {pos_result['actual_switching_time']*1e9:.2f} ns\n"
            f"RESET Switching Time: {neg_result['actual_switching_time']*1e9:.2f} ns\n"
        )
        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, 
                        fontsize=12, verticalalignment='top')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{merged_result['material']}_cycle_analysis.png")
        plt.savefig(save_path)
        plt.close()
    
    def _print_result_summary(self, result):
        print("\n===== Evaluation Result Summary =====")
        print(f"Material: {result['material']}")
        print(f"SET Voltage: {result['pos_voltage']:.2f}V")
        print(f"RESET Voltage: {result['neg_voltage']:.2f}V")
        print(f"Average Endurance: {result['avg_endurance']:.2e} cycles")
        print(f"Frequency: {result['frequency']:.2e} Hz")
        print(f"Total Switching Time: {result['total_switching_time']*1e9:.2f} ns")
        print(f"Total Energy Consumption: {result['total_energy']*1e12:.2f} pJ")
        print(f"Average Temperature: {result['avg_temperature']:.2f} K")
        print(f"SET Switching Time: {result['pos_switching_time']*1e9:.2f} ns")
        print(f"RESET Switching Time: {result['neg_switching_time']*1e9:.2f} ns")
        print(f"Detailed results saved to: {self.output_dir}")
    
    def _save_result(self, result):
        import json
        import pandas as pd
        
        df = pd.DataFrame([{
            'material': result['material'],
            'pos_voltage': result['pos_voltage'],
            'neg_voltage': result['neg_voltage'],
            'avg_endurance': result['avg_endurance'],
            'frequency': result['frequency'],
            'total_switching_time': result['total_switching_time'],
            'total_energy': result['total_energy'],
            'avg_temperature': result['avg_temperature'],
            'pos_switching_time': result['pos_switching_time'],
            'neg_switching_time': result['neg_switching_time'],
        }])
        
        csv_path = os.path.join(self.output_dir, 'cycle_evaluation_result.csv')
        df.to_csv(csv_path, index=False)
        
        json_result = {k: v for k, v in result.items() if k not in ['pos_result', 'neg_result']}
        
        def serialize_tensor(data):
            # Recursively handle tensors in nested dicts
            if isinstance(data, dict):
                return {k: serialize_tensor(v) for k, v in data.items()}
            if isinstance(data, torch.Tensor):
                return data.tolist() if data.dim() else data.item()
            if isinstance(data, (np.ndarray, np.float32, np.float64)):
                return float(data)
            return data
            
        json_result['pos_result'] = serialize_tensor({k: v for k, v in result['pos_result'].items() 
                                                       if not isinstance(v, (torch.Tensor, np.ndarray))})
        json_result['neg_result'] = serialize_tensor({k: v for k, v in result['neg_result'].items() 
                                                       if not isinstance(v, (torch.Tensor, np.ndarray))})
        
        json_path = os.path.join(self.output_dir, 'cycle_evaluation_result.json')
        with open(json_path, 'w') as f:
            json.dump(json_result, f, indent=4, default=serialize_tensor)

    def _plot_sequence(self, idx, time, voltage, true_current, pred_current, gap, result, stable_idx, operation_type):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        time_np = time.detach().cpu().numpy()
        voltage_np = voltage.detach().cpu().numpy()
        true_current_np = true_current.detach().cpu().numpy()
        pred_current_np = pred_current.detach().cpu().numpy()
        gap_np = gap.detach().cpu().numpy()
        
        ax1.semilogx(time_np, voltage_np, 'b-', label='Voltage')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title(f'Material: {result["material"]}, {operation_type} Operation at {result["voltage"]:.2f}V')
        ax1.legend()
        ax1.grid(True)
        
        ax2.semilogx(time_np, true_current_np, 'g-', label='True Current')
        ax2.semilogx(time_np, pred_current_np, 'r--', label='Predicted Current')
        if stable_idx < len(time_np):
            ax2.axvline(x=time_np[stable_idx], color='m', linestyle='--', label='Stable Point')
            ax2.plot(time_np[stable_idx], pred_current_np[stable_idx], 'mo', markersize=8)
        ax2.set_ylabel('Current (A)')
        ax2.legend()
        ax2.grid(True)
        
        ax3.semilogx(time_np, gap_np, 'k-', label='Gap')
        if stable_idx < len(time_np):
            ax3.axvline(x=time_np[stable_idx], color='m', linestyle='--')
            ax3.semilogx(time_np[stable_idx], gap_np[stable_idx], 'mo', markersize=8)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Gap (m)')
        ax3.legend()
        ax3.grid(True)
        
        result_text = (
            f'Operation: {operation_type}\n'
            f'Endurance: {result["endurance"]:.2e} cycles\n'
            f'Switching time: {result["actual_switching_time"]*1e9:.2f} ns\n'
            f'Total energy: {result["total_energy"]*1e12:.2f} pJ\n'
            f'Temperature: {result["temperature"]:.2f} K'
        )
        plt.figtext(0.7, 0.01, result_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        save_path = os.path.join(self.output_dir, f'{result["material"]}_{operation_type}_{abs(result["voltage"]):.2f}V.png')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def get_energy_consumption(time_sequence, voltage_sequence, current_sequence, material_idx):
    materials = ['HfO2', 'Al2O3', 'TiO2']
    material = materials[material_idx]
    
    energy_thresholds = {
        'HfO2': [0.1e-12, 1e-12, 5e-12],
        'Al2O3': [0.1e-12, 1e-12, 5e-12],
        'TiO2': [0.1e-12, 1e-12, 5e-12]
    }
    
    if material not in energy_thresholds:
        print(f"Unknown material: {material}, using HfO2 thresholds")
        material = 'HfO2'
        
    threshold_times = {th: None for th in energy_thresholds[material]}

    cumulative_energy = torch.tensor(0.0, device=time_sequence.device)
    for i in range(1, len(time_sequence)):
        delta_t = time_sequence[i] - time_sequence[i-1]
        avg_power = voltage_sequence[i] * current_sequence[i]
        energy_step = avg_power * delta_t
        cumulative_energy += energy_step
        for threshold in energy_thresholds[material]:
            if threshold_times[threshold] is None and cumulative_energy.item() >= threshold:
                threshold_times[threshold] = time_sequence[i].item()
    
    return threshold_times, cumulative_energy

class EnduranceCalculator:
    def __init__(self):
        self.kb = 1.380649e-23        # Boltzmann constant [J/K]
        self.q = 1.60217663e-19       # Elementary charge [C]
        self.T0 = 273 + 25            # Ambient temperature [K]
        self.a0 = 0.25e-9             # Atomic distance [m]
        self.tox = 5e-9               # Oxide thickness [m]
        self.f0 = 1e13                # Attempt frequency [Hz]
        
        self.material_params = {
            'HfO2': {
                'Us': 1.2,            # Same as Eag in Constants (Switching barrier)
                'Uf': 1.6,            # Failure barrier > Us
                'Cth': 2.17e-17,
                'Tau_th': 3.5e-10       # Thermal time constant [s]
            },
            'Al2O3': {
                'Us': 1.0,             # Same as Eag in Constants
                'Uf': 1.6,             # Failure barrier > Us  
                'Cth': 2.12e-17,
                'Tau_th': 3.5e-10
            },
            'TiO2': {
                'Us': 1.7,            # Same as Eag in Constants
                'Uf': 2.3,            # Failure barrier > Us
                'Cth': 2.26e-17,
                'Tau_th': 3.5e-10
            }
        }
        self.current_material = 'HfO2'
        
    def calculate_temperature(self, current: float, voltage: float, dt) -> float:
        params = self.material_params[self.current_material]
        Cth = params['Cth']
        tau_th = params['Tau_th']
        
        if isinstance(dt, torch.Tensor):
            if dt.device.type != 'cpu':
                dt = dt.cpu()
            dt = dt.mean().item()
        else:
            dt = np.mean(dt)
            
        power = abs(voltage * current)
        # Simplified transient thermal model calculation
        delta_T = dt * (power/Cth + self.T0/tau_th) / (1 + dt/tau_th) /100 
        return self.T0 + delta_T

    def calculate_run_time(self, voltage: float, temperature: float) -> float:
        Us_joules = self.material_params[self.current_material]['Us'] * self.q  # Convert eV to Joules
        
        # Characteristic Run Time (ts) based on voltage and temperature
        return (2 * self.tox) / (self.f0 * self.a0) * \
            np.exp(Us_joules / (self.kb * temperature)) * \
            np.exp(-self.q * abs(voltage) * self.a0 / (2 * self.kb * temperature * self.tox))

    def calculate_failure_time(self, voltage: float, temperature: float) -> float:
        Uf_joules = self.material_params[self.current_material]['Uf'] * self.q  # Convert eV to Joules
        
        # Characteristic Failure Time (tf) based on voltage and temperature
        return (2 * self.tox) / (self.f0 * self.a0) * \
            np.exp(Uf_joules / (self.kb * temperature)) * \
            np.exp(-self.q * abs(voltage) * self.a0 / (2 * self.kb * temperature * self.tox))

    def calculate_endurance(self, current: float, voltage: float, time_step,
                              material: str = 'HfO2') -> Tuple[float, Dict]:
        self.update_material(material)
        
        temperature = self.calculate_temperature(current, voltage, time_step)
        ts = self.calculate_run_time(voltage, temperature)
        tf = self.calculate_failure_time(voltage, temperature)
        endurance = tf / ts

        metrics = {
            'temperature': temperature,
            'run_time': ts,
            'failure_time': tf,
            'endurance_cycles': endurance,
            'current': current,
            'voltage': voltage,
            'run_energy': abs(current * voltage * ts),
            'failure_energy': abs(current * voltage * tf)
        }

        return endurance, metrics

    def update_material(self, material: str):
        if material not in self.material_params:
            raise ValueError(f"Unknown material: {material}")
        self.current_material = material
        
def find_index(array, threshold, is_increasing):
    if is_increasing:
        indices = np.where(array >= threshold)[0]
        return indices[0] if len(indices) > 0 else -1
    else:
        indices = np.where(array <= threshold)[0]
        return indices[0] if len(indices) > 0 else -1
        
def get_stable_index(current_sequence):
    if isinstance(current_sequence, torch.Tensor):
        current_sequence_np = current_sequence.detach().cpu().numpy()
    else:
        current_sequence_np = np.asarray(current_sequence)

    end_current = current_sequence_np[-1]
    start_current = current_sequence_np[0]
    is_increasing = end_current > start_current
    delta_current = end_current - start_current
    
    # Define switching point as 90% of the total current change
    threshold_90 = start_current + 0.9 * delta_current
    
    idx_90 = find_index(current_sequence_np, threshold_90, is_increasing)
    return idx_90

def get_frequency(switching_time):
    # Frequency is the inverse of the total switching time (SET + RESET)
    frequency = 1.0 / switching_time
    
    max_frequency = 1e9 # Upper limit for frequency
    if frequency > max_frequency:
        # print(f"Warning: {frequency:.2e}Hz exceeds the upper limit, limited to {max_frequency}Hz")
        frequency = max_frequency
    
    return frequency

def main():
    parser = argparse.ArgumentParser(description='RRAM model evaluation tool')
    parser.add_argument('--model_path', type=str, default='results/uniform_data/pde/seed_122/best_mean_error_checkpoint.pth', help='model checkpoint path')
    parser.add_argument('--data_path', type=str, default='data/rram_sequences_asu_final_v2_full.mat', help='dataset path')
    parser.add_argument('--output_dir', type=str, default='test_results', help='output directory')
    parser.add_argument('--device', type=str, default='cuda', help='calculation device (cuda/cpu)')
    parser.add_argument('--material', type=str, required=True, choices=['HfO2', 'Al2O3', 'TiO2'], help='material type')
    parser.add_argument('--pos_voltage', type=float, required=True, help='positive voltage value (V)')
    parser.add_argument('--neg_voltage', type=float, required=True, help='negative voltage value (V)')
    
    args = parser.parse_args()
    
    evaluator = RRAMEvaluator(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        device=args.device
    )
    try:
        result = evaluator.evaluate(args.material, args.pos_voltage, args.neg_voltage)
        evaluator._print_result_summary(result)
        evaluator._save_result(result)
        evaluator._plot_combined_results(result)
        print("\nEvaluation completed successfully!")
    except Exception as e:
        print(f"\nError occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # This main function is primarily for standalone testing/manual use.
    # In the generative flow, RRAMEvaluator is instantiated by RRAMCTGANRecommender.
    main()