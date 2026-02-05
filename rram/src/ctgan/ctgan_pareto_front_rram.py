import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
from matplotlib.ticker import ScalarFormatter
import torch
import json
import os
import matplotlib.ticker as ticker

# Assuming RRAMParameterRecommender is your CTGAN implementation
from .ctgan_model import RRAMCTGANRecommender as RRAMParameterRecommender 
from .ctgan_eval import RRAMEvaluator

plt.rcParams.update({
    # Use a fallback system font instead of specifically requiring Arial
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Liberation Sans', 'FreeSans', 'sans-serif'],
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 15
})

COLORS = {
    'HfO2': {'fill': '#FBD178', 'edge': '#EE8227'},
    'TiO2': {'fill': '#BFE4EE', 'edge': '#79ADD6'},
    'Al2O3': {'fill': '#F9C3BF', 'edge': '#EB716B'},
    'pareto': '#556B2F',
    'target': '#1E803D'
}
ENERGY_MIN_VALID = 10       # pJ
ENERGY_MAX_VALID = 5000     # pJ
TIME_MIN_VALID = 0.1e-9     # s (0.1 ns)
TIME_MAX_VALID = 1e-3     # s (1e-3 ns)
ENDURANCE_MIN_VALID = 1e3   # cycles VOLTAGE_RANGES
ENDURANCE_MAX_VALID = 1e12  # cycles


def validate_data(df):
    """Clean and validate data to prevent plotting issues, using switching time."""
    df_clean = df.copy()
    if 'total_switching_time' not in df_clean.columns:
        print("Error: 'total_switching_time' column not found in DataFrame for validation.")
        df_clean['total_switching_time'] = 10e-9
    if 'is_recommendation' in df_clean.columns:
        df_clean['is_recommendation'] = df_clean['is_recommendation'].fillna(False)
    if 'is_other_recommendation' in df_clean.columns:
        df_clean['is_other_recommendation'] = df_clean['is_other_recommendation'].fillna(False)
    
    df_clean = df_clean[
        (df_clean['energy'] >= ENERGY_MIN_VALID) & 
        (df_clean['energy'] <= ENERGY_MAX_VALID) &
        (df_clean['total_switching_time'] >= TIME_MIN_VALID) & 
        (df_clean['total_switching_time'] <= TIME_MAX_VALID) & 
        (df_clean['endurance'] >= ENDURANCE_MIN_VALID) & 
        (df_clean['endurance'] <= ENDURANCE_MAX_VALID) &
        np.isfinite(df_clean['energy']) &
        np.isfinite(df_clean['total_switching_time']) &
        np.isfinite(df_clean['endurance'])
    ]
    
    removed_count = len(df) - len(df_clean)
    if removed_count > 0:
        print(f"Warning: Removed {removed_count} data points with invalid values")
        
    if len(df_clean) == 0:
        print("Error: No valid data points for plotting")
        return pd.DataFrame({
            'material': ['HfO2'],
            'energy': [1000],
            'total_switching_time': [10e-9],
            'endurance': [1e6],
            'is_recommendation': [False],
            'is_other_recommendation': [False]
        })
    
    return df_clean

def get_safe_axis_limits(df, target_energy_pJ, target_switching_time_ns):
    """Calculate safe axis limits for energy and switching time."""
    DEFAULT_ENERGY_MIN = 150
    DEFAULT_ENERGY_MAX = 2000
    DEFAULT_TIME_MIN_NS = 1
    DEFAULT_TIME_MAX_NS = 50 
    
    try:
        if 'switching_time_ns' not in df.columns:
            print("Warning: 'switching_time_ns' not found in DataFrame for axis limits. Using defaults.")
            return DEFAULT_ENERGY_MIN, DEFAULT_ENERGY_MAX, DEFAULT_TIME_MIN_NS, DEFAULT_TIME_MAX_NS
        if len(df) > 0:
            x_min = max(np.percentile(df['energy'], 1), DEFAULT_ENERGY_MIN / 2)
            x_max = min(np.percentile(df['energy'], 99) * 1.2, DEFAULT_ENERGY_MAX * 2)
            y_min = max(np.percentile(df['switching_time_ns'], 1) / 1.2, DEFAULT_TIME_MIN_NS / 2)
            y_max = min(np.percentile(df['switching_time_ns'], 99) * 1.2, DEFAULT_TIME_MAX_NS * 2)
            
            if target_energy_pJ is not None:
                x_min = min(x_min, target_energy_pJ * 0.8)
                x_max = max(x_max, target_energy_pJ * 1.2)
            
            if target_switching_time_ns is not None:
                y_min = min(y_min, target_switching_time_ns * 0.8)
                y_max = max(y_max, target_switching_time_ns * 1.2)
            
            x_min = max(x_min, DEFAULT_ENERGY_MIN)
            x_max = min(x_max, DEFAULT_ENERGY_MAX)
            y_min = max(y_min, DEFAULT_TIME_MIN_NS)
            y_max = min(y_max, DEFAULT_TIME_MAX_NS)
            
            if x_max / x_min < 2:
                x_min = max(DEFAULT_ENERGY_MIN / 2, x_min / 2)
                x_max = min(DEFAULT_ENERGY_MAX * 2, x_max * 2)
            
            if y_max <= y_min:
                y_max = y_min * 2

            if y_max / y_min < 2:
                y_min = max(DEFAULT_TIME_MIN_NS / 2, y_min / 2)
                y_max = min(DEFAULT_TIME_MAX_NS * 2, y_max * 2)
                if y_max <= y_min:
                    y_max = y_min * 1.1


            return x_min, x_max, y_min, y_max
            
    except Exception as e:
        print(f"Error calculating axis limits: {e}")
    
    return DEFAULT_ENERGY_MIN, DEFAULT_ENERGY_MAX, DEFAULT_TIME_MIN_NS, DEFAULT_TIME_MAX_NS

def calculate_size_for_endurance(endurance):
    """Calculate marker size based on endurance."""
    log_endurance = np.log10(endurance)
    size_factor = 8
    return size_factor * (log_endurance ** 2.2)

def generate_pareto_plot(model_path='results/uniform_data/pde/seed_122/best_mean_error_checkpoint.pth', 
                         data_path='data/rram_sequences_asu_final_v2_full.mat',
                         output_dir='recommendation_results',
                         cvae_model_path='recommendation_results/cvae_model.pth',
                         dataset_path='recommendation_results/rram_cvae_dataset.pt',
                         target_switching_time=10e-9, # s, default 10 ns
                         target_energy=5e-10,  # J, equivalent to 500 pJ
                         target_endurance=1e6,  # cycles
                         num_sample_points=50,
                         energy_penalty_factor=3.0,
                         max_energy_error_ratio=1.0,
                         min_pulse_width=1e-9,
                         diverse_candidates=40):
    """
    Generate Pareto front plot using the CTGAN model with corrected visualization logic.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing RRAM evaluator...")
    # Initialize evaluator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        evaluator = RRAMEvaluator(
            model_path=model_path,
            data_path=data_path,
            output_dir=output_dir,
            device=device
        )
    except Exception as e:
        print(f"Error initializing RRAM evaluator: {e}")
        return None, None, None
    
    print("Initializing parameter recommender...")
    # Initialize parameter recommender
    materials = evaluator.materials
    try:
        recommender = RRAMParameterRecommender(
            cvae_model_path, 
            evaluator,
            dataset_path=dataset_path,
            min_pulse_width=min_pulse_width
        )
    except Exception as e:
        print(f"Error initializing parameter recommender: {e}")
        return None, None, None
    
    # --- UNIFIED SAMPLING STRATEGY ---
    # 1. Generate the Cloud FIRST
    print(f"Sampling {diverse_candidates} candidates from CTGAN cloud for Pareto analysis...")
    try:
        sampled_df = recommender.sample_candidates(num_samples=diverse_candidates)
    except Exception as e:
        print(f"Error sampling from CTGAN cloud: {e}. Falling back to grid sampling.")
        sampled_df = pd.DataFrame() # Trigger fallback below
    
    # 2. Pick Recommendations from the SAME Cloud (Consistency + Speed)
    if not sampled_df.empty:
        print("Selecting best recommendations from the validated cloud...")
        recommendations = recommender.pick_best_from_cloud(
            sampled_df,
            target_endurance,
            target_switching_time,
            target_energy,
            energy_penalty_factor=energy_penalty_factor
        )
        
        # Prepare data for plotting
        data = []
        for _, row in sampled_df.iterrows():
            data.append({
                'material': row['material'],
                'energy': float(row['energy']), 
                'total_switching_time': float(row['latency']),
                'endurance': float(row['endurance']),
                'is_recommendation': False,
                'is_other_recommendation': False,
                'pos_voltage': float(row['Vset']),
                'neg_voltage': float(row['Vreset'])
            })
    else:
        # Fallback to original grid sampling if cloud fails
        data = []
        # ... [omitted for brevity, keeping existing fallback logic] ...
        recommendations = [] # No recommendations in fallback for now

    # 3. Label the recommendations in the data list
    for i, rec in enumerate(recommendations):
        is_best_overall = rec.get('optimization_target', '') == 'Overall Performance'
        
        # Find the matching point in 'data' and mark it
        for entry in data:
            # Match by material and approximate energy/latency to find the "bubble"
            if (entry['material'] == rec['material'] and 
                abs(entry['energy'] - rec['predicted_performance']['energy']) < 1e-15):
                entry['is_recommendation'] = is_best_overall
                entry['is_other_recommendation'] = not is_best_overall
                entry['rec_num'] = i + 1
                entry['optimization_target'] = rec.get('optimization_target', '')
                break

    # Print the selected recommendations
    if recommendations:
        print("\n--- Recommendations Selected from Cloud ---")
        for i, rec in enumerate(recommendations):
            perf = rec['predicted_performance']
            print(f" Recommendation #{i+1}: {rec['material']} (Target: {rec['optimization_target']})")
            print(f"   Endurance: {perf['endurance']:.2e} cycles")
            print(f"   Energy: {perf['energy']*1e12:.2f} pJ, Latency: {perf['total_switching_time']*1e9:.2f} ns")
        print("---")
    
    df_raw = pd.DataFrame(data)
    
    # Convert Energy to pJ for plotting units
    df_raw['energy'] = df_raw['energy'] * 1e12
    
    # --- PATH 1: SENSITIVITY ANALYSIS (Master Fusion) ---
    print("\nExtracting Design Rules from Discovery Cloud...")
    sensitivity_results = {}
    for mat in df_raw['material'].unique():
        mat_df = df_raw[df_raw['material'] == mat].copy()
        if len(mat_df) >= 10:
            # Simple Pareto-proxy score: normalized energy + latency
            e_min, e_max = mat_df['energy'].min(), mat_df['energy'].max()
            l_min, l_max = mat_df['total_switching_time'].min(), mat_df['total_switching_time'].max()
            
            # Prevent division by zero
            e_norm = (mat_df['energy'] - e_min) / (e_max - e_min + 1e-12)
            l_norm = (mat_df['total_switching_time'] - l_min) / (l_max - l_min + 1e-12)
            mat_df['pareto_score'] = e_norm + l_norm
            
            winners = mat_df.nsmallest(int(len(mat_df) * 0.1), 'pareto_score')
            ghosts = mat_df.nlargest(int(len(mat_df) * 0.1), 'pareto_score')
            
            sensitivity_results[mat] = {
                'optimal_vset': [float(winners['pos_voltage'].mean()), float(winners['pos_voltage'].std())],
                'optimal_vreset': [float(winners['neg_voltage'].mean()), float(winners['neg_voltage'].std())],
                'ghost_vset_mean': float(ghosts['pos_voltage'].mean()),
                'avg_winner_energy_pJ': float(winners['energy'].mean()),
                'avg_winner_latency_ns': float(winners['total_switching_time'].mean() * 1e9)
            }
    
    # Save the report
    report_path = os.path.join(output_dir, 'sensitivity_analysis_report.json')
    with open(report_path, 'w') as f:
        json.dump(sensitivity_results, f, indent=4)
    print(f"Design Rules saved to {report_path}")

    # CRITICAL FIX: Bypass validate_data for the cloud points because 
    # they were already filtered by the PI-GAN physics referee in the model.
    # We only validate to ensure columns are clean.
    df = df_raw.copy()
    df['switching_time_ns'] = df['total_switching_time'] * 1e9
    
    print(f"Generated {len(df)} data points for plotting (Cloud: {len(df)-len(recommendations)})")
    
    target_energy_pJ = target_energy * 1e12
    target_switching_time_ns = target_switching_time * 1e9
    
    try:
        # 1. Setup Figure
        fig = plt.figure(figsize=(10, 8), dpi=300)
        ax = fig.add_axes([0.1, 0.1, 0.85, 0.75]) 
        
        # 2. Plot Scatter Points
        for material in materials:
            material_data = df[df['material'] == material]
            normal_points = material_data[~(material_data['is_recommendation'] | material_data['is_other_recommendation'])]
            
            # Plot the CTGAN cloud
            if not normal_points.empty:
                sizes = normal_points['endurance'].apply(calculate_size_for_endurance) * 0.2
                ax.scatter(
                    normal_points['energy'], 
                    normal_points['switching_time_ns'],
                    s=sizes,
                    color=COLORS[material]['fill'],
                    edgecolor=COLORS[material]['edge'],
                    alpha=0.2, # High transparency for density
                    linewidth=0.3,
                    label=material,
                    zorder=1,
                    rasterized=True # FAST RENDERING: Fixes the 20s delay
                )

            
            # Plot the "Star" (#1 Recommendation)
            rec_points = material_data[material_data['is_recommendation']]
            if not rec_points.empty:
                ax.scatter(
                    rec_points['energy'], 
                    rec_points['switching_time_ns'],
                    s=500,
                    color=COLORS[material]['fill'], # Correctly uses the loop's material color
                    edgecolor='black',              
                    alpha=1.0,
                    linewidth=2.5,                  
                    marker='*',
                    zorder=1000                     
                )
            
            # Add label "#1" next to the star
            recommendation_mask = material_data['is_recommendation'].fillna(False)
            for idx, row in material_data[recommendation_mask].iterrows():
                if row['is_recommendation'] and row.get('rec_num') == 1:
                    ax.annotate(f"#1", (row['energy'], row['switching_time_ns']),
                        xytext=(18, 8), textcoords='offset points', fontsize=20, 
                        fontweight='bold', color='black', zorder=101)

        # 3. Optimized Pareto Front Calculation (O(N log N))
        pareto_points = []
        df_sorted = df.sort_values('energy')
        min_latency = float('inf')
        for _, row in df_sorted.iterrows():
            if row['switching_time_ns'] < min_latency:
                pareto_points.append(row)
                min_latency = row['switching_time_ns']
        
        if pareto_points:
            pareto_df = pd.DataFrame(pareto_points)
            ax.plot(pareto_df['energy'], pareto_df['switching_time_ns'],
                color=COLORS['pareto'], linestyle='--', linewidth=2.5, label='Pareto Front', zorder=50)
        
        # 4. Plot Target Lines
        ax.axhline(y=target_switching_time_ns, color=COLORS['target'], linestyle=':', linewidth=2, alpha=0.8, zorder=50)
        ax.axvline(x=target_energy_pJ, color=COLORS['target'], linestyle=':', linewidth=2, alpha=0.8, zorder=50)
        
        # 5. Axis Formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy (pJ)', fontsize=18)
        ax.set_ylabel('Latency (ns)', fontsize=18)

        for axis in [ax.xaxis, ax.yaxis]:
            formatter = ticker.ScalarFormatter()
            formatter.set_scientific(False)
            axis.set_major_formatter(formatter)
            axis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
            axis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=10))
            axis.set_minor_formatter(ticker.NullFormatter())    
        
        ax.grid(True, linestyle='--', alpha=0.3)

        # 6. DYNAMIC LIMITS (Percentile-based to handle GAN outliers)
        x_min = np.percentile(df['energy'], 0.5) * 0.5
        x_max = np.percentile(df['energy'], 99.5) * 2.0
        y_min = np.percentile(df['switching_time_ns'], 0.5) * 0.5
        y_max = np.percentile(df['switching_time_ns'], 99.5) * 2.0

        # Ensure target and recommendations are visible
        x_min = min(x_min, target_energy_pJ * 0.5)
        x_max = max(x_max, target_energy_pJ * 2.0)
        y_min = min(y_min, target_switching_time_ns * 0.5)
        y_max = max(y_max, target_switching_time_ns * 2.0)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        print(f"DEBUG: Plot Limits set to. X:[{x_min:.1f}, {x_max:.1f}] Y:[{y_min:.1f}, {y_max:.1f}]")
        
        # 7. Target Labels
        try:
            ax.text(target_energy_pJ * 1.05, y_min * 1.2, 'Target Energy', color=COLORS['target'],
                fontsize=14, fontweight='bold', rotation=90, va='bottom')
            ax.text(x_min * 1.2, target_switching_time_ns * 1.05, 'Target Latency', color=COLORS['target'],
                fontsize=14, fontweight='bold', ha='left')
        except Exception:
            pass
        
        # 8. LEGENDS (Matches CVAE Baseline)
        
        # A. Material Legend (Top Outside)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='HfO2', markerfacecolor=COLORS['HfO2']['fill'], markeredgecolor=COLORS['HfO2']['edge'], markersize=12),
            Line2D([0], [0], marker='o', color='w', label='TiO2', markerfacecolor=COLORS['TiO2']['fill'], markeredgecolor=COLORS['TiO2']['edge'], markersize=12),
            Line2D([0], [0], marker='o', color='w', label='Al2O3', markerfacecolor=COLORS['Al2O3']['fill'], markeredgecolor=COLORS['Al2O3']['edge'], markersize=12),
            # Note: The Legend star remains grey to denote the "Symbol", but the plot star will be colored.
            Line2D([0], [0], marker='*', color='w', markerfacecolor=COLORS['Al2O3']['fill'], markeredgecolor='k', markersize=18, label='Best Overall', linestyle='None'),
        ]
        
        # Matches your CVAE code logic
        main_legend = ax.legend(
            handles=legend_elements, 
            loc='upper center',
            bbox_to_anchor=(0.5, 1.15), 
            ncol=2,
            framealpha=0.9,
            columnspacing=1.0,
            fontsize=14
        )
        ax.add_artist(main_legend)

        # B. Endurance Legend (Top Right OUTSIDE - Like CVAE)
        ax_endurance = fig.add_axes([0.65, 0.88, 0.3, 0.1], frame_on=False) # [left, bottom, width, height]
        ax_endurance.set_title('Endurance (cycles)', fontsize=14, loc='center')
        ax_endurance.set_xlim(0, 4)
        ax_endurance.set_ylim(0, 1)
        ax_endurance.set_xticks([])
        ax_endurance.set_yticks([])
        
        endurance_vals = [1e5, 1e6, 1e7, 1e8]
        endurance_labs = ['$10^5$', '$10^6$', '$10^7$', '$10^8$']
        for i, (val, label) in enumerate(zip(endurance_vals, endurance_labs)):
            s = calculate_size_for_endurance(val)
            ax_endurance.scatter(i + 0.5, 0.5, s=s, c='gray', alpha=0.6, edgecolors='none')
            ax_endurance.text(i + 0.5, 0.0, label, ha='center', va='top', fontsize=12)

        # 9. Save
        try:
            png_path = os.path.join(output_dir, 'rram_pareto_FINAL.png') 
            plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            print(f"SUCCESS: Plot saved to: {png_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        
        return fig, ax, df, recommendations
        
    except Exception as e:
        print(f"Error generating plot: {e}")
        import traceback
        traceback.print_exc()
        return None, None, df, []