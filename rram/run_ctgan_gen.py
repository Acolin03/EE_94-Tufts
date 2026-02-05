import argparse
import sys
import os
import torch
import json
import traceback

# Adjust sys path for local imports
# NOTE: This path adjustment assumes a specific directory structure.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the CTGAN-specific modules
# NOTE: Adjusted imports to reflect the new file names
# Import the logic from the old file
from src.ctgan.ctgan_model import RRAMCTGANRecommender, train_ctgan
# Import the NEW plot function from your updated file
from src.ctgan.ctgan_pareto_front_rram import generate_pareto_plot
from src.ctgan.ctgan_eval import RRAMEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description='RRAM CTGAN Model and Pareto Front Generation Tool')

    # General arguments
    parser.add_argument('--model_path', type=str, default='checkpoints/pinn_sparse.pth', help='Path to the trained PINN model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/rram_stanford.mat', help='Path to the RRAM dataset')
    parser.add_argument('--output_dir', type=str, default='ctgan_recommendation_results', help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for computation (cuda/cpu)')

    # Execution Mode
    # Default to True - use --no-generate-pareto to disable
    parser.add_argument('--generate_pareto', action='store_true', default=False, help='Generate and save the Pareto front plot.')
    parser.add_argument('--no-generate-pareto', dest='generate_pareto', action='store_false', help='Disable Pareto plot generation.')

    # CTGAN training and recommendation arguments
    ctgan_group = parser.add_argument_group('CTGAN Training and Recommendation')
    ctgan_group.add_argument('--train_ctgan', action='store_true', help='Train a new CTGAN model')
    ctgan_group.add_argument('--create_new_dataset', action='store_true', help='Create a new dataset for CTGAN training')
    ctgan_group.add_argument('--ctgan_model_path', type=str, default='ctgan_recommendation_results/ctgan_model.pkl', help='Path to CTGAN model (using .pkl)')
    ctgan_group.add_argument('--dataset_path', type=str, default='ctgan_recommendation_results/rram_ctgan_dataset.pt', help='Path to CTGAN dataset')
    ctgan_group.add_argument('--ctgan_epochs', type=int, default=300, help='Number of epochs for CTGAN training')
    ctgan_group.add_argument('--max_rows_per_file', type=int, default=10000, help='Max rows to sample from PINN dataset for CTGAN training')
    
    # Recommendation and Pareto specific arguments
    rec_group = parser.add_argument_group('Recommendation and Pareto Plot Parameters')
    rec_group.add_argument('--target_endurance', type=float, default=1e6, help='Target endurance (cycles)')
    rec_group.add_argument('--target_switching_time', type=float, default=10e-9, help='Target switching time (s), default 5ns')
    rec_group.add_argument('--target_energy', type=float, default=500e-12, help='Target energy consumption (J)')
    rec_group.add_argument('--min_pulse_width', type=float, default=1e-9, help='Minimum pulse width for recommendations (default: 1ns)')
    rec_group.add_argument('--diverse_candidates', type=int, default=10000, help='Number of diverse candidates for recommendation generation')
    rec_group.add_argument('--energy_penalty_factor', type=float, default=3.0, help='Penalty factor for exceeding target energy in recommendations.')
    rec_group.add_argument('--max_energy_error_ratio', type=float, default=0.0, help='Maximum allowed ratio of predicted energy over target energy. Set to -1 to disable.')
    rec_group.add_argument('--num_recommendations', type=int, default=5, help='Number of recommendations to generate')
    rec_group.add_argument('--num_sample_points', type=int, default=50, help='Number of sample points per material for Pareto plot (max 100)')

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # Default generate_pareto to True if neither flag was explicitly provided
    # Check if --no-generate-pareto was in sys.argv to determine if user explicitly disabled it
    if '--no-generate-pareto' not in sys.argv and '--generate_pareto' not in sys.argv:
        args.generate_pareto = True

    # --- Mode: Pareto Front Generation ---
    if args.generate_pareto:
        print("--- Generating Pareto Front Plot using CTGAN ---")
        try:
            # Capture recommendations from the plot function
            _, _, _, recommendations = generate_pareto_plot(
                model_path=args.model_path,
                data_path=args.data_path,
                output_dir=args.output_dir,
                cvae_model_path=args.ctgan_model_path,
                dataset_path=args.dataset_path,
                target_switching_time=args.target_switching_time,
                target_energy=args.target_energy,
                target_endurance=args.target_endurance,
                num_sample_points=args.num_sample_points,
                energy_penalty_factor=args.energy_penalty_factor,
                max_energy_error_ratio=float('inf') if args.max_energy_error_ratio < 0 else args.max_energy_error_ratio,
                min_pulse_width=args.min_pulse_width,
                diverse_candidates=args.diverse_candidates
            )
            
            # Save the new recommendations to a UNIQUE Pareto file
            if recommendations:
                filepath = os.path.join(args.output_dir, 'ctgan_pareto_recommendations.json')
                with open(filepath, 'w') as f:
                    json.dump(recommendations, f, indent=4)
                print(f"Final Pareto recommendations saved to {filepath}")
            
            print(f"Pareto plot generated and saved to {args.output_dir}/rram_pareto_FINAL.png")
            
        except Exception as e:
            print(f"\nError occurred during Pareto generation: {e}")
            traceback.print_exc()

    # --- Mode: Training and Recommendation ---
    else:
        print("--- Running CTGAN Parameter Recommendation ---")
        try:
            # ... [evaluator initialization code] ...
            evaluator = RRAMEvaluator(
                model_path=args.model_path,
                data_path=args.data_path,
                output_dir=args.output_dir,
                device=device
            )
            # ... [training code] ...
            if args.train_ctgan:
                train_ctgan(evaluator, args.ctgan_epochs, args.dataset_path, args.ctgan_model_path, args.create_new_dataset, args.max_rows_per_file, args.min_pulse_width)

            # ... [recommender code] ...
            recommender = RRAMCTGANRecommender(args.ctgan_model_path, evaluator, args.dataset_path, args.min_pulse_width)
            max_error_ratio_param = float('inf') if args.max_energy_error_ratio < 0 else args.max_energy_error_ratio
            recommendations = recommender.recommend_parameters(args.target_endurance, args.target_switching_time, args.target_energy, args.num_recommendations, args.diverse_candidates, args.energy_penalty_factor, max_error_ratio_param)

            if recommendations:
                # Save the initial "Teacher" recommendations
                filepath = os.path.join(args.output_dir, 'ctgan_training_recommendations.json')
                with open(filepath, 'w') as f:
                    json.dump(recommendations, f, indent=4)
                print(f"\nSaved training-phase recommendations to {filepath}")
            else:
                print("No suitable recommendations found for the given targets.")
                
        except Exception as e:
            print(f"\nError occurred during Recommendation process: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()