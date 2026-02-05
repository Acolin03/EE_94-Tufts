# PINNs for Non-Volatile Memory Simulation

This project leverages **Physics-Informed Neural Networks (PINNs)** to study and predict the behavior of non-volatile memory technologies, specifically **Resistive Random Access Memory (ReRAM)**. The primary objective is to utilize PINNs to solve the partial differential equations (PDEs) governing the physical processes within these devices, aiming to improve the understanding and prediction of their operational behavior.
This project utilizes **Physics-Informed Neural Networks (PINNs)** to simulate the behavior of **Resistive Random Access Memory (ReRAM)** devices by solving the governing partial differential equations (PDEs). Building upon this simulation capability, the project also incorporates a **Conditional Variational Autoencoder (CVAE)** to suggest optimal device parameters (material, operating voltages, pulse width) based on desired performance targets (e.g., endurance, speed, energy efficiency). The overall goal is to provide a framework for both understanding ReRAM physics and facilitating device design optimization.

## Project Overview

This project integrates two primary components to bridge the gap between physical simulation and design optimization for ReRAM devices:

1.  **Physics-Informed Simulation (PINNs)**:
    *   Focuses on modeling ReRAM devices using PINNs based on the physical mechanisms described by Jiang et al. (2016) [1]. This involves simulating the complex switching behavior driven by ionic migration and vacancy generation in the metal-oxide layer.
    *   **Modeling Approach**: We employ PINNs to solve the PDEs describing the conductive filament gap distance ($g$) evolution influenced by time ($t$), voltage ($V$), and temperature ($T$).
    *   **Key Equations**: The model incorporates ion kinetics, considering voltage-driven hopping and thermal activation (Eqs. (1)-(3) in [1]). Current flow ($I$) is modeled based on gap distance and voltage (Eq. (6) in [1]).

2.  **NN-Driven Parameter Generation (CVAE)**:
    *   Utilizes a Conditional Variational Autoencoder (`generative_model.py`) trained on simulation data to suggest optimal RRAM parameters (material, operating voltages, pulse width) that meet specific performance targets (e.g., endurance, frequency, energy).
    *   **Methodology**: The CVAE generates candidate parameter sets, which are then evaluated for their actual performance using the trained PINN model via the `RRAMEvaluator` (`evaluator.py`). This allows the system to suggest parameters likely to achieve the desired outcomes.
    *   **Goal**: To accelerate the device design cycle by automatically exploring the parameter space and identifying promising configurations.

*(Future work may extend this integrated framework to Ferroelectric memory devices.)*

## Requirements

This project uses [Conda](https://docs.conda.io/en/latest/) for managing dependencies. To set up the environment, you will need to have Conda installed.

### Installation

1.  **Clone the Repository**

    ```bash
    git clone git@github.com:TuftsECS/pinn_nvm.git
    cd pinn_nvm
    ```

2.  **Create and Activate the Conda Environment**

    Use the provided `rram/requirements.yaml` file to create the Conda environment. This will install all the necessary packages, including PyTorch with the correct CUDA version.

    ```bash
    conda env create -f rram/requirements.yaml
    ```

    After the environment is created, activate it:

    ```bash
    conda activate pigen
    ```

    Now you are ready to run the project scripts.

## Project Structure

The project is organized under a main `rram/` directory, which contains all the code, scripts, data, and results related to the RRAM simulation.

```
.
├── rram/
│   ├── main.py                 # Main script for training the PINN model
│   ├── run_gen.py              # Script for CVAE analysis and Pareto front generation
│   ├── requirements.yaml       # Conda environment file for dependencies
│   ├── data/                   # Contains the raw dataset files (e.g., rram_stanford.mat)
│   ├── checkpoints/            # Stores trained PINN model checkpoints (*.pth)
│   ├── recommendation_results/ # Default output directory for run_gen.py
│   └── src/                    # Source code for the models and utilities
│       ├── __init__.py
│       ├── data.py             # Data loading and preprocessing
│       ├── models.py           # PINN and MLP model definitions
│       ├── loss.py             # Loss functions for training
│       ├── training.py         # Training loops
│       ├── utils.py            # Utility functions
│       └── cvae/               # Sub-package for the CVAE model
│           ├── __init__.py
│           ├── generative_model.py # CVAE model and recommender logic
│           ├── metric_eval.py      # Evaluator to test parameters with PINN
│           └── pareto_front.py     # Pareto front plotting logic
├── .gitignore
└── README.md                 # This file
```

## Quick Start

1.  **Set up the environment**: Ensure you have created and activated the `pigen` Conda environment as described above.
2.  **Navigate to the `rram` directory**: All scripts are intended to be run from within the `rram` directory.

    ```bash
    cd rram
    ```

3.  **Run the main training script**:

    ```bash
    python main.py --exp_name <your_experiment_name> [options]
    ```

    **Key Arguments**:
    *   `--exp_name`: (Required) A unique name for your experiment run.
    *   `--epochs`: Number of training epochs (default: 10).
    *   `--learning_rate_pinn`: Learning rate for the PINN model optimizer (default: 1e-4).
    *   `--learning_rate_mlp`: Learning rate for the MLP model optimizer (default: 1e-4).
    *   `--hidden_size`: Number of neurons in hidden layers (default: 64).
    *   `--save_dir`: Directory to save results (default: 'results').
    *   `--data_path`: Path to the dataset file (default: 'data/rram_sequences.mat').
    *   `--use_pde`: Flag to include PDE loss in training (default: False).
    *   `--use_full_dataset`: Flag to use the entire dataset instead of a subset (default: False).
    *   `--voltage_stride`: Stride for sampling voltage sequences (default: 10).
    *   `--seed`: Random seed for reproducibility (default: 42).

4.  **Monitor Training**: The script will output metrics during training, including:
    *   Training and validation losses (MLP and PINN components).
    *   Validation accuracy.
    *   Mean and maximum errors for gap prediction.
    *   Current learning rates.

5.  **Checkpoints**: The best model checkpoints (based on validation mean error and accuracy) and logs will be saved under the `rram/checkpoints/` directory.

## Parameter Recommendation and Pareto Front Generation (`run_gen.py`)

The `run_gen.py` script is the main interface for leveraging the trained PINN model to either recommend optimal device parameters or visualize the performance landscape through a Pareto front plot. It integrates the capabilities of the CVAE-based generative model and the Pareto front analysis.

**Note**: This script should also be run from within the `rram/` directory.

**Workflow:**

1.  **(Optional) Train the CVAE**: If you need to train or retrain the generative model.
2.  **Generate Recommendations**: Use the trained CVAE to find optimal parameters (material, voltages) for your desired performance targets (endurance, switching time, energy).
3.  **Generate Pareto Front Plot**: Create a plot to visualize the trade-offs between key performance metrics (e.g., energy vs. latency) across a range of parameters.

---

### 1. Generating the Pareto Front Plot

To visualize the trade-offs between energy consumption and switching latency, you can generate a Pareto front plot. This helps in understanding the performance limits of the device technology.

```bash
python run_gen.py --generate_pareto \
                       --model_path <path_to_pinn_mlp_checkpoint.pth> \
                       --data_path <path_to_rram_dataset.mat> \
                       --output_dir <directory_to_save_plot>
```

-   `--generate_pareto`: This flag activates the plot generation mode.
-   The script will use the CVAE model to sample a wide range of parameters, evaluate them with the PINN model, and plot the results, highlighting the Pareto optimal points.
-   The plot (`rram_pareto_front.png` and `.pdf`) will be saved in the specified `--output_dir`.
-   You can also provide `--target_*` arguments to see your desired performance targets visualized on the plot.

---

### 2. Training the CVAE Model (Optional)

If you have a new PINN model or want to retrain the CVAE, run the following command. This step is necessary before generating recommendations.

```bash
python run_gen.py --train_cvae \
                       --model_path <path_to_pinn_mlp_checkpoint.pth> \
                       --data_path <path_to_rram_dataset.mat> \
                       --cvae_model_path <path_to_save_cvae_model.pth> \
                       --dataset_path <path_to_save_cvae_dataset.pt>
```

-   `--train_cvae`: Flag to enable CVAE training mode.
-   `--model_path`: Path to the PINN/MLP checkpoint (`.pth`) from `main.py`. This is crucial as the CVAE dataset is generated by evaluating this model.
-   `--cvae_model_path`: Path where the trained CVAE model will be saved.
-   `--dataset_path`: Path to save the dataset generated for CVAE training.
-   `--create_new_dataset`: (Optional) Force creation of a new CVAE dataset.

---

### 3. Generating Parameter Recommendations

Once the CVAE is trained, you can use it to get parameter recommendations for specific performance goals.

```bash
python run_gen.py --model_path <path_to_pinn_mlp_checkpoint.pth> \
                       --data_path <path_to_rram_dataset.mat> \
                       --cvae_model_path <path_to_trained_cvae_model.pth> \
                       --dataset_path <path_to_cvae_dataset.pt> \
                       --target_endurance 1e7 \
                       --target_switching_time 5e-9 \
                       --target_energy 1e-12
```

-   Provide the paths to the PINN model, CVAE model, and CVAE dataset.
-   Set your desired performance with the `--target_*` arguments:
    -   `--target_endurance`: Desired minimum endurance in cycles (e.g., `1e6`).
    -   `--target_switching_time`: Desired maximum switching time in seconds (e.g., `10e-9` for 10 ns).
    -   `--target_energy`: Desired maximum energy consumption per cycle in Joules (e.g., `5e-12` for 5 pJ).

The script will output the best parameter sets (material, voltages, etc.) and save the detailed results in `recommendations.json` within the `--output_dir`.

## Citation

If you use this code or find the methodology helpful, please consider citing the relevant works:
[placeholder]

