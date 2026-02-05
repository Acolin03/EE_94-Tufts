# PI-GAN for BCI-Optimized Non-Volatile Memory

This project introduces a **Physics-Informed Generative Adversarial Network (PI-GAN)** framework for the autonomous discovery and optimization of **Resistive Random Access Memory (ReRAM)** devices. By integrating a **Physics-Informed Neural Network (PINN)** as a real-time "Referee," the system transcends traditional grid-based simulations to explore high-density performance manifolds for Brain-Computer Interface (BCI) hardware.

## Key Innovations

* **Adversarial Discovery Engine**: Replaces the baseline CVAE with a **Conditional Tabular GAN (CTGAN)** capable of generating 10,000+ physically vetted design candidates.
* **Dual-Adversarial Logic**: A hybrid verification system where candidates must pass both a **Statistical Discriminator** (for data realism) and a **PINN Referee** (for thermodynamic consistency with Arrhenius filamentary laws).
* **Robustness via Density**: Utilizes **Alpha-Channel Density** in Pareto visualizations to distinguish between **"Stochastic Convergence"** (robust design windows) and **"Physical Outliers"** (sensitive peak performance).
* **Round-Robin Balanced Training**: Implements a material-agnostic training protocol (3,333 points per oxide) to eliminate bias and enable the autonomous discovery of emergent material preferences like the **$Al_2O_3$ Robust Window**.



## Execution Workflow

All scripts should be run from within the `/rram` directory. The workflow consists of two phases: Training (Expansion) and Discovery (Pareto).

### Phase 1: Model Training & Synthetic Expansion (10k Teachers)
This phase uses the PINN to simulate 10,000 balanced physical sequences to train the CTGAN. The `--create_new_dataset` flag ensures the 10k "Teachers" are generated before training begins.

```bash
python run_ctgan_gen.py --train_ctgan --create_new_dataset --no-generate-pareto \
    --ctgan_epochs 300 \
    --max_rows_per_file 10000 \
    --output_dir ctgan_recommendation_results \
    --model_path checkpoints/pinn_sparse.pth \
    --data_path data/rram_stanford.mat \
    --ctgan_model_path ctgan_recommendation_results/ctgan_model.pkl \
    --dataset_path ctgan_recommendation_results/rram_ctgan_dataset.pt

Once trained, use this command to generate 10,000 "Student" candidates. Each point is vetted by the **PINN Referee** in real-time to ensure physical validity before plotting.

```bash
python run_ctgan_gen.py --generate_pareto \
    --diverse_candidates 10000 \
    --output_dir ctgan_recommendation_results \
    --model_path checkpoints/pinn_sparse.pth \
    --data_path data/rram_stanford.mat \
    --ctgan_model_path ctgan_recommendation_results/ctgan_model.pkl \
    --dataset_path ctgan_recommendation_results/rram_ctgan_dataset.pt