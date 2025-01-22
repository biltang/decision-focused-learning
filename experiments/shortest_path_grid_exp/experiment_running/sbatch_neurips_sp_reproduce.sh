#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --mail-user=yongpeng@usc.edu
#SBATCH --account=vayanou_651
#SBATCH --output=./outputs/slurmlogs/%x_%j.out
#SBATCH --error=./outputs/slurmlogs/%x_%j.err

module purge
module load gcc 

cd /home1/yongpeng/decision-focused-learning/experiments/shortest_path_grid_exp/experiment_running

eval "$(conda shell.bash hook)"
conda activate pyepo_dsl

python neurips_sp_reproduce.py