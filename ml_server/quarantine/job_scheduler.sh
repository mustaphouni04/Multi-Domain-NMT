#!/bin/bash
#SBATCH --job-name=evaluation_models       # Job name
#SBATCH --output=output_%j.log         # Standard output and error log (%j expands to job ID)
#SBATCH --partition=gpu               # Partition (choose std, fat, gpu, or express)
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=32             # Number of CPUs per task
#SBATCH --mem=16G                      # Memory per node (adjust as needed)
#SBATCH --time=02:00:00               # Time limit (HH:MM:SS)
#SBATCH --gres=gpu:1                  # Request 1 GPU

# Load required modules
module load conda
conda init
conda activate our_torch_env

# Define directories
HOME_DIR=$HOME                        # User home directory
DATA_DIR=$DATA                        # Data directory
SCRATCH_DIR=$SCRATCH                  # Scratch directory

# Navigate to the scratch directory
cd $SCRATCH_DIR

# Copy input data to scratch
cp $HOME/synthesis-II/dataset.csv $SCRATCH_DIR/

# Check if fail
if [ $? -ne 0 ]; then
    echo "Failed to copy input files to $SCRATCH_DIR. Exiting..."
    exit 1
fi

# Create a symbolic link in $HOME to point to the file in $SCRATCH
ln -sf $SCRATCH_DIR/dataset.csv $HOME/synthesis-II/dataset.csv
if [ $? -ne 0 ]; then
    echo "Failed to create symbolic link in $HOME. Exiting..."
    exit 1
fi

# Copy before running script
cp $HOME/synthesis-II/translation_model_evaluation.py $SCRATCH_DIR/
if [ $? -ne 0 ]; then
    echo "Failed to copy Python script to $SCRATCH_DIR. Exiting..."
    exit 1
fi

# Execute the Python script
python3 translation_model_evaluation.py

# Check if the Python script executed successfully
if [ $? -eq 0 ]; then
    echo "Python script executed successfully."
else
    echo "Python script failed. Exiting..."
    exit 1
fi

# Copy results to home
cp $SCRATCH_DIR/translation_evaluation_results.csv $HOME/
if [ $? -ne 0 ]; then
    echo "Failed to copy output files to $HOME. Exiting..."
    exit 1
fi

echo "Job completed. Results copied to $HOME."
