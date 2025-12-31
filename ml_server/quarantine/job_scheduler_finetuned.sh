#!/bin/bash
#SBATCH --job-name=finetuned_eval       # Job name
#SBATCH --output=finetuned_output_%j.log # Standard output and error log
#SBATCH --partition=gpu                  # Partition (gpu)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=32               # Number of CPUs per task
#SBATCH --mem=64G                        # Memory per node
#SBATCH --time=01:00:00                  # Time limit (HH:MM:SS)
#SBATCH --gres=gpu:h100:1                # Request 1 GPU

# Load required modules
module load conda
conda init
source ~/.bashrc
conda activate our_torch_env

# Install required packages
pip install safetensors --quiet

# Define directories
HOME_DIR=$HOME
SCRATCH_DIR=$SCRATCH

# Navigate to the scratch directory
cd $SCRATCH_DIR

# Print GPU information
echo "===== GPU INFORMATION ====="
nvidia-smi
echo "=========================="

# List model files for debugging
echo "Listing model files..."
ls -la $HOME/synthesis-II/marianNMT_automobiles/
echo "=========================="

# Copy test data to scratch
cp $HOME/synthesis-II/test_set.csv $SCRATCH_DIR/
if [ $? -ne 0 ]; then
    echo "Failed to copy test_set.csv to $SCRATCH_DIR. Exiting..."
    exit 1
fi

# Copy the Python script to scratch
cp $HOME/synthesis-II/finetuned_model_eval.py $SCRATCH_DIR/
if [ $? -ne 0 ]; then
    echo "Failed to copy Python script to $SCRATCH_DIR. Exiting..."
    exit 1
fi

# Set unlimited stack size to handle large headers
ulimit -s unlimited

# Execute the Python script
python3 $SCRATCH_DIR/finetuned_model_eval.py

# Check if the Python script executed successfully
if [ $? -eq 0 ]; then
    echo "Python script executed successfully."
else
    echo "Python script failed. Exiting..."
    exit 1
fi

# Copy results to home directory
cp $SCRATCH_DIR/finetuned_marianmt_results.csv $HOME/synthesis-II/
if [ $? -ne 0 ]; then
    echo "Failed to copy results to $HOME/synthesis-II/. Exiting..."
    exit 1
fi

echo "Job completed. Results copied to $HOME/synthesis-II/finetuned_marianmt_results.csv"