import subprocess
import os
import sys
import time

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the paths to the scripts
SRC_DIR = os.path.abspath('src')
SCRIPTS_DIR = os.path.abspath('scripts')

PREPROCESS_DATA_SCRIPT = os.path.join(SRC_DIR, 'preprocess_data.py')
BUILD_VOCAB_SCRIPT = os.path.join(SRC_DIR, 'build_vocab.py')
TRAIN_SCRIPT = os.path.join(SRC_DIR, 'train_aarambh.py')
EVALUATE_SCRIPT = os.path.join(SRC_DIR, 'evaluate_aarambh.py')
START_SERVER_SCRIPT = os.path.join(SRC_DIR, 'main.py')

# Define scripts from the scripts folder
AUTOMATED_DATA_INGESTION_SCRIPT = os.path.join(SCRIPTS_DIR, 'automated_data_ingestion.py')
AUTOMATED_DATA_INGESTION_VERSIONING_SCRIPT = os.path.join(SCRIPTS_DIR, 'automated_data_ingestion_with_versioning.py')
CONTINUOUS_DATA_PIPELINE_SCRIPT = os.path.join(SCRIPTS_DIR, 'continuous_data_pipeline.py')
CONTINUOUS_TRAINING_SCRIPT = os.path.join(SCRIPTS_DIR, 'continuous_training.py')
ENSEMBLE_LEARNING_TRAINING_SCRIPT = os.path.join(SCRIPTS_DIR, 'ensemble_learning_training.py')
HYPERPARAMETER_TUNING_SCRIPT = os.path.join(SCRIPTS_DIR, 'hyperparameter_tuning.py')
INCREMENTAL_TRAINING_SCRIPT = os.path.join(SCRIPTS_DIR, 'incremental_training.py')
META_LEARNING_TRAINING_SCRIPT = os.path.join(SCRIPTS_DIR, 'meta_learning_training.py')
SELF_ASSESSMENT_TRAINING_SCRIPT = os.path.join(SCRIPTS_DIR, 'self_assessment_training.py')
TRAIN_RL_SCRIPT = os.path.join(SCRIPTS_DIR, 'train_rl.py')

def run_script(script_path, args=[]):
    """Runs a script with the given arguments and logs its progress."""
    command = [sys.executable, script_path] + args
    logger.info(f"Running script: {script_path} with arguments: {args}")
    try:
        subprocess.run(command, check=True)
        logger.info(f"Finished running script: {script_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running script: {script_path}")
        logger.error(e)
        sys.exit(1)

def main():
    """Main function to orchestrate the running of all necessary scripts."""
    logger.info("Starting the workflow...")

    # Step 1: Preprocess the data
    run_script(PREPROCESS_DATA_SCRIPT)

    # Step 2: Build the vocabulary
    run_script(BUILD_VOCAB_SCRIPT)

    # Step 3: Automated Data Ingestion (if needed)
    run_script(AUTOMATED_DATA_INGESTION_SCRIPT)

    # Step 4: Automated Data Ingestion with Versioning (if needed)
    run_script(AUTOMATED_DATA_INGESTION_VERSIONING_SCRIPT)

    # Step 5: Continuous Data Pipeline (if needed)
    run_script(CONTINUOUS_DATA_PIPELINE_SCRIPT)

    # Step 6: Train the model
    run_script(TRAIN_SCRIPT)

    # Step 7: Continuous Training (if needed)
    run_script(CONTINUOUS_TRAINING_SCRIPT)

    # Step 8: Hyperparameter Tuning (if needed)
    run_script(HYPERPARAMETER_TUNING_SCRIPT)

    # Step 9: Incremental Training (if needed)
    run_script(INCREMENTAL_TRAINING_SCRIPT)

    # Step 10: Meta Learning Training (if needed)
    run_script(META_LEARNING_TRAINING_SCRIPT)

    # Step 11: Self Assessment Training (if needed)
    run_script(SELF_ASSESSMENT_TRAINING_SCRIPT)

    # Step 12: Train Reinforcement Learning (if needed)
    run_script(TRAIN_RL_SCRIPT)

    # Step 13: Ensemble Learning Training (if needed)
    run_script(ENSEMBLE_LEARNING_TRAINING_SCRIPT)

    # Step 14: Evaluate the model
    run_script(EVALUATE_SCRIPT)

    # Step 15: Start the FastAPI server
    run_script(START_SERVER_SCRIPT)

if __name__ == "__main__":
    main()
