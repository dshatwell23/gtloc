import os
import shutil
from datetime import datetime


def create_results_dir(experiment_name):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d_%H-%M-%S')

    results_folder_name = f"{dt_string}_{experiment_name}"
    results_folder_path = os.path.join('results', results_folder_name)
    os.mkdir(results_folder_path)
    copy_configs(results_folder_path)

    print(f"Experiment folder: {results_folder_name}")

    return results_folder_path


def copy_configs(dst_path):
    # Create subdirectory for config files within the results directory
    config_folder_path = os.path.join(dst_path, "configs")
    if not os.path.exists(config_folder_path):
        os.mkdir(config_folder_path)

    # Path to the directory where your config files are stored
    # Assuming the configs directory is in the same directory as this script
    configs_directory = os.path.join(os.getcwd(), "configs")

    # Fetch all Python files in the configs directory
    config_files = [f for f in os.listdir(configs_directory) if f.endswith('.py')]

    # Copy all Python config files to the newly created subdirectory in the results directory
    for file_name in config_files:
        source_file_path = os.path.join(configs_directory, file_name)
        destination_file_path = os.path.join(config_folder_path, file_name)
        shutil.copy(source_file_path, destination_file_path)