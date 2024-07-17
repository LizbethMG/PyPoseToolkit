"""
Analysis of Pose Data from DLC

Objective: This script analyzes pose data from a dual-camera setup (each camera facing the other) using DeepLabCut
for animal movement studies. Key features include:

    - Polygon Construction: Transforms DLC pixel coordinates into a polygon representing the animal's body.
    - Centroid Calculation: Determines the central point of the body polygon for movement analysis.
    - Optimal Camera Selection: Automatically selects the camera with the best view of the animal.
    - Data Smoothing: Applies filtering to refine the output and reduce noise.
Input: CSV file containing all pose folders with the following structure by column
        Animal ID | Compound name | Dose in mg/kg | Path to DLC folder

Output Metrics: - Averaged Instant Speed: Calculates the animal's average speed over a set period. - Activity Level
Classification: Categorizes the animal's activity as low, high, or NA (based on a movement threshold) and provides
the percentage time spent in each category. # .\.venv\Scripts\activate """
from experiment import Experiment
from pre_processing import *
import sys
import os
import pandas as pd

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# --------------------------------------
# Initialize variables and flags
# --------------------------------------
isElectrophySyncEnabled = 0  # 1 when sync with electrophysiology is necessary
single_experiment = 1  # 0 for all experiments found in the csv file, 1 for specific experiment
# Colors
yellow = '#EDB120'
orange = "#D95319"
purple = "#7E2F8E"
green = "#77AC30"
light_blue = "#4DBEEE"

# If multiple experiments: replace with the path to the CSV file containing the path to each experiment to analyze
csvFilePath = '//l2export/iss02.nerb/nerb-md/decimotiv/Decimotiv_Recording/DREADD_Project/1_PoseAnalysis' \
              '/Experiments_DLC.csv '

# --------------------------------------
# Load experiments 
# --------------------------------------
print('///////////////////////////////')
print('Loading the list of experiments...')
if os.path.exists(csvFilePath):
    print(f">   The file containing the list exists: {csvFilePath}.")
    try:  # Read the CSV file into a DataFrame
        experiments_table = pd.read_csv(csvFilePath, sep=';')
        path_list = experiments_table.iloc[:, 4]
        print(f">   It lists {path_list.size} experiments.")
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        exit(1)
else:
    print(f"The file {csvFilePath} does not exists")
    exit(1)
print('Done!')

# For the moment this scripts parts from the fact that centroid calculation was done elsewhere
# and saved in a csv file

# For single experiment analysis or manual selection , select the experiment to analyse
if single_experiment == 1:
    print('Single experiment selected:')
    single_experiment = Experiment(
        animal=755,
        compound="21",
        dose="1",  # mg/kg
        timepoint=0,  # hours
        experiments_table=experiments_table,
    )
    print(f">   Pre-processing:")
    x = single_experiment.point_positions_extended['x_centroid']
    y = single_experiment.point_positions_extended['y_centroid']
    zscore_threshold = 3
    window_size = 5
    gap_threshold = 10

    x1, y1, pre_proc_results1 = slmg_remove_outliers(x, y, zscore_threshold, plot=True)
    slmg_interpolate(x1, y1, gap_threshold, plot=True)

# Multiple experiments to analyze listed in a csv file 
elif single_experiment == 0:

    for i in path_list.index:

        print(f"Starting to analyze experiment {i + 1} / {path_list.size}: ")
        path_to_experiment = path_list[i]

        if os.path.exists(path_to_experiment):

            current_experiment = Experiment(
                animal=experiments_table.iloc[i, 0],
                compound=experiments_table.iloc[i, 1],
                dose=experiments_table.iloc[i, 2],  # mg/kg
                timepoint=experiments_table.iloc[i, 3],  # hours
                experiments_table=experiments_table,
            )


        else:
            print(f"The file {path_to_experiment} does not exists")
            exit(1)
else:
    print(f"Error selecting experiment to analyze")
print('done!')
print(r'\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
