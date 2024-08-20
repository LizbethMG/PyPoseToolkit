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
from activity_analysis import *
from results_manager import *

import sys
import os
import pandas as pd

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# --------------------------------------
# ----------> To define by user <--------
# Initialize variables and flags
# --------------------------------------
single_experiment = False  # 0 for all experiments found in the csv file, 1 for specific experiment
isElectrophySyncEnabled = 0  # 1 when sync with electrophysiology is necessary


fps = 25  # Video frames per seconds
zscore_threshold = 4  # for outlier removal
gap_threshold = 25  # for interpolation
window_size = 30  # for median filter smoothing
sigma = 2  # for Gaussian smoothing
s = 100  # for Spline smoothing
window_length = 30  # for Savitzky-Golay filter
polyorder = 3  # for Savitzky-Golay filter
sync_time = 60  # Sync signal in s (For example LED in video on)
window = { # Used to trim the data, define duration or start & end time otherwise 'None'
    "duration": 600,  # in s for trimming the data (10 min = 10 x 60 s)
    "start_time": None,  # in s for trimming the data (1 min = 1 x 60)
    "end_time":  None,  # in s for trimming the data (11 min = 11 x 60)
    "fps": 25,
}

threshold_method = 3  # 3-MAD method

# Colors
yellow = '#EDB120'
orange = "#D95319"
purple = "#7E2F8E"
green = "#77AC30"
light_blue = "#4DBEEE"


# Replace with the path to the CSV file containing the path to each experiment to analyze
csvFilePath = '//l2export/iss02.nerb/nerb-md/decimotiv/Decimotiv_Recording/DREADD_Project/1_PoseAnalysis' \
              '/Experiments_DLC.csv '
results_path = 'D:/OneDrive - ICM/Liz_NeuroMatlab/DLC_PoseAnalysis/Results'
results_filename = "results.csv"

# --------------------------------------
# Load experiments 
# --------------------------------------
print('/////////////////////////////////////////////////////////////////////////////////')


print('>   Reading file containing list of experiments:')
if os.path.exists(csvFilePath):
    print(f"        The file containing the list of experiments exists: \n        {csvFilePath}.")
    try:  # Read the CSV file into a DataFrame
        experiments_table = pd.read_csv(csvFilePath, sep=';')
        path_list = experiments_table.iloc[:, 4]
        print(f"        It lists {path_list.size} experiments.")
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        exit(1)
else:
    print(f"The file {csvFilePath} does not exists")
    exit(1)

# Calculate durations for all experiments using the imported function
durations, min_duration, max_duration = get_recording_durations(experiments_table, fps)
print(f"Durations for each experiment: {durations}")
print(f"Minimum duration: {min_duration} min")
print(f"Maximum duration: {max_duration} min")


# For the moment this scripts parts from the fact that centroid calculation was done elsewhere
# and saved in a csv file

# --------------------------------------
# Single experiment
# --------------------------------------
if single_experiment:
    current_experiment = Experiment(
        animal=755,
        compound="21",
        dose="1",  # mg/kg
        timepoint=0,  # hours
        experiments_table=experiments_table,
    )

    print('Single experiment selected:')
    print(f">   Pre-processing:")
    x = current_experiment.point_positions_extended['x_centroid']
    y = current_experiment.point_positions_extended['y_centroid']

    x1, y1, outlier_stats = slmg_remove_outliers(x, y, zscore_threshold, plot=False)
    x2, y2, interpol_stats = slmg_interpolate(x1, y1, gap_threshold, plot=False)
    x3, y3, smooth_stats = slmg_gaussian_smooth(x2, y2, sigma, plot=False)
    # x3, y3, smooth_stats = slmg_spline_smooth(x2, y2, s)
    # x3, y3, smooth_stats = slmg_median_smooth(x2, y2, window_size, plot=False)

    # Pass the required stats for the recap function
    x_percentageNaNs = outlier_stats['x_percentageNaNs']
    y_percentageNaNs = outlier_stats['y_percentageNaNs']
    xs_percentageNaNs = smooth_stats['x_percentageNaNs']
    ys_percentageNaNs = smooth_stats['y_percentageNaNs']

    recap_results = slmg_recap_preprocessing(x, y, x3, y3,
                                             x_percentageNaNs, y_percentageNaNs,
                                             xs_percentageNaNs, ys_percentageNaNs,
                                             fps, plot=False, summary=False)
    # Compute instant speed
    print(f">   Compute instant speed:")
    print('*       Verify more than one camara used:')
    if hasattr(current_experiment, 'cam_used'):
        print("           More than one camara used.")
        x4, y4, y4_MeanSpeed, y4_NanRate, = slmg_inst_speed(x3, y3, fps, current_experiment.cam_used, plot=False)
    else:
        print("           Only one camara used.")
        x4, y4, y4_MeanSpeed, y4_NanRate, = slmg_inst_speed(x3, y3, fps, cam_used=None)
    # Smoothing using median filter
    y5, y5_NanRate = slmg_median_smooth_2(x4, y4, window_size, plot=False)
    recap_results_2 = slmg_recap_preprocessing_2(y4, y5, y4_NanRate, y5_NanRate, fps)
    # sync_time = slmg_get_sync_time(current_experiment)
    # Recalibrate data
    y6 = slmg_recalibrate_data(y5, fps, sync_time)
    # Trims data
    y7 = slmg_window_data(y6, window["start_time"], window["end_time"], window["duration"], window["fps"])
    # Segmentation: high vs. low speed
    y8 = slmg_computeActivityLevels(y7, fps, threshold_method, current_experiment)

    # Statistics and Metrics
# --------------------------------------
# Multiple experiments to analyze listed in a csv file
# --------------------------------------
elif not single_experiment:
    # Check if the experiment results file already exist if nt it creates it
    slmg_results_file_exist(results_path, results_filename)

    for i in path_list.index:
        print(f" *******************************************************************************")
        print(f" Multiple experiment selected. Current experiment:  {i + 1} / {path_list.size}: ")
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

        print(f">   Pre-processing:")
        x = current_experiment.point_positions_extended['x_centroid']
        y = current_experiment.point_positions_extended['y_centroid']

        x1, y1, outlier_stats = slmg_remove_outliers(x, y, zscore_threshold, plot=False)
        x2, y2, interpol_stats = slmg_interpolate(x1, y1, gap_threshold, plot=False)
        x3, y3, smooth_stats = slmg_gaussian_smooth(x2, y2, sigma, plot=False)

        # Pass the required stats for the recap function
        x_percentageNaNs = outlier_stats['x_percentageNaNs']
        y_percentageNaNs = outlier_stats['y_percentageNaNs']
        xs_percentageNaNs = smooth_stats['x_percentageNaNs']
        ys_percentageNaNs = smooth_stats['y_percentageNaNs']

        recap_results = slmg_recap_preprocessing(x, y, x3, y3,
                                                 x_percentageNaNs, y_percentageNaNs,
                                                 xs_percentageNaNs, ys_percentageNaNs,
                                                 fps, plot=False, summary=False)
        # Compute instant speed
        print(f">   Compute instant speed:")
        print('        Verify more than one camara used:')
        if hasattr(current_experiment, 'cam_used'):
            print("           More than one camara used.")
            x4, y4, y4_MeanSpeed, y4_NanRate, = slmg_inst_speed(x3, y3, fps, current_experiment.cam_used, plot=False)
        else:
            print("           Only one camara used.")
            x4, y4, y4_MeanSpeed, y4_NanRate, = slmg_inst_speed(x3, y3, fps, cam_used=None)

        # Smoothing using median filter
        y5, y5_NanRate = slmg_median_smooth_2(x4, y4, window_size, plot=False)
        recap_results_2 = slmg_recap_preprocessing_2(y4, y5, y4_NanRate, y5_NanRate, fps)
        # Recalibrate data
        y6 = slmg_recalibrate_data(y5, fps, sync_time)
        # Trims data
        y7 = slmg_window_data(y6, window["start_time"], window["end_time"], window["duration"], window["fps"])
        # Segmentation: high vs. low speed
        result = slmg_computeActivityLevels(y7, fps, threshold_method, current_experiment, plot=True)
        # Append new results to the results CSV file
        slmg_append_results(results_path, results_filename, current_experiment, sync_time,
                            window, threshold_method, result)

else:
    print(f"Error selecting experiment to analyze")

print('done!')
print('/////////////////////////////////////////////////////////////////////////////////')
