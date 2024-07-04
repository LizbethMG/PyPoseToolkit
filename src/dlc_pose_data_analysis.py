"""
Analysis of Pose Data from DLC

Objective:
    This script analyzes pose data from a dual-camera setup (each camera facing the other) using DeepLabCut for animal movement studies. Key features include:

    - Polygon Construction: Transforms DLC pixel coordinates into a polygon representing the animal's body.
    - Centroid Calculation: Determines the central point of the body polygon for movement analysis.
    - Optimal Camera Selection: Automatically selects the camera with the best view of the animal.
    - Data Smoothing: Applies filtering to refine the output and reduce noise.
Input: CSV file containing all pose folders with the following structure by column
        Animal ID | Compound name | Dose in mg/kg | Path to DLC folder

Output Metrics:
    - Averaged Instant Speed: Calculates the animal's average speed over a set period.
    - Activity Level Classification: Categorizes the animal's activity as low, high, or NA (based on a movement threshold) and provides the percentage time spent in each category.

"""
#Add the src directory to the system path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import pandas as pd

from experiment import Experiment
from get_centroid import *


# Initialize variables and flags
isElectrophySyncEnabled = 0  # 1 when sync with electrophy is necessary
single_experiment = 0 # 0 for all experiments found in the csv file, 1 for specific experiment

# Replace with the path to the CSV file
csvFilePath = '//l2export/iss02.nerb/nerb-md/decimotiv/Decimotiv_Recording/DREADD_Project/1_PoseAnalysis/Experiments_DLC.csv'

if os.path.exists(csvFilePath):
    print(f"The file {csvFilePath} exists.")
else:
    print(f"The file {csvFilePath} does not exists")
    exit(1)

# Colors
yellow = '#EDB120'
orange = "#D95319"
purple = "#7E2F8E"
green = "#77AC30"
light_blue = "#4DBEEE"

mainDir = os.getcwd()
print('///////////////////////////////')
print('Loading the experiments list...')

# Read the CSV file into a DataFrame
try: 
    dataTable = pd.read_csv(csvFilePath, sep=';')
    print(dataTable.head())
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    exit(1)

# Assuming the paths to DLC output are in the 5th column, extract that column, convert the DataFrame column to a list
Experiments_list = dataTable.iloc[:, 4].tolist()

# Get variable names from the DataFrame if you need them for reference
variableNames = dataTable.columns.tolist()

print('done!')
# TODO: If the pose data has to be syncronized with an electrophysiology recording we need to access the TTL. Works for Intan acquired data.
if isElectrophySyncEnabled == 1:
    print('Syncronization between video recording and electrophysiology required.')
else:
    print('No video syncronization with electrophysiology required.')

'''From Deeplabcut files (camera 1 and camera 2 facing each other) the function extracts bodyparts, 
creates a polygom approximating the body contour ans calculates the centroid. '''

# For single experiment analysis, select the experiment to analyse
if single_experiment == 1:
    single_experiment = Experiment(
        animal = 755,
        compound = "21",
        dose = "1", # mg/kg
        timepoint = 0, # hours
        dataTable= dataTable
    ) 
    path_list = single_experiment.get_experiment_path()
    print(f"Experiment path: {path_list}")
elif single_experiment == 0:
    path_list = dataTable.iloc[:,4]
    print(f"The file contains {path_list.size} files to analyze.")
else: 
    print(f"Error selecting experiment to anaylse")

for i in path_list.index:
    print(f">   Starting to analyze experiment {i+1} / {path_list.size}: ")
    print(f">   Animal ID {dataTable.iloc[i,0]}, compound {dataTable.iloc[i,1]}, dose {dataTable.iloc[i,2]} mg/kg, timepoint {dataTable.iloc[i,3]} h post injection")

    path_to_experiment = path_list[i]

    #TODO: For the moment this scripts parts from the fact that centroid calculation was done elsewhere
    #Cc, Sry, Cc_array = get_centroid(path_to_experiment)
    


print(r'\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')