import os
import pandas as pd
import numpy as np

def get_centroid(path_to_experiment):
    """
    Analyzes CSV files from DeepLabCut and returns centroid and body part coordinates.
    
    This function processes CSV files from multiple cameras to determine the coordinates and likelihood of body parts
    and calculates the centroid coordinates for each frame. The function performs the following steps:
    
    1. Loads CSV files containing tracking data from the specified directory.
    2. Filters out data points with a likelihood (lk) less than 0.7 and computes corrected likelihood values.
    3. Determines the best camera view for each frame based on the highest likelihood and calculates body part coordinates accordingly.
    4. Computes the centroid of the tracked points (body parts) for each frame, considering only points with lk > 0.7.
    5. Computes summary metrics such as the number of visible body parts per frame and the percentage of frames each camera was chosen.
    6. Saves the results in Excel files including summary, centroid coordinates, analysis tables, and visibility metrics.
    
    Inputs:
    - path_to_experiment: Path to the directory containing the CSV files to be analyzed.
    
    Outputs:
    - Cc: DataFrame containing the centroid coordinates and the points used for the centroid calculation for each frame.
    - Sry: DataFrame summarizing the chosen camera for each frame, coordinates of body parts from the best camera, and other metrics.
    - Cc_array: Array form of the centroid coordinates.
    
    The Excel files created are named 'Results<experiment>.xlsx' and are
    saved in the specified directory. Where <experiment> corresponds to the unique identifier for each experiment. These files are saved in the directory specified by path_to_experiment.
    The length or duration of the video analyzed can be determined from the 'Frames_ms' column in the 'Cc' table, which represents the time in milliseconds.
    """

    # 1. Initialize the output variables
    Cc = None
    Sry = None
    Cc_array = None

    # 2. Define headers for the tables 
    headers_table = ["Scorer", "Frames_ms", "Sn_x", "Sn_y", "Sn_lk", "lear_x", "lear_y", "lear_lk", "rear_x", "rear_y", "rear_lk", 
                     "he_x", "he_y", "he_lk", "bc_x", "bc_y", "bc_lk", "tb_x", "tb_y", "tb_lk", "lfrontp_x", "lfrontp_y", "lfrontp_lk",
                     "rfrontp_x", "rfrontp_y", "rfrontp_lk", "lbackl_x", "lbackl_y", "lbackl_lk", "rbackl_x", "rbackl_y", "rbackl_lk",
                     "lk_corr_sn", "lk_corr_lear", "lk_corr_rear", "lk_corr_he", "lk_corr_bc", "lk_corr_tb", "lk_corr_lfrontp", 
                     "lk_corr_rfrontp", "lk_corr_lbackl", "lk_corr_rbackl", "Sum_lk_corr"]
    
    headers_Sry = ["Scorer", "Frames_ms", "Ch?", "Cam_ID_if_same_lk", "Ch>3s", "Same_lk", "Sn_x", "Sn_y", "Sn_lk", "lear_x", 
                   "lear_y", "lear_lk", "rear_x", "rear_y", "rear_lk", "he_x", "he_y", "he_lk", "bc_x", "bc_y", "bc_lk", 
                   "tb_x", "tb_y", "tb_lk", "lfrontp_x", "lfrontp_y", "lfrontp_lk", "rfrontp_x", "rfrontp_y", "rfrontp_lk", 
                   "lbackl_x", "lbackl_y", "lbackl_lk", "rbackl_x", "rbackl_y", "rbackl_lk"]
    
    headers_Cc = ["Scorer", "Frames_ms", "x_centroid", "y_centroid", "Pt_1", "Pt_2", "Pt_3", "Pt_4", "Pt_5", "Pt_6", 
                  "Pt_7", "Pt_8", "Pt_9", "Pt_10"]
    
    # 3. Reading CSV Files: Reads all CSV files in the specified directory and stores them in Cell as tuples of the shortened file name and the DataFrame. 
    #   Two files corresponding to the output of DLC for two cameras
    # Sry : summary : choice of cam and coordinates given best camera 
    # Cc : centroid coordinates 
    # Cell : stores tables of coordinates obtained with each camera
    
    # All CSV files in the specified directory are listed 
    csvfiles = [f for f in os.listdir(path_to_experiment) if f.endswith('.csv')]
    Cell = []
    
    for csvfile in csvfiles:
        T = pd.read_csv(os.path.join(path_to_experiment, csvfile), skiprows=2)
        ix = [i for i, char in enumerate(csvfile) if char == '_']
        vidname = csvfile[:4] + csvfile[ix[3]:ix[4]]
        Cell.append((vidname, T))
    
    # 4. Corrected Likelihood Calculation.
    # 
    for vidname, T in Cell:  # k=nb of csv files ; j= nb of lk_corr to calculate ; i=row (frame)
        T['Frames_ms'] = T.iloc[:, 0].astype(float) / 0.025
        Tarray = T.values
        index = range(5, 33, 3)  # cols corresponding to lk in table 
        lk_corr = np.zeros((Tarray.shape[0], len(index)))

        for j, col_idx in enumerate(index):
            if Tarray[0, col_idx] > 0.7 and Tarray[1, col_idx] > 0.7:
                lk_corr[0, j] = 1
            else:
                lk_corr[0, j] = 0

            for i in range(1, Tarray.shape[0] - 1):
                if (Tarray[i, col_idx] > 0.7 and (Tarray[i - 1, col_idx] > 0.7 or Tarray[i + 1, col_idx] > 0.7)) or \
                   (Tarray[i - 1, col_idx] > 0.7 and Tarray[i + 1, col_idx] > 0.7):
                    lk_corr[i, j] = 1
                else:
                    lk_corr[i, j] = 0

            if Tarray[-2, col_idx] > 0.7 and Tarray[-1, col_idx] > 0.7:
                lk_corr[-1, j] = 1
            else:
                lk_corr[-1, j] = 0

        Sum_lk_corr = np.sum(lk_corr, axis=1)
        Tarray = np.hstack([Tarray, lk_corr, Sum_lk_corr[:, np.newaxis]])
        T = pd.DataFrame(Tarray, columns=headers_table)
        Cell[Cell.index((vidname, T))] = (vidname, T)
    
    return Cc, Sry, Cc_array

