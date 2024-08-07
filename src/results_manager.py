import os
import pandas as pd

# Define the headers
header = [
    "ID", "Compound", "Dose (mg/kg)", "Post-injection (h)", "Original dur (min)",
    "Sync (s)", "Desired duration (s)", "Desired Start (s)", "Desired end (s)", "Threshold method",
    "High activity percentage", "Low activity percentage", "Occlusion percentage"
]


# Function to check if the results file exists and has all required columns
def slmg_results_file_exist(results_dir, file_name):
    print(">   Check if - Results - file exists:")
    file_path = os.path.join(results_dir, file_name)
    if not os.path.exists(file_path):  # Does not exists
        df = pd.DataFrame(columns=header)
        df.to_csv(file_path, index=False)
        print(f"        File {file_path} created with header.")
    else:  # Exists
        df = pd.read_csv(file_path)
        missing_columns = [col for col in header if col not in df.columns]
        if missing_columns:
            print(f"        File {file_path} is missing columns: {missing_columns}. Recreating the file.")
            df = pd.DataFrame(columns=header)
            df.to_csv(file_path, index=False)
        else:
            print(f"        File {file_path} exists and has all required columns.")


# Function to check if the experiment results already exist
def slmg_results_exist(results_dir, file_name, current_experiment):

    file_path = os.path.join(results_dir, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Returns True if there are matching rows, and False if not
        return not df[(df['ID'] == current_experiment.animal) &
                      (df['Compound'] == current_experiment.compound) &
                      (df['Dose (mg/kg)'] == current_experiment.dose) &
                      (df['Post-injection (h)'] == current_experiment.timepoint)].empty
    return False


# Function to append new results to the CSV file
def slmg_append_results(results_dir, file_name, current_experiment, sync_time, window, threshold_method, result):
    file_path = os.path.join(results_dir, file_name)

    if slmg_results_exist(results_dir, file_name, current_experiment):
        print(f"*   Results for ID {current_experiment.animal} already exist. Skipping.")
        return False
    else:
        new_result = {
            "ID": current_experiment.animal,
            "Compound": current_experiment.compound,
            "Dose (mg/kg)": current_experiment.dose,
            "Post-injection (h)": current_experiment.timepoint,
            "Original dur (min)": current_experiment.video_min,
            "Sync (s)": sync_time,
            "Desired duration (s)": window.get('duration', None),
            "Desired Start (s)": window.get('start_time', None),
            "Desired end (s)": window.get('end_time', None),
            "Threshold method": threshold_method,
            "Mean": result.get('mean', None),
            "Std deviation": result.get('std_dev', None),
            "High activity percentage": result.get('high_activity', None),
            "Low activity percentage": result.get('low_activity', None),
            "Occlusion percentage": result.get('occlusion', None),
            "ADR Low/High+Occ": result.get('adr_low_high_occlusion',None),
            "ADR Low/High": result.get('adr_low_high', None),
            "skewness_low": result.get('skewness_low', None),
            "skewness_high": result.get('skewness_high', None),
            'normalized entropy': result.get('normalized_entropy', None)
        }

        df = pd.DataFrame([new_result])
        with open(file_path, 'a', newline='') as f:
            df.to_csv(f, header=False, index=False)
        print(f"*   Results for ID {current_experiment.animal} saved.")
        return True