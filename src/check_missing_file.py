import os
import pandas as pd




def check_missing_results_files(path_to_csv):
    # Initialize return values
    missing_files = []
    total_folders = 0

    # Try reading the CSV file with different options to handle potential issues
    try:
        df = pd.read_csv(path_to_csv, delimiter=';', quotechar='"', on_bad_lines='skip')
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return missing_files, total_folders, len(missing_files)

    # Extract the fifth column which contains the paths
    try:
        folder_paths = df.iloc[:, 4]  # 0-based index, 5th column is index 4
    except IndexError:
        print("The CSV file does not have a fifth column.")
        return missing_files, total_folders, len(missing_files)

    # Iterate over each folder path
    for folder_path in folder_paths:
        # Check if the folder_path is not null or empty
        if pd.notnull(folder_path):
            total_folders += 1
            # Construct the full paths to the expected files
            file1_path = os.path.join(folder_path, 'Results_M408.xlsx')
            file2_path = os.path.join(folder_path, 'Results_resnet50.xlsx')

            # Check if neither of the files exists
            if not (os.path.isfile(file1_path) or os.path.isfile(file2_path)):
                # Add the folder path to the missing_files list
                missing_files.append(folder_path)

    return missing_files, total_folders, len(missing_files)

# Example usage
path_to_csv = '//l2export/iss02.nerb/nerb-md/decimotiv/Decimotiv_Recording/DREADD_Project/1_PoseAnalysis/Experiments_DLC.csv'
missing_folders, total_folders, missing_count = check_missing_results_files(path_to_csv)

# Output results
print(f"Total folders checked: {total_folders}")
print(f"Number of folders missing both 'Results_M408.xlsx' and 'Results_resnet50.xlsx': {missing_count}")
print("Folders where neither 'Results_M408.xlsx' nor 'Results_resnet50.xlsx' is present:")
for folder in missing_folders:
    print(folder)