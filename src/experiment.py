import glob2
import os
import pandas as pd
import numpy as np


class Experiment:
    path_to_experiment = ''
    point_positions = None
    bodypart_positions = None
    point_positions_extended = None

    def __init__(self, animal, compound, dose, timepoint, experiments_table):
        self.animal = animal  # id number ex. 755
        self.compound = compound  # compound name
        self.dose = dose  # mg/kg
        self.timepoint = timepoint  # hours after injection
        self.experimentsTable = experiments_table
        self.get_experiment_path()
        self.get_point_positions()

    def get_experiment_path(self):
        """
        Search for the path in the fifth column based on the values of the first four columns.
        Returns:
        The value in the fifth column if a match is found, otherwise None.
        """
        result = self.experimentsTable[
            (self.experimentsTable.iloc[:, 0] == self.animal) &
            (self.experimentsTable.iloc[:, 1] == self.compound) &
            (self.experimentsTable.iloc[:, 2] == self.dose) &
            (self.experimentsTable.iloc[:, 3] == self.timepoint)
            ]
        if not result.empty:
            print(f">   Experiment path: {result.iloc[0, 4]}")
            self.path_to_experiment = result.iloc[0, 4]

    def get_point_positions(self):
        """
        Finds the first Excel file in the specified folder and reads a specific sheet.
        Inputs:
        - folder_path: Path to the folder containing the Excel file.
        Outputs:
        - DataFrame containing the data from the specified sheet, or None if no Excel file is found or an error occurs.
        """
        # Use glob to find all .xlsx files in the folder
        excel_file = glob2.glob(os.path.join(self.path_to_experiment, '*.xlsx'))

        if excel_file:
            # If there are any .xlsx files, take the first one
            file_path = excel_file[0]
            print(f"   Found Excel file containg the data points for experiment:")
            print(
                f"   Animal ID {self.animal}, compound {self.compound}, dose {self.dose} mg/kg, timepoint {self.timepoint} h post injection")
            try:
                # Read the specified sheet from the Excel file
                self.point_positions = pd.read_excel(file_path, sheet_name='Coord_centroid')
                self.bodypart_positions = pd.read_excel(file_path, sheet_name='Coord_bodyparts')
                video_min = self.bodypart_positions['Frames_ms'].iloc[-1] / (60 * 1000)  # Video duration in minutes
                nb_frames = self.bodypart_positions['Frames_ms'].shape[0]
                print(f"       Video duration {video_min} min, with {nb_frames} frames")

                # --- When there are more than 1 camera, load the camera id information ---
                sheet_name = 'Coord_bodyparts'
                column_name = 'Ch>3s'
                # Create an ExcelFile object
                xls = pd.ExcelFile(file_path)
                # Check if the target sheet exists
                if sheet_name in xls.sheet_names:
                    # Read the specific sheet
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    # Extract the specific column
                    if column_name in df.columns:
                        self.cam_used = df[column_name]
                    else:
                        print(f"Column '{column_name}' does not exist in the sheet '{sheet_name}'.")
                else:
                    print(f"Sheet '{sheet_name}' does not exist in the Excel file.")
                # ---

                # Note: when there are few visible points there is no centroid calculation possible and values are nan.
                # When this happens in the last frames, there is a mismatch in the dataframes that needs to be corrected
                # as follows :

                # Check the number of rows in both dataframes
                rows_bodypart = self.bodypart_positions.shape[0]
                rows_point = self.point_positions.shape[0]

                # If the number of rows in point_positions is less than bodypart_positions, fill the missing rows
                # with NaN
                if rows_point < rows_bodypart:
                    # Calculate the number of rows to add
                    rows_to_add = rows_bodypart - rows_point

                    # Create a dataframe with NaN values of the same columns as point_positions
                    nan_rows = pd.DataFrame(np.nan, index=range(rows_to_add), columns=self.point_positions.columns)

                    # Append the NaN rows to point_positions
                    self.point_positions_extended = pd.concat([self.point_positions, nan_rows], ignore_index=True)

                    # Verify the number of rows are now the same
                    assert self.point_positions_extended.shape[0] == self.bodypart_positions.shape[
                        0], "The number of rows in point_positions still does not match bodypart_positions"
                    print(f"       Modified point_positions DataFrame with new rows added")
                else:
                    self.point_positions_extended = self.points_positions
                    print(
                        f"       The point_positions DataFrame is already longer or equal in rows compared to "
                        f"bodpart_positions.")
            except Exception as e:
                print(f" X: An error occurred while reading the Excel file: {e}")
        else:
            print("X: No Excel files found in the specified folder.")
