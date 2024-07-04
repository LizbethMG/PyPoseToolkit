import glob2
import os
import pandas as pd

class Experiment: 
    path_to_experiment = ''
    point_positions = None
    def __init__(self, animal, compound, dose, timepoint, experimentsTable):
        self.animal = animal # id number ex. 755
        self.compound = compound # compound name
        self.dose = dose # mg/kg
        self.timepoint = timepoint # hours after injection
        self.experimentsTable = experimentsTable
        self.get_experiment_path()
        self.get_point_positions()

    def get_experiment_path(self):
        """
        Search for the path in the fifth column based on the values of the first four columns.
        Returns:
        The value in the fifth column if a match is found, otherwise None.
        """
        result = self.experimentsTable[
            (self.experimentsTable.iloc[:,0] == self.animal) &
            (self.experimentsTable.iloc[:,1] == self.compound) &
            (self.experimentsTable.iloc[:,2] == self.dose) &
            (self.experimentsTable.iloc[:,3] == self.timepoint)
        ]
        if not result.empty:
            print(f">   Experiment path: {result.iloc[0,4]}")
            self.path_to_experiment = result.iloc[0,4]

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
            print(f">   Found Excel file containg the data points for experiment:")
            print(f">   Animal ID {self.animal}, compound {self.compound}, dose {self.dose} mg/kg, timepoint {self.timepoint} h post injection")
            try:
                # Read the specified sheet from the Excel file
                self.point_positions = pd.read_excel(file_path, sheet_name='Coord_centroid')
            except Exception as e:
                print(f"An error occurred while reading the Excel file: {e}")
                
        else:
            print("No Excel files found in the specified folder.")
           
        