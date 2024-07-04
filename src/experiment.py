class Experiment: 
    def __init__(self, animal, compound, dose, timepoint, dataTable):
        self.animal = animal # id number ex. 755
        self.compound = compound # compound name
        self.dose = dose # mg/kg
        self.timepoint = timepoint # hours after injection
        self.dataTable = dataTable

    def get_experiment_path(self):
        """
        Search for the path in the fifth column based on the values of the first four columns.
        Returns:
        The value in the fifth column if a match is found, otherwise None.
        """
        result = self.dataTable[
            (self.dataTable.iloc[:,0] == self.animal) &
            (self.dataTable.iloc[:,1] == self.compound) &
            (self.dataTable.iloc[:,2] == self.dose) &
            (self.dataTable.iloc[:,3] == self.timepoint)
        ]
        if not result.empty:
            return result.iloc[0,4]
        else: 
            return None


    # Debug print to confirm method existence
    print("Experiment class loaded with get_experiment_path method.")