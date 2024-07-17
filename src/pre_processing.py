import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.interpolate import interp1d

def slmg_remove_outliers(x, y, zscore_threshold, plot=True):
    """
    slmg_remove_outliers - Remove outliers from DeepLabCut data using Z-score method,
    interpolate, and apply median filter while handling large gaps.

    Parameters:
        x (array): Original x coordinates.
        y (array): Original y coordinates.
        zscore_threshold (float): Z-score threshold for outlier detection.
        window_size (int): Window size for the median filter.
        gap_threshold (int): Threshold for defining large gaps in the data.

    Returns:
        tuple: Smoothed x and y coordinates.
    """
    # Colors for plots:
    light_blue = '#92DCE5'
    raspberry = '#D81159'
    quinacridone = '#8F2D56'
    midnight_green = '#004E64'
    xanthous = '#FFBC42'

    # 1. Remove outliers from DeepLabCut data using Z-score method.
    print(f"1       Removing outliers with {zscore_threshold:.2f} Z-score threshold for outlier detection:")

    # Calculate the percentage of NaN values relative to the total number of elements from the original data
    x_numNaNs = np.isnan(x).sum()  # Count the number of NaN values
    y_numNaNs = np.isnan(y).sum()
    x_totalElements = len(x)  # Calculate the total number of elements in the vector
    y_totalElements = len(y)
    x_percentageNaNs = round((x_numNaNs / x_totalElements) * 100,2)  # Calculate the percentage of NaN values
    y_percentageNaNs = round((y_numNaNs / y_totalElements) * 100,2)

    # Calculate mean and standard deviation ignoring NaNs
    x_mean = np.nanmean(x)
    x_std = np.nanstd(x)
    y_mean = np.nanmean(y)
    y_std = np.nanstd(y)

    # Calculate Z-scores
    x_zscore = (x - x_mean) / x_std
    y_zscore = (y - y_mean) / y_std

    # Identify outliers based on Z-score threshold
    x_outliers = np.abs(x_zscore) > zscore_threshold
    y_outliers = np.abs(y_zscore) > zscore_threshold
    x_totalOutliers = np.sum(x_outliers)
    y_totalOutliers = np.sum(y_outliers)

    # Replace outliers with NaN in both x and y (because it has to be consistent in the body part coordinates)
    x_clean = np.copy(x)
    y_clean = np.copy(y)
    x_clean[x_outliers] = np.nan
    x_clean[y_outliers] = np.nan
    y_clean[y_outliers] = np.nan
    y_clean[x_outliers] = np.nan

    # Calculate the percentage of NaN values after removing outliers
    x_numNaNs2 = np.isnan(x_clean).sum()
    y_numNaNs2 = np.isnan(y_clean).sum()
    x_percentageNaNs2 = round((x_numNaNs2 / x_totalElements) * 100,2)
    y_percentageNaNs2 = round((y_numNaNs2 / y_totalElements) * 100,2)

    # Plot the original and cleaned data for comparison if plot flag is True
    if plot:
        # Plot the original and cleaned data for comparison

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=np.arange(len(x)), y=x, mode='lines', name='Original X', line=dict(color=midnight_green)))
        fig.add_trace(go.Scatter(x=np.where(x_outliers)[0], y=x[x_outliers], mode='markers', name='Outliers X',
                                 marker=dict(color=raspberry)))
        fig.add_trace(go.Scatter(x=np.arange(len(x_clean)), y=x_clean, mode='lines', name='Cleaned X',
                                 line=dict(color=light_blue)))

        fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y, mode='lines', name='Original Y', line=dict(color=midnight_green)))
        fig.add_trace(go.Scatter(x=np.where(y_outliers)[0], y=y[y_outliers], mode='markers', name='Outliers Y',
                                 marker=dict(color=raspberry)))
        fig.add_trace(go.Scatter(x=np.arange(len(y_clean)), y=y_clean, mode='lines', name='Cleaned Y',
                                 line=dict(color=xanthous)))

        fig.update_layout(title='Original and Cleaned Data with Outliers Marked',
                          xaxis_title='Index', yaxis_title='Value')
        fig.show()

    print(f'        > Original number of NaN in x - {x_numNaNs} representing {x_percentageNaNs}% of the data')
    print(f'        > Original number of NaN in y - {y_numNaNs} representing {x_percentageNaNs}% of the data')
    print(f'        > Numbers of outliers found : X coordinates - {x_totalOutliers} outliers, Y coordinates - {y_totalOutliers} outliers')
    print(f'        > Numbers of NaN after removing outliers : In X {x_numNaNs2} representing {x_percentageNaNs2} % of the data')
    print(f'        > Numbers of NaN after removing outliers : In Y {y_numNaNs2} representing {y_percentageNaNs2} % of the data')

    return x_clean, y_clean, {
        'x_percentageNaNs': x_percentageNaNs,
        'y_percentageNaNs': y_percentageNaNs,
        'x_numOutliers': np.sum(x_outliers),
        'y_numOutliers': np.sum(y_outliers),
    }

    # Placeholder for additional processing (interpolation, median filter, gap handling)
    # interpolate_missing_data()
    # apply_median_filter()
    # handle_large_gaps()

def slmg_interpolate(x, y, threshold_gap, plot=True):
    # Function to mark NaN values and interpolate data while handling large gaps

    def mark_and_interpolate(vector, threshold_gap):
        # Identify gaps
        nan_idx = np.isnan(vector)
        gap_starts = np.where(np.diff(np.concatenate(([0], nan_idx))) == 1)[0]
        gap_ends = np.where(np.diff(np.concatenate((nan_idx, [0]))) == -1)[0]
        gap_lengths = gap_ends - gap_starts + 1

        # Create a copy for interpolation
        vector_interpolated = vector.copy()

        # Interpolate only gaps smaller than the threshold
        for i in range(len(gap_lengths)):
            if gap_lengths[i] < threshold_gap:
                gap_start = gap_starts[i]
                gap_end = gap_ends[i]
                # Use cubic spline interpolation
                interp_func = interp1d(np.where(~nan_idx)[0], vector[~nan_idx], kind='cubic', fill_value='extrapolate')
                vector_interpolated[gap_start:gap_end + 1] = interp_func(np.arange(gap_start, gap_end + 1))

        return vector_interpolated, nan_idx

        # Function to plot the original and interpolated data
    def plot_data(x, y, x_interp, y_interp, x_nan_indices, y_nan_indices):
            # Plot interpolated data
            # Colors for plots:
            light_blue = '#92DCE5'
            raspberry = '#D81159'
            quinacridone = '#8F2D56'
            midnight_green = '#004E64'
            xanthous = '#FFBC42'
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=np.arange(len(x_interp)), y=x_interp, mode='lines', name='Interpolated X',
                                     line=dict(color=quinacridone)))
            fig.add_trace(go.Scatter(x=np.arange(len(x)), y=x, mode='lines', name='Original X', line=dict(color=light_blue)))
            fig.add_trace(go.Scatter(x=np.where(x_nan_indices)[0], y=np.zeros(np.sum(x_nan_indices)), mode='markers',
                                     name='NaN Values X', marker=dict(color=raspberry, size=5)))

            fig.add_trace(go.Scatter(x=np.arange(len(y_interp)), y=y_interp, mode='lines', name='Interpolated Y',
                                     line=dict(color=quinacridone)))
            fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y, mode='lines', name='Original Y', line=dict(color=xanthous)))
            fig.add_trace(go.Scatter(x=np.where(y_nan_indices)[0], y=np.zeros(np.sum(y_nan_indices)), mode='markers',
                                     name='NaN Values Y', marker=dict(color=raspberry, size=5)))

            fig.update_layout(title='Interpolated Time Series',
                              xaxis_title='Index', yaxis_title='Value')
            fig.show()

    print('2       Apply spline interpolation while handling large gaps.')
    print(f'         > Make evident NAN values.')
    print(f'         > Apply interpolation to fill small gaps.')

    # Mark NaNs and interpolate the data
    x_interp, x_nan_indices = mark_and_interpolate(x, threshold_gap)
    y_interp, y_nan_indices = mark_and_interpolate(y, threshold_gap)

    # Count and print the number of NaN values
    # Calculate the percentage of NaN values after removing outliers
    x_num_nans = np.isnan(x_interp).sum()
    y_num_nans = np.isnan(y_interp).sum()
    x_perc_nans = round((x_num_nans / x_interp.size) * 100, 2)
    y_perc_nans = round((y_num_nans / y_interp.size) * 100, 2)

    print(f'        > Numbers of NaN after interpolation: In X {x_num_nans} representing {x_perc_nans} % of the data')
    print(f'        > Numbers of NaN after interpolation : In Y {y_num_nans} representing {y_perc_nans} % of the data')

    if plot:
        # Plot the original and interpolated data
        plot_data(x, y, x_interp, y_interp, x_nan_indices, y_nan_indices)