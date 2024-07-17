import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import numpy as np

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
    print(f'        > Make evident NAN values.')
    print(f'        > Apply interpolation to fill small gaps.')

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

    return x_interp, y_interp, {
        'x_percentageNaNs': x_perc_nans,
        'y_percentageNaNs': y_perc_nans}

def slmg_smooth(x,y, window_size, plot=True):
    """
        slmg_smooth - Smooth the x and y data using a median filter.

        Parameters:
            x (array): x coordinates.
            y (array): y coordinates.
            window_size (int): Window size for the median filter.
            plot (bool): Flag to plot the data.

        Returns:
            tuple: Smoothed x and y coordinates.
        """

    def custom_medfilt(data, window_size):
        """
        Apply a median filter to the data while preserving NaN values.

        Parameters:
            data (array): Input data.
            window_size (int): Window size for the median filter.

        Returns:
            array: Smoothed data with NaNs preserved.
        """
        half_window = window_size // 2
        data_smooth = np.copy(data)

        for i in range(len(data)):
            if np.isnan(data[i]):
                continue
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            window_data = data[start:end]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) > 0:
                data_smooth[i] = np.median(window_data)

        return data_smooth

    print(f'3       Smooth the x and y data using a median filter with a windows size of {window_size}')

    # Apply median filter while preserving NaN values and restore NaNs to their original positions after filtering.
    # Change variable names to use the original data arrays (x, y) instead of interpolated ones (x_interp, y_interp).

    x_smooth = custom_medfilt(x, window_size)
    y_smooth = custom_medfilt(y, window_size)

    # Plot the smoothed data if plot flag is True
    if plot:
        raspberry = '#D81159'
        xanthous = '#FFBC42'

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=np.arange(len(x)), y=x, mode='lines', name='X',
                                 line=dict(color=raspberry)))
        fig.add_trace(go.Scatter(x=np.arange(len(x_smooth)), y=x_smooth, mode='lines', name='Smoothed X',
                                 line=dict(color=xanthous)))

        fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y, mode='lines', name='Y',
                                 line=dict(color=raspberry)))
        fig.add_trace(go.Scatter(x=np.arange(len(y_smooth)), y=y_smooth, mode='lines', name='Smoothed Y',
                                 line=dict(color=xanthous)))

        fig.update_layout(title='Smoothed Time Series',
                          xaxis_title='Index', yaxis_title='Value')
        fig.show()

    x_num_nans = np.isnan(x_smooth).sum()
    y_num_nans = np.isnan(y_smooth).sum()
    x_perc_nans = round((x_num_nans / len(x_smooth)) * 100, 2)
    y_perc_nans = round((y_num_nans / len(y_smooth)) * 100, 2)

    return x_smooth, y_smooth, {
        'x_percentageNaNs': x_perc_nans,
        'y_percentageNaNs': y_perc_nans}


def gaussian_smooth(x, y, sigma, plot=True):
    """
    Apply Gaussian smoothing to the x and y coordinates while preserving NaN values.

    Parameters:
        x (array): x coordinates.
        y (array): y coordinates.
        sigma (float): Standard deviation for Gaussian kernel.
        plot (bool): Flag to plot the data.

    Returns:
        tuple: Smoothed x and y coordinates.
    """
    x_smooth = gaussian_filter1d(np.nan_to_num(x, nan=np.nan), sigma=sigma)
    y_smooth = gaussian_filter1d(np.nan_to_num(y, nan=np.nan), sigma=sigma)

    # Restore NaNs to the filtered data
    x_smooth[np.isnan(x)] = np.nan
    y_smooth[np.isnan(y)] = np.nan

    # Ensure the number of NaNs is the same
    assert np.isnan(x_smooth).sum() == np.isnan(y_smooth).sum()


    if plot:
        raspberry = '#D81159'
        xanthous = '#FFBC42'

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(x)), y=x, mode='lines', name='Original X', line=dict(color=raspberry)))
        fig.add_trace(go.Scatter(x=np.arange(len(x_smooth)), y=x_smooth, mode='lines', name='Smoothed X', line=dict(color=xanthous)))
        fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y, mode='lines', name='Original Y', line=dict(color=raspberry)))
        fig.add_trace(go.Scatter(x=np.arange(len(y_smooth)), y=y_smooth, mode='lines', name='Smoothed Y', line=dict(color=xanthous)))
        fig.update_layout(title='Gaussian Smoothing', xaxis_title='Index', yaxis_title='Value')
        fig.show()

    x_num_nans = np.isnan(x_smooth).sum()
    y_num_nans = np.isnan(y_smooth).sum()
    x_perc_nans = round((x_num_nans / len(x_smooth)) * 100, 2)
    y_perc_nans = round((y_num_nans / len(y_smooth)) * 100, 2)

    return x_smooth, y_smooth, {
        'x_percentageNaNs': x_perc_nans,
        'y_percentageNaNs': y_perc_nans}

def slmg_recap_preprocessing(x, y, x_smooth, y_smooth, x_percentageNaNs, y_percentageNaNs, xs_percentageNaNs,
                             ys_percentageNaNs, fps=25):
    """
    slmg_recap_preprocessing - Recap the pre-processing steps, perform checks, analyze noise, and plot the results.

    Parameters:
        x (array): Original x coordinates.
        y (array): Original y coordinates.
        x_smooth (array): Smoothed x coordinates.
        y_smooth (array): Smoothed y coordinates.
        x_percentageNaNs (float): Percentage of NaN values in original x data.
        y_percentageNaNs (float): Percentage of NaN values in original y data.
        xs_percentageNaNs (float): Percentage of NaN values in smoothed x data.
        ys_percentageNaNs (float): Percentage of NaN values in smoothed y data.
        fps (int): Frames per second. Default is 25.

    Returns:
        dict: Summary of the pre-processing results.
    """

    if len(x) != len(x_smooth):
        raise ValueError('The lengths of x and pre-processed x do not match.')

    if x_percentageNaNs != y_percentageNaNs:
        raise ValueError('The % occlusion in x and y do not match.')
    elif xs_percentageNaNs != ys_percentageNaNs:
        raise ValueError('The % occlusion in x pre-processes and y pre-processes do not match.')

    # Level of noise before and after processing the original data
    validIndices = ~np.isnan(x) & ~np.isnan(x_smooth)

    # Calculate Mean Squared Error (MSE), ignoring NaN values
    mse_value = np.mean((x[validIndices] - x_smooth[validIndices]) ** 2)

    # Calculate the standard deviation of the noise, ignoring NaN values
    noise_std_before = np.std(x[validIndices] - np.mean(x[validIndices]))
    noise_std_after = np.std(x_smooth[validIndices] - np.mean(x_smooth[validIndices]))

    # Calculate Signal-to-Noise Ratio (SNR) improvement
    signal_power = np.mean(x[validIndices] ** 2)
    snr_before = signal_power / noise_std_before ** 2
    snr_after = signal_power / noise_std_after ** 2
    snr_improvement = 10 * np.log10(snr_after / snr_before)

    # Convert time to minutes
    num_samples = len(x)
    time_ms = np.arange(num_samples) * (1000 / fps)  # Each sample is (1000/fps) ms apart
    time_minutes = time_ms / 60000

    print('______________________________')
    print('Summary of the pre-processing done:')
    print(f'    Recording duration: {time_minutes[-1]:.2f} minutes at {fps} frames/second')
    print('     Original data:')
    print(f'           Number of NaN values: {np.isnan(x).sum()}')
    print(f'           Percentage of NaN values: {x_percentageNaNs:.2f}%')
    print('     After pre-processing:')
    print(f'           Number of NaN values: {np.isnan(x_smooth).sum()}')
    print(f'           Percentage of NaN values: {xs_percentageNaNs:.2f}%')
    print(f'           Mean Squared Error (MSE) before and after smoothing: {mse_value:.4f}')
    print(f'           Standard deviation of noise before smoothing: {noise_std_before:.4f}')
    print(f'           Standard deviation of noise after smoothing: {noise_std_after:.4f}')
    print(f'           SNR improvement: {snr_improvement:.4f} dB')

    pre_proc_results = {
        'recording_duration': time_minutes[-1],
        'fps': fps,
        'x_percentageNaNs': x_percentageNaNs,
        'ys_percentageNaNs': ys_percentageNaNs,
        'snr_improvement': snr_improvement
    }

    # Plot original vs pre-processed data
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=time_minutes, y=x, mode='lines', name='Original X', line=dict(color='#004E64')))
    fig.add_trace(go.Scatter(x=time_minutes, y=x_smooth, mode='lines', name='Smoothed X', line=dict(color='#FFBC42')))

    fig.add_trace(go.Scatter(x=time_minutes, y=y, mode='lines', name='Original Y', line=dict(color='#004E64')))
    fig.add_trace(go.Scatter(x=time_minutes, y=y_smooth, mode='lines', name='Smoothed Y', line=dict(color='#FFBC42')))

    fig.update_layout(title='Smoothed Time Series',
                      xaxis_title='Time (minutes)', yaxis_title='Value',
                      legend_title='Data Type')
    fig.show()

    return pre_proc_results