import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.signal import medfilt
import plotly.io as pio

pio.renderers.default = 'iframe'
import warnings
import numpy as np

""" ---------- Pre-processing functions for handling separate datasets -------------------
 These functions are useful for scenarios where you have two different sets of data,
 such as the x and y pixel coordinates of tracked points  The functions aim to
 clean, smooth, and analyze these coordinates to facilitate further processing
 and visualization."""


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
    print(f"*       Removing outliers with {zscore_threshold:.2f} Z-score threshold for outlier detection:")

    # Calculate the percentage of NaN values relative to the total number of elements from the original data
    x_numNaNs = np.isnan(x).sum()  # Count the number of NaN values
    y_numNaNs = np.isnan(y).sum()
    x_totalElements = len(x)  # Calculate the total number of elements in the vector
    y_totalElements = len(y)
    x_percentageNaNs = round((x_numNaNs / x_totalElements) * 100, 2)  # Calculate the percentage of NaN values
    y_percentageNaNs = round((y_numNaNs / y_totalElements) * 100, 2)

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
    x_percentageNaNs2 = round((x_numNaNs2 / x_totalElements) * 100, 2)
    y_percentageNaNs2 = round((y_numNaNs2 / y_totalElements) * 100, 2)

    # Plot the original and cleaned data for comparison if plot flag is True
    if plot:
        # Plot the original and cleaned data for comparison

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=np.arange(len(x)), y=x, mode='lines', name='Original X', line=dict(color=midnight_green)))
        fig.add_trace(go.Scatter(x=np.where(x_outliers)[0], y=x[x_outliers], mode='markers', name='Outliers X',
                                 marker=dict(color=raspberry)))
        fig.add_trace(go.Scatter(x=np.arange(len(x_clean)), y=x_clean, mode='lines', name='Cleaned X',
                                 line=dict(color=light_blue)))

        fig.add_trace(
            go.Scatter(x=np.arange(len(y)), y=y, mode='lines', name='Original Y', line=dict(color=midnight_green)))
        fig.add_trace(go.Scatter(x=np.where(y_outliers)[0], y=y[y_outliers], mode='markers', name='Outliers Y',
                                 marker=dict(color=raspberry)))
        fig.add_trace(go.Scatter(x=np.arange(len(y_clean)), y=y_clean, mode='lines', name='Cleaned Y',
                                 line=dict(color=xanthous)))

        fig.update_layout(title='Original and Cleaned Data with Outliers Marked',
                          xaxis_title='Index', yaxis_title='Value')
        fig.show()

    print(f'        > Original number of NaN in x - {x_numNaNs} representing {x_percentageNaNs}% of the data')
    print(f'        > Original number of NaN in y - {y_numNaNs} representing {x_percentageNaNs}% of the data')
    print(
        f'        > Numbers of outliers found : X coordinates - {x_totalOutliers} outliers, Y coordinates - {y_totalOutliers} outliers')
    print(
        f'        > Numbers of NaN after removing outliers : In X {x_numNaNs2} representing {x_percentageNaNs2} % of the data')
    print(
        f'        > Numbers of NaN after removing outliers : In Y {y_numNaNs2} representing {y_percentageNaNs2} % of the data')

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
        fig.add_trace(
            go.Scatter(x=np.arange(len(x)), y=x, mode='lines', name='Original X', line=dict(color=light_blue)))
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

    print('*       Apply spline interpolation while handling large gaps.')
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


def slmg_median_smooth(x, y, window_size, plot=True):
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

    print(f'*       Smooth the x and y data using a median filter with a windows size of {window_size}')

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


def slmg_gaussian_smooth(x, y, sigma, plot=True):
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
    print(f'*       Smooth the x and y data using a gaussian filter with a sigma value  of {sigma}')
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
        fig.add_trace(go.Scatter(x=np.arange(len(x_smooth)), y=x_smooth, mode='lines', name='Smoothed X',
                                 line=dict(color=xanthous)))
        fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y, mode='lines', name='Original Y', line=dict(color=raspberry)))
        fig.add_trace(go.Scatter(x=np.arange(len(y_smooth)), y=y_smooth, mode='lines', name='Smoothed Y',
                                 line=dict(color=xanthous)))
        fig.update_layout(title='Gaussian Smoothing', xaxis_title='Index', yaxis_title='Value')
        fig.show()

    x_num_nans = np.isnan(x_smooth).sum()
    y_num_nans = np.isnan(y_smooth).sum()
    x_perc_nans = round((x_num_nans / len(x_smooth)) * 100, 2)
    y_perc_nans = round((y_num_nans / len(y_smooth)) * 100, 2)

    return x_smooth, y_smooth, {
        'x_percentageNaNs': x_perc_nans,
        'y_percentageNaNs': y_perc_nans}


def slmg_spline_smooth(x, y, s, plot=True):
    """
        Apply spline smoothing to the x and y coordinates while preserving NaN values.

        Parameters:
            x (array): x coordinates.
            y (array): y coordinates.
            s (float): Smoothing factor.
            plot (bool): Flag to plot the data.

        Returns:
            tuple: Smoothed x and y coordinates along with percentage of NaNs in x and y.
        """
    print(f'*       Smooth the x and y data using spline curves with a smoothing factor s of  {s}')

    # Identify valid (non-NaN) indices
    valid_indices = ~(np.isnan(x) | np.isnan(y))

    # Ensure we have valid data points
    if valid_indices.sum() == 0:
        raise ValueError("All values are NaN. Cannot perform smoothing.")

    # Create the spline for valid data
    x_spline = UnivariateSpline(np.arange(len(x))[valid_indices], x[valid_indices], s=s)
    y_spline = UnivariateSpline(np.arange(len(y))[valid_indices], y[valid_indices], s=s)

    # Create copies to hold the smoothed values, preserving NaNs
    x_smooth = np.copy(x)
    y_smooth = np.copy(y)

    # Apply spline only to the valid points
    x_smooth[valid_indices] = x_spline(np.arange(len(x))[valid_indices])
    y_smooth[valid_indices] = y_spline(np.arange(len(y))[valid_indices])

    # Ensure the number of NaNs is the same
    assert np.isnan(x).sum() == np.isnan(x_smooth).sum()
    assert np.isnan(y).sum() == np.isnan(y_smooth).sum()

    if plot:
        raspberry = '#D81159'
        xanthous = '#FFBC42'

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(x)), y=x, mode='lines', name='Original X', line=dict(color=raspberry)))
        fig.add_trace(go.Scatter(x=np.arange(len(x_smooth)), y=x_smooth, mode='lines', name='Smoothed X',
                                 line=dict(color=xanthous)))
        fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y, mode='lines', name='Original Y', line=dict(color=raspberry)))
        fig.add_trace(go.Scatter(x=np.arange(len(y_smooth)), y=y_smooth, mode='lines', name='Smoothed Y',
                                 line=dict(color=xanthous)))
        fig.update_layout(title='Spline Smoothing', xaxis_title='Index', yaxis_title='Value')
        fig.show()

    x_num_nans = np.isnan(x_smooth).sum()
    y_num_nans = np.isnan(y_smooth).sum()
    x_perc_nans = round((x_num_nans / len(x_smooth)) * 100, 2)
    y_perc_nans = round((y_num_nans / len(y_smooth)) * 100, 2)

    return x_smooth, y_smooth, {
        'x_percentageNaNs': x_perc_nans,
        'y_percentageNaNs': y_perc_nans}


def slmg_recap_preprocessing(x, y, x_smooth, y_smooth, x_percentageNaNs, y_percentageNaNs, xs_percentageNaNs,
                             ys_percentageNaNs, fps=25, plot=True, summary=False):
    """
    slmg_recap_preprocessing - Recap the pre-processing steps, perform checks, analyze noise, and plot the results.
    When x and Y have been modified (before speed calculation)

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

    if summary == True:
        print('>   ______________________________')
        print('>   Summary of the pre-processing done:')
        print(f'        Recording duration: {time_minutes[-1]:.2f} minutes at {fps} frames/second')
        print('         Original data:')
        print(f'           Number of NaN values: {np.isnan(x).sum()}')
        print(f'           Percentage of NaN values: {x_percentageNaNs:.2f}%')
        print('         After pre-processing:')
        print(f'           Number of NaN values: {np.isnan(x_smooth).sum()}')
        print(f'           Percentage of NaN values: {xs_percentageNaNs:.2f}%')
        print(f'           Mean Squared Error (MSE) before and after smoothing: {mse_value:.4f}')
        print(f'           Standard deviation of noise before smoothing: {noise_std_before:.4f}')
        print(f'           Standard deviation of noise after smoothing: {noise_std_after:.4f}')
        print(f'           SNR improvement: {snr_improvement:.4f} dB')
        print('>   ______________________________')

    pre_proc_results = {
        'recording_duration': time_minutes[-1],
        'fps': fps,
        'x_percentageNaNs': x_percentageNaNs,
        'ys_percentageNaNs': ys_percentageNaNs,
        'snr_improvement': snr_improvement
    }

    if plot != True:
        pass
    # Plot original vs pre-processed data
    else:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=time_minutes, y=x, mode='lines', name='Original X', line=dict(color='#004E64')))
        fig.add_trace(
            go.Scatter(x=time_minutes, y=x_smooth, mode='lines', name='Smoothed X', line=dict(color='#FFBC42')))

        fig.add_trace(go.Scatter(x=time_minutes, y=y, mode='lines', name='Original Y', line=dict(color='#004E64')))
        fig.add_trace(
            go.Scatter(x=time_minutes, y=y_smooth, mode='lines', name='Smoothed Y', line=dict(color='#FFBC42')))

        fig.update_layout(title='Smoothed Time Series',
                          xaxis_title='Time (minutes)', yaxis_title='Value',
                          legend_title='Data Type')
        fig.show()

    return pre_proc_results


def slmg_inst_speed(x, y, fps, cam_used, plot=True):
    """
        Computes the instantaneous speed and handles camera switches (if two camaras used)

        Parameters:
        x (np.ndarray): The x-coordinate data.
        y (np.ndarray): The y-coordinate data.
        cam_switch_exists: True when there are camera switches (more than one camera used)
                            False when only one camera was used
        Sry : The camera switch data (a vector containing the camera id used)

        Returns:
         NanRate
         MeanSpeed
        """
    # Initialize variable
    nb_total_frame = x.shape[0]
    spd = np.zeros((nb_total_frame, 4))
    spd[0, :] = np.nan
    tilt = []
    # Calculate the time per frame in milliseconds
    time_per_frame_s = 1 / fps

    # Compute instant speed
    for frame in range(1, nb_total_frame):
        spd[frame, 0] = frame * time_per_frame_s  # time in seconds
        dx = x[frame] - x[frame - 1]  # x2 - x1
        dy = y[frame] - y[frame - 1]  # y2 - y1
        dt = (x[frame] - x[frame - 1]) / 1000  # dt in seconds
        d = np.sqrt(dx ** 2 + dy ** 2)  # Euclidean distance
        spd[frame, 1] = d / time_per_frame_s  # Instantaneous velocity in pixels per second

        if cam_used is not None:  # cam switch exists
            if cam_used[frame] != cam_used[frame - 1]:  # if there is a camera switch
                tilt.append(frame * time_per_frame_s)  # add timestamp of the frame to tilt in seconds

    # Extract data for plotting
    x_wcam = spd[:, 0]
    y_wcam = spd[:, 1]

    # Identify NaN values
    nan_indices = np.isnan(y_wcam)

    if plot:
        # Create plotly figure
        fig = go.Figure()

        # Add speed trace
        fig.add_trace(
            go.Scatter(x=x_wcam, y=y_wcam, mode='lines', name='Instant Speed', line=dict(color='black', width=1)))
        # Add small gray dots for NaN values at y=0
        fig.add_trace(go.Scatter(x=x_wcam[nan_indices], y=np.zeros(np.sum(nan_indices)), mode='markers',
                                 name='NaN Values', marker=dict(color='gold', size=3)))

        if cam_used is not None:  # cam switch exists
            # Add vertical lines for camera switches
            for switch_time in tilt:
                fig.add_vline(x=switch_time, line=dict(color='darksalmon'), annotation_text='Camera Switch')
        # Customize layout
        fig.update_layout(
            title='Instant speed - raw data with camera switches and nan values',
            xaxis_title='Time (s)',
            yaxis_title='Pixels/s',
            showlegend=True
        )

        # Show plot
        fig.show()

    # Print NaN Rate and Mean Speed if required
    nanRate = round((np.isnan(y_wcam).sum() / y_wcam.size) * 100, 2)
    meanSpeed = round(np.nanmean(y_wcam), 2)

    print(f"           NaN Rate: {nanRate} %")
    print(f"           Mean Speed: {meanSpeed} Pixels/s")

    return x_wcam, y_wcam, meanSpeed, nanRate


""" ---------- Pre-processing functions to handle a single dataset -------------------
These functions are designed for cases where the x-coordinate represents the time vector
 and the y-coordinate represents the data values (e.g., speed). 
"""


def slmg_savgol_smooth(x, y, window_length=51, polyorder=3, plot=True):
    """
        Smoothing  Y using Savitzky-Golay filter
        WARNING: can give negative values
        Parameters:
            x (array): x coordinates.
            y (array): y coordinates.
            window_size (int): Window size for the median filter.
            plot (bool): Flag to plot the data.

        Returns:
            tuple: Smoothed x and y coordinates.
        """
    y_smooth = savgol_filter(y, window_length, polyorder)

    if plot:
        raspberry = '#D81159'
        xanthous = '#FFBC42'

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Original Y', line=dict(color=raspberry)))
        fig.add_trace(go.Scatter(x=x, y=y_smooth, mode='lines', name='Smoothed Y',
                                 line=dict(color=xanthous)))
        fig.update_layout(title='Savitzky-Golay filter', xaxis_title='Index', yaxis_title='Value')
        fig.show()

    y_num_nans = np.isnan(y_smooth).sum()
    y_perc_nans = round((y_num_nans / len(y_smooth)) * 100, 2)

    return y_smooth, y_perc_nans


def slmg_median_smooth_2(x, y, window_size, plot=True):
    """
        slmg_smooth - Smooth the  y data using a median filter.

        Parameters:
            x (array): time vector
            y (array): y coordinates.
            window_size (int): Window size for the median filter.
            plot (bool): Flag to plot the data.

        Returns:
            tuple: Smoothed y coordinates.
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

    print(f'*       Smooth the data using a median filter with a windows size of {window_size}')

    # Apply median filter while preserving NaN values and restore NaNs to their original positions after filtering.
    # Change variable names to use the original data arrays (x, y) instead of interpolated ones (x_interp, y_interp).

    y_smooth = custom_medfilt(y, window_size)

    # Plot the smoothed data if plot flag is True
    if plot:
        raspberry = '#D81159'
        xanthous = '#FFBC42'

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Y',
                                 line=dict(color=raspberry)))
        fig.add_trace(go.Scatter(x=x, y=y_smooth, mode='lines', name='Smoothed Y',
                                 line=dict(color=xanthous)))
        fig.update_layout(title='Smoothed Time Series',
                          xaxis_title='Index', yaxis_title='Value')
        fig.show()

    y_num_nans = np.isnan(y_smooth).sum()
    y_perc_nans = round((y_num_nans / len(y_smooth)) * 100, 2)

    return y_smooth, y_perc_nans


def slmg_recap_preprocessing_2(y, y_smooth, y_percentageNaNs, ys_percentageNaNs, fps=25):
    """
    slmg_recap_preprocessing - Recap the pre-processing steps, perform checks, analyze noise, and plot the results.
    When x and Y have been modified (before speed calculation)

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

    # Level of noise before and after processing the original data
    validIndices = ~np.isnan(y) & ~np.isnan(y_smooth)

    # Calculate Mean Squared Error (MSE), ignoring NaN values
    mse_value = np.mean((y[validIndices] - y_smooth[validIndices]) ** 2)

    # Calculate the standard deviation of the noise, ignoring NaN values
    noise_std_before = np.std(y[validIndices] - np.mean(y[validIndices]))
    noise_std_after = np.std(y_smooth[validIndices] - np.mean(y_smooth[validIndices]))

    # Calculate Signal-to-Noise Ratio (SNR) improvement
    signal_power = np.mean(y[validIndices] ** 2)
    snr_before = signal_power / noise_std_before ** 2
    snr_after = signal_power / noise_std_after ** 2
    snr_improvement = 10 * np.log10(snr_after / snr_before)

    # Convert time to minutes
    num_samples = len(y)
    time_ms = np.arange(num_samples) * (1000 / fps)  # Each sample is (1000/fps) ms apart
    time_minutes = time_ms / 60000

    print('>   Summary of the pre-processing done:')
    print(f'        Recording duration: {time_minutes[-1]:.2f} minutes at {fps} frames/second')
    print('         Original data:')
    print(f'           Number of NaN values: {np.isnan(y).sum()}')
    print(f'           Percentage of NaN values: {y_percentageNaNs:.2f}%')
    print('         After pre-processing:')
    print(f'           Number of NaN values: {np.isnan(y_smooth).sum()}')
    print(f'           Percentage of NaN values: {ys_percentageNaNs:.2f}%')
    print(f'           Mean Squared Error (MSE) before and after smoothing: {mse_value:.4f}')
    print(f'           Standard deviation of noise before smoothing: {noise_std_before:.4f}')
    print(f'           Standard deviation of noise after smoothing: {noise_std_after:.4f}')
    print(f'           SNR improvement: {snr_improvement:.4f} dB')

    pre_proc_results = {
        'recording_duration': time_minutes[-1],
        'fps': fps,
        'y_percentageNaNs': ys_percentageNaNs,
        'snr_improvement': snr_improvement
    }

    return pre_proc_results


def slmg_recalibrate_data(data, fps, sync_time):
    """
       Trims the data to start at the specified sync_time.

       Parameters:
       - data: np.ndarray with shape (n_samples,), representing the signal values.
       - sync_time: The synchronization time in seconds.
       - fps: Frames per second of the data.

       Returns:
       - np.ndarray with data starting from the sync_time.
       """
    # Convert sync_time to frame number
    print(f">   Recalibrates the timestamps in the data to start at the specified sync_time: ")
    print(f'        The synchronization time in seconds: {sync_time}')

    # Convert sync_time to frame number
    sync_frame = int(sync_time * fps)

    # Trim the data to start from the sync_frame
    recalibrated_data = data[sync_frame:]

    # Convert time to minutes
    num_samples = len(data)
    time_ms = np.arange(num_samples) * (1000 / fps)  # Each sample is (1000/fps) ms apart
    time_minutes = time_ms / 60000

    num_samples2 = len(recalibrated_data)
    time_ms2 = np.arange(num_samples2) * (1000 / fps)  # Each sample is (1000/fps) ms apart
    time_minutes2 = time_ms2 / 60000

    print(f'        Original recording duration: {time_minutes[-1]:.2f} minutes')
    print(f'        Original recording duration: {time_minutes2[-1]:.2f} minutes')

    return recalibrated_data


def slmg_window_data(data, start_time=None, end_time=None, duration=None, fps=25):
    """
        Trims the data based on start and end time limits or duration from the start.

        Parameters:
        - data: np.ndarray with shape (n_samples,), representing the signal values.
        - start_time: Optional. Time in seconds to start the window.
        - end_time: Optional. Time in seconds to end the window.
        - duration: Optional. Duration in seconds for the window starting from start_time or beginning of data.
        - fps: Frames per second of the data. Default is 1.

        Returns:
        - np.ndarray containing the windowed data.
        """
    # Convert sync_time to frame number
    print(f">   Trims the data based on start and end time limits OR duration from the start. ")

    n_samples = len(data)
    max_time = n_samples / fps  # Maximum time based on the length of data and fps

    if duration is not None:
        if start_time is not None or end_time is not None:
            raise ValueError("Provide either start_time and end_time or duration, not both.")
        start_time = 0 if start_time is None else start_time
        end_time = start_time + duration

    if start_time is None or end_time is None:
        raise ValueError("Either start_time and end_time or start_time and duration must be provided.")

    # Ensure the times are within the length of data
    print(f'        Desired start time in s: {start_time} and end time in s: {end_time}')
    start_time = max(0, min(start_time, max_time))
    end_time = max(0, min(end_time, max_time))

    if end_time > max_time:
        warnings.warn("The specified window exceeds the length of the data. The window will be truncated.")

    # Convert start_time and end_time to frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Ensure end_frame is within the length of data
    end_frame = min(n_samples - 1, end_frame)

    # Window the data based on frame indices
    windowed_data = data[start_frame:end_frame + 1]

    num_samples = len(windowed_data)
    time_ms = np.arange(num_samples) * (1000 / fps)  # Each sample is (1000/fps) ms apart
    time_minutes = time_ms / 60000

    print(f'        New recording duration: {time_minutes[-1]:.2f} minutes')

    return windowed_data
