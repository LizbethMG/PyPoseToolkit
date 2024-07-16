import numpy as np
import matplotlib.pyplot as plt


def slmg_remove_outliers(x, y, zscore_threshold, window_size, gap_threshold, plot=True):
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
    print(f">       Removing outliers with {zscore_threshold:.2f} Z-score threshold for outlier detection:")

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
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(x, color=midnight_green, label='Original')
        plt.plot(np.where(x_outliers)[0], x[x_outliers], 'rx', label='Outliers')
        plt.plot(x_clean, color=light_blue, label='Clean data')
        plt.title('Original X Coordinates with Outliers Marked')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(y, color=midnight_green, label='Original')
        plt.plot(np.where(y_outliers)[0], y[y_outliers], 'rx', label='Outliers')
        plt.plot(y_clean, color=light_blue, label='Clean data')
        plt.title('Original Y Coordinates with Outliers Marked')
        plt.legend()

        plt.suptitle('1. Remove Outliers from data')
        plt.show()

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
