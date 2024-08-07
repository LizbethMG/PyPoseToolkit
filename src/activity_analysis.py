import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import scipy.stats as stats


def slmg_computeActivityLevels(data, fps, threshold_method, experiment, plot=True):
    """
        Analyzes periods of high and low activity in the speed data.

        Parameters:
        data (np.ndarray): Numpy array containing instantaneous speed data (pixels per second) with NaN values for occlusion.
        fps (int): Frames per second of the video.
        factor (float): Multiple factor of the standard deviation to set the threshold for high activity.

        Returns:
        dict: Dictionary containing percentages of low, high, and occlusion periods.
        """
    # Type checks
    if not isinstance(data, np.ndarray):
        raise ValueError("Data should be a numpy array")
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Data array should contain numeric values")
    if not isinstance(fps, int):
        raise ValueError("FPS should be an integer")

    print(f">   Calculates  average speed.  ")

    print(f">   Calculates speed variability.  ")

    print(f">   Segment data in low vs. high activity according to the chosen method. ")

    # Calculate the mean ignoring NaN values
    mean = np.nanmean(data)
    # Calculate the standard deviation ignoring NaN values
    std_dev = np.nanstd(data)

    if threshold_method == 1:
        factor = 1
        t_method = 'Mean + STD'
        print(f'        Threshold method: Mean + STD ')
        # Determine the threshold for high activity
        threshold = mean + (std_dev * factor)
    elif threshold_method == 2:  # Percentile-Based Threshold:
        t_method = 'Percentile-Based Threshold'
        print(f'        Theshold method: Percentile-Based Threshold ')
        # Define the percentile threshold (e.g., 75th percentile)
        percentile = 75
        # Calculate the threshold ignoring NaN values
        threshold = np.nanpercentile(data, percentile)
    elif threshold_method == 3:  # Median and Median Absolute Deviation (MAD):
        t_method = 'Median and Median Absolute Deviation (MAD)'
        print(f'        Threshold method: Median and Median Absolute Deviation (MAD) ')
        # Calculate the median ignoring NaN values
        median = np.nanmedian(data)
        # Calculate the Median Absolute Deviation (MAD) ignoring NaN values
        mad = np.nanmedian(np.abs(data - median))
        # Define the factor (e.g., 1.5 times the MAD)
        factor = 1.5
        # Determine the threshold for high activity
        threshold = median + (mad * factor)

    # Classify each period
    high_activity = np.sum(data > threshold)
    low_activity = np.sum((data <= threshold) & (~np.isnan(data)))
    occlusion = np.sum(np.isnan(data))

    total = len(data)

    # Calculate percentages
    high_percentage = round((high_activity / total) * 100, 2)
    low_percentage = round((low_activity / total) * 100, 2)
    occlusion_percentage = round((occlusion / total) * 100, 2)

    # Calculate the Activity Distribution Ratio (ADR) for low, high, and occlusion periods.
    adr_low_high_occlusion = round(low_activity / (high_activity + occlusion), 2)
    adr_low_high = round(low_activity / high_activity,2)

    # Calculate the skewness of low activity distribution over time
    time = np.arange(total) / fps  # Time indices
    low_activity_times = time[data <= threshold]

    if len(low_activity_times) > 0:
        skewness_low = round(stats.skew(low_activity_times), 2)
    else:
        skewness_low = np.nan  # Handle case with no high activity

    # Calculate the temporal skewness of high activity distribution over time
    time = np.arange(total) / fps  # Time indices
    high_activity_times = time[data > threshold]

    if len(high_activity_times) > 0:
        skewness_high = round(stats.skew(high_activity_times), 2)
    else:
        skewness_high = np.nan  # Handle case with no high activity

    # Calculate the entropy of activity distribution across the entire dataset
    # Calculate probabilities
    probabilities = [
         high_activity / total if high_activity > 0 else 0,
         low_activity / total if low_activity > 0 else 0,
         occlusion / total if occlusion > 0 else 0
    ]
    entropy = round(-np.sum(p * np.log(p) for p in probabilities if p > 0), 2)
    # Maximum entropy for three states
    max_entropy = np.log2(3)
    # Normalize entropy
    normalized_entropy = round(entropy / max_entropy, 2) if max_entropy > 0 else 0

    result = {
        'mean': mean,
        'std_dev': std_dev,
        'high_activity': high_percentage,
        'low_activity': low_percentage,
        'occlusion': occlusion_percentage,
        'ADR Low/High+Occ': adr_low_high_occlusion,
        'ADR Low/High': adr_low_high,
        'skewness_low': skewness_low,
        'skewness_high': skewness_high,
        'normalized entropy': normalized_entropy
    }


    # Separate into chunks
    is_high = data > threshold
    is_low = (data <= threshold) & (~np.isnan(data))

    chunks = []
    current_chunk = []
    current_type = None

    for i in range(total):
        if np.isnan(data[i]):
            if current_chunk:
                chunks.append((current_chunk, current_type))
                current_chunk = []
            current_type = 'occlusion'
            chunks.append(([i], current_type))
        elif is_high[i]:
            if current_type != 'high':
                if current_chunk:
                    chunks.append((current_chunk, current_type))
                current_chunk = [i]
                current_type = 'high'
            else:
                current_chunk.append(i)
        elif is_low[i]:
            if current_type != 'low':
                if current_chunk:
                    chunks.append((current_chunk, current_type))
                current_chunk = [i]
                current_type = 'low'
            else:
                current_chunk.append(i)

    if current_chunk:
        chunks.append((current_chunk, current_type))

        # Plot the data
    fig = go.Figure()

    # Add the chunks to the plot
    for chunk, chunk_type in chunks:
        x = time[chunk]
        y = data[chunk]
        if chunk_type == 'high':
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                name='High Activity',
                line=dict(color='#8338EC'),
                showlegend=False,
                hoverinfo='x+y',
                fill='tozeroy'
            ))
        elif chunk_type == 'low':
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                name='Low Activity',
                line=dict(color='#FFBE0B'),
                showlegend=False,
                hoverinfo='x+y',
                fill='tozeroy'
            ))
        elif chunk_type == 'occlusion':
            fig.add_trace(go.Scatter(
                x=x, y=[0] * len(x),
                mode='markers',
                name='Occlusion',
                marker=dict(color='#3A86FF', size=5),
                showlegend=False,
                hoverinfo='x+y'
            ))

    if plot:
        # Add threshold line
        fig.add_trace(go.Scatter(
            x=[time[0], time[-1]], y=[threshold, threshold],
            mode='lines',
            name='Threshold',
            line=dict(color='#FB5607', dash='dash'),
            showlegend=False,
            hoverinfo='none'
        ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Periods of high and low activity - Threshold method used: {t_method}<br>'
                     f'Animal ID {experiment.animal}, compound {experiment.compound}, dose {experiment.dose} mg/kg,'
                     f' timepoint {experiment.timepoint} h post injection ',
                x=0.5,  # Center the title
                xanchor='center'
            ),
            xaxis_title='Time (seconds)',
            yaxis_title='Speed (pixels per second)',
            xaxis=dict(showline=True, linecolor='black', linewidth=1),
            yaxis=dict(showline=True, linecolor='black', linewidth=1),
            showlegend=False,  # Hide the legend
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
            paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper background
        )

        # Set the renderer to browser and explicitly show the plot
        pio.renderers.default = 'browser'
        fig.show()

    print(f'_____________ Results _________________')
    print(f'            High activity %: {high_percentage} ')
    print(f'            Low activity %: {low_percentage} ')
    print(f'            Occlusion activity %: {occlusion_percentage} ')
    print(f'            ADR Low/High+Occ: {adr_low_high_occlusion} and ADR Low/High: {adr_low_high}')
    print(f'            Temporal Skewness of high activity: {skewness_high} and low activity {skewness_low}')
    print(f'            Activity entropy: {normalized_entropy} ')
    print(f'________________________________________')

    return result
