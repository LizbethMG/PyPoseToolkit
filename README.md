# Pose Analysis with Python 

This repository contains a collection of useful functions created to post-analyze pose data, such as data generated from DeepLabCut.

## Table of Contents

- [I. Introduction](#introduction)
- [II. Pre-processing Techniques](#pre-processing-techniques)
- [III. Metrics for Activity Analysis](#metrics-for-activity-analysis)

<a id="introduction"></a>
## I. Introduction
TODO

<a id="pre-processing-techniques"></a>
## II. Pre-processing Techniques

### Outlier Removal

### Interpolation

### Smoothing 

Each smoothing technique has its unique characteristics and is suitable for different types of data and noise patterns. Here's a brief overview of the differences between Gaussian Smoothing, Savitzky-Golay Filter, Exponential Moving Average (EMA), and Spline Smoothing, along with their typical use cases:

#### 1. Gaussian Smoothing

**Characteristics:**

- Uses a Gaussian kernel to smooth the data.
- Effective at reducing high-frequency noise while preserving low-frequency components.
- The amount of smoothing is controlled by the standard deviation (sigma) of the Gaussian kernel.

**Use Cases:**

- Suitable for data with high-frequency noise.
- When you need to preserve the overall trend of the data but want to remove rapid fluctuations.

<a id="metrics-for-activity-analysis"></a>
## III. Metrics for Activity Analysis

### 1. Average Speed

Calculate the average speed over the duration of each recording. This provides a general sense of how quickly the animal is moving on average.

### 2. Speed Variability

Assess the variability in speed, using measures such as standard deviation, to understand how consistent or variable the animal's movement is.

### 3. Segmentation: Low vs. High Activity

Measure the percentage of total time the animal is in active motion versus being stationary. 

```python
result = slmg_computeActivityLevels(data, fps, threshold_method, experiment_info, plot=True)
```
**Threshold Methods:**

1. **Mean and Standard Deviation Method:**
   - Best for normally distributed data with minimal outliers.
2. **Percentile-Based Threshold:**
   - Ideal for setting intuitive and robust thresholds, especially in the presence of outliers.
3. **Median and Median Absolute Deviation (MAD) Method:**
   - Suitable for data with significant outliers or non-normal distributions.

### 4. Temporal Resolution

```python
result = slmg_computeActivityLevels(data, fps, threshold_method, current_experiment, plot=True)
```
Captures the temporal distribution of activity levels to assess the distribution and concentration of occlusion, high, and low activity periods throughout the recording. Additionally, the following metrics complement the temporal distribution activity:

#### 4.1 Activity Distribution Ratio (ADR)

The Activity Distribution Ratio (ADR) provides insights into the relative proportions of time spent in low activity compared to high activity and occlusion periods. The ADR helps to understand the overall balance of activity states, making it useful for comparing activity across different conditions or sessions to identify shifts in activity patterns.

- **ADR Low/High/Occlusion**:
  - **Greater than 1 (>1)**: Indicates more time spent in low activity compared to the combined time of high activity and occlusion. This suggests a tendency towards inactivity.
  - **Equal to 1 (=1)**: Indicates equal time spent in low activity and the combined time of high activity and occlusion.
  - **Less than 1 (<1)**: Indicates more time spent in high activity and occlusion compared to low activity, suggesting more dynamic behavior.

- **ADR Low/High**:
  - **Greater than 1 (>1)**: Indicates more time spent in low activity compared to high activity. This suggests inactivity dominates.
  - **Equal to 1 (=1)**: Indicates equal time spent in low and high activity.
  - **Less than 1 (<1)**: Indicates more time spent in high activity compared to low activity, suggesting a predominance of active behavior.

#### 4.2 Temporal Skewness

Measures whether high and low activity are more common at certain times in the recording. It computes the skewness for high and low profiles. Analyzing skewness helps identify patterns in animal behavior, such as initial bursts of activity or increased activity towards the session's end, which may be influenced by environmental or experimental conditions.

- **Positive Skewness**:
  - Indicates the activity (high or low) is more concentrated towards the end of the recording.
  - The distribution has a longer tail on the right side.

- **Negative Skewness**:
  - Indicates that the activity is more concentrated towards the beginning of the recording.
  - The distribution has a longer tail on the left side.

- **Skewness Near Zero**:
  - Suggests that the activity is symmetrically distributed around the middle of the recording.
  - Indicates a balanced or uniform activity distribution.

#### 4.3 Activity Entropy

It measures the normalized entropy of the activity distribution to assess how uniformly or randomly activity is spread across the recording. 

- **0**: Indicates complete dominance by a single state.
- **1**: Indicates an equal distribution across all states, meaning that the animal does not predominantly remain in one state.
- **Values between 0 and 1**: Reflect varying degrees of distribution, with values closer to 1 indicating more uniformity. A low entropy value indicates that the activity is dominated by one or a few states. For example, if most of the recording consists of high activity, with little time in low activity or occlusion, the entropy would be low. Low entropy implies predictability, as the animal spends most of its time in a particular state. A lower entropy value indicates that one or more states (high, low, or occlusion) dominate the recording.

### Acceleration Events (TODO)

Analyze the frequency and magnitude of acceleration and deceleration events. Frequent or large accelerations may indicate bursts of activity.

### Total Distance Traveled (TODO)




     
