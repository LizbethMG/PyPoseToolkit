  # Pose Analysis with Python 

This repository contains a collection of useful functions created to post-analyze pose data, such as data generated from DeepLabCut.
## Table of Contents
- [I. Introduction](#item-one)
- [II. Pre-processing techniques](#item-two)

<a id="item-one"></a>
## I. Introduction
TODO

<a id="item-two"></a>
## II. Pre-processing techniques
### Outlier removal
### Interpolation 
### Smoothing 
Each smoothing technique has its unique characteristics and is suitable for different types of data and noise patterns. 
Here's a brief overview of the differences between Gaussian Smoothing, Savitzky-Golay Filter, Exponential Moving Average (EMA), and Spline Smoothing, along with their typical use cases:
#### 1. Gaussian Smoothing
**Characteristics:**
+ Uses a Gaussian kernel to smooth the data.
+ Effective at reducing high-frequency noise while preserving low-frequency components.
+ The amount of smoothing is controlled by the standard deviation (sigma) of the Gaussian kernel.

**Use Cases:**
+ Suitable for data with high-frequency noise.
+ When you need to preserve the overall trend of the data but want to remove rapid fluctuations.
