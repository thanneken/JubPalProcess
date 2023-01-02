# JubPalProcess
Python tools for processing images from multispectral data of cultural heritage artifacts

## Getting Started

This is a public alpha for users already comfortable in Python and YAML. 
Users will need to edit the path to Python in `jubpalprocess.py` and `jubpalfunctions.py` and install all dependencies. 
Subsequently, basic changes will be made in `jubpaloptions.yaml` following the exemples provided. 
Selections from among those options are made in an interactive command-line interface upon running `jubpalprocess.py`.

## Features 

The processing is designed to be run on one image set (page) at a time. 
Multiple processes can be run simultaneously, with the result that command-line output mixes multiple threads. 

The following processes are throughly tested:

1. Blur and Divide, following the method used by Roger Easton except that a Gaussian blur is used. The sigma (radius) of the blur can be selected from multiple options. More than one can be selected, with the result that downstream methods will run for each sigma selected.
1. Principal Component Analysis (PCA) requires at least one Region of Interest (ROI) to be specified in `jubpaloptions.yaml` and one to be selected in the user interface.
1. Minimum Noise Fraction (MNF) requires a ROI as with PCA, as well as a region that defines “noise.” 
1. Fast Independent Component Analysis (FICA) requires significant processing time and produces good results. Unlike PCA and MNF, the most helpful component can be anywhere so the number of components should always be set to maximum.

The following processes are not presently functioning:

1. Canonical Correlation Analysis (CCA) is an active interest but not very far along in testing.
1. Kernel Principal Component Analysis (KPCA) does not seem to produce results that justify the substantial processing time required. 
1. Spectral Embedding (specembed) does not seem to produce results that justify the substantial processsing time required. 

The processes above produce results in 32-bit floating point. 
Some adjustment is required and helpful to visualize that data to a human eye through a computer screen. 
One or more options can be selected. 

1. Equalize most often produces good results without further processing. 
1. Adaptive requires more processing time and sometimes produces good results.
1. Rescale is a minimal process to fit the floating point data on a scale from 0 to 1, and is required for conversion to integer formats for PNG and JPG.
1. None leaves the data as floating-point, can only be saved as TIFF, and defers the necessary scaling.

The resulting data can be saved in one or more file formats. 

1. JPG has the best compression and should be used for previewing images for basic quality. Each pixel is downscaled to eight bits, which limits further histogram adjustment. 
1. TIFF saves all thirty-two bits per pixel with no compression. Once the best images are identified from JPG preview, TIFF files should be used for additional processing.
1. PNG can be thought of as half-way between JPG and TIFF. Each pixel as saved with sixteen bits and the compression is lossless. 

Other options:

1. Number of components should always be set to maximum when using FICA. When using PCA and MNF it is safe to assume that the most helpful component will be in the first five, ten at most.
1. Skip files with UVB or UVP in the name is necessary unless the registration error caused by the thickness of the filter has been corrected in software. 
1. FICA accepts options for the maximum number of iterations and tolerance. Adjusting these may impact processing time and address a warning during fitting, “FastICA did not converge.”
1. An option remains to save the output as a stack/cube. This only works with TIFF and presumes good resources for transfering and reading large amounts of data.
1. A cache can be used to preserve data from Blur and Divide between runs. 

Presently only one Region of Interest (ROI) can be run at a time. 

If most of this makes sense to you and we don’t already know each other, please be in touch. It’s a small community.

