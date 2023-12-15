# JubPalProcess
Python tools for processing images from multispectral data of cultural heritage artifacts

## Getting Started

This is a public alpha for users already comfortable in Python and YAML. 
Users will need to edit the path to Python in `process.py` and install all dependencies. 
Subsequently, basic changes will be made in `options.yaml` following the examples provided. 
Additional metadata is specified on a per-project basis following the model [Ambrosiana\_F130sup.yaml](https://palimpsest.stmarytx.edu/Ambrosiana2023/Ambrosiana_F130sup/Ambrosiana_F130sup.yaml).
Colorcheker calibrated color processing requires additional metadata following the model [Calibration.yaml](https://palimpsest.stmarytx.edu/Ambrosiana2023/Calibration/Calibration.yaml).
(Yaml files do not allow tabs for indentation, only spaces.)
Selections from among the options specified in yaml files are made in an interactive command-line interface upon running `process.py`.
A non-interactive mode assuming coded defaults can be started by adding the argument “noninteractive” to the command line. 
As of 2023, detailed documentation and a community support forum remain long-term desiderata. 
Reaching out to [Todd Hanneken](mailto:thanneken@stmarytx.edu) is welcome for anyone ready and willing to learn to process MSI data in this way. 
We are a small and welcoming community. 

## Features 

The processing is designed to be run on one image set (page) at a time. 
It can run in non-interactive mode on large batches of data if required metadata is specified in yaml files.

The following processes are thoroughly tested:

1. Colorchecker calibrated color processing produces very accurate color (0.7 ΔE). 
It requires that a color checker with known XYZ patch values has been imaged with the same capture configuration.
1. Blur and Divide, following the method used by Roger Easton except that a Gaussian blur is used. 
The sigma (radius) of the blur can be selected from multiple options. 
More than one can be selected, with the result that downstream methods will run for each sigma selected.
1. Principal Component Analysis (PCA) requires at least one Region of Interest (ROI) to be specified in `options.yaml` and one to be selected in the user interface.
1. Minimum Noise Fraction (MNF) requires a ROI as with PCA, as well as a region that defines “noise.” 
1. Fast Independent Component Analysis (FICA) produces dramatic results, often dramatically good results. 
Unlike PCA and MNF, the most helpful component can be anywhere so the number of components should always be set to maximum.

The following processes are not presently functioning:

1. Canonical Correlation Analysis (CCA) is an active interest but not very far along in testing.
1. Kernel Principal Component Analysis (KPCA) does not seem to produce results that justify the substantial processing time required. 
1. Spectral Embedding (specembed) does not seem to produce results that justify the substantial processing time required. 

The linear transformation processes above (ICA, PCA, MNF) produce results in 32-bit floating point. 
Some adjustment is required and helpful to visualize that data to a human eye through a computer screen. 
One or more histogram adjustment options can be selected. 

1. Equalize most often produces good results without further processing. 
1. Adaptive requires more processing time and sometimes produces good results.
1. Rescale is a minimal process to fit the floating point data on a scale from 0 to 1, and is required for conversion to integer formats for PNG and JPG.
1. None leaves the data as floating-point, can only be saved as TIFF, and defers the necessary scaling.

The resulting data can be saved in one or more file formats. 

1. JPG has the best compression and should be used for previewing images for basic quality. 
Each pixel is downscaled to eight bits, which limits further histogram adjustment. 
1. TIFF saves all thirty-two bits per pixel with no compression. 
Once the best images are identified from JPG preview, TIFF files should be used for additional processing.
1. PNG can be thought of as half-way between JPG and TIFF. 
Each pixel is saved with sixteen bits and the compression is lossless. 

Other options:

1. Number of components should always be set to maximum when using FICA. 
When using PCA and MNF it is safe to assume that the most helpful component will be in the first five, ten at most.
1. Skip files with UVB or UVP in the name may be helpful unless the registration error caused by the thickness of the filter has been corrected in software. 
1. FICA accepts options for the maximum number of iterations and tolerance. 
Adjusting these may impact processing time and address a warning during fitting, “FastICA did not converge.”
1. An option remains to save the output as a stack/cube. 
This only works with TIFF and presumes good resources for transferring and reading large amounts of data.
1. A cache can be used to preserve data from flattening and Blur and Divide between runs. 

Presently only one Region of Interest (ROI) can be run at a time. 
Multiple ROIs can be specified in metadata and selected in interactive mode.

Flattening and rotation are fully functional.
Relative path to flats should be specified in the project metadata file. 
Rotation relative to bit sequence should be specified in the project metadata file (if not 0). 
Beware of confusion introduced by file header metadata rotation. 

In addition to `process.py` and `options.py`, three scripts add additional functionality. 

1. `preview.py` creates jpeg files for each raw captured image. 
It applies flattening, rotation, gamma correction, and jpeg compression. 
It is not interactive and does not accept arguments, so it will be necessary to edit the code to specify file paths. 
1. `stack2rgb.bash` relies on imagemagick to convert three monochrome images into a single pseudocolor image. 
It takes four arguments. 
The first three are the filenames of the input channels without the paths. 
The script will search for those filenames in all directories under the parent of the current working directory. 
The fourth argument is the name of the file to be written in the current working directory. 
All possible combinations of the three input files into the RGB channels will be written with 1-6 appended to the output filename. 
1. `raking.py` produces color raking images from monochrome raking images and diffuse color images.

Some additional files are not generally recommended.

1. `white.py` has the ability to create white balance files readable by MegaVision PhotoShoot. 
It also can create spreadsheets of measurements across a session.
1. `color.py` can produce decent calibrated color when a colorchecker is not available but the wavelengths are known.
1. `measurecolor.py` was developed to compare the ΔE values of different LAB images and reference tables.
1. `checkercolor.py` is a standalone version of the “color” method built into process.py.

If most of this makes sense to you and we don’t already know each other, please be in touch. 
It’s a small community.

