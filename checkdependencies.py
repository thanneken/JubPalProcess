#!/usr/bin/env python

try:
	import sys
except:
	print("Failed to import sys")
else:
	print("Success! sys")
	print (sys.version)
try:
	from os import listdir, makedirs
except:
	print("Failed to import from os, supposed to be standard with python")
else:
	print("Success! os")
try:
	from os.path import exists, join
except:
	print("Failed to import from os, supposed to be standard with python")
else:
	print("Success! os.path")
try:
	import time 
except:
	print("Failed to import time, consider conda install conda-forge::time")
else:
	print("Success! time")
try:
	import numpy 
except:
	print("Failed to import numpy, consider conda install conda-forge::numpy")
else:
	print("Success! numpy")
try:
	import multiprocessing
except:
	print("Failed to import multiprocessing module, consider conda install conda-forge::multiprocess")
else:
	print("Success! multiprocessing")
try:
	from skimage import io, img_as_float32, img_as_uint, img_as_ubyte, filters, exposure, color
except:
	print("Failed to import from skimage, consider conda install conda-forge::scikit-image")
else:
	print("Success! skimage")
try:
	import yaml 
except:
	print("Failed to import yaml, consider conda install conda-forge::yaml")
else:
	print("Success! yaml")
try:
	import inquirer
except:
	print("Failed to import inquirer, consider conda install conda-forge::inquirer")
else:
	print("Success! inquirer")
try:
	import logging
except:
	print("Failed to import logging, try pip install logging")
else:
	print("Success! logging")
try:
	from sklearn.decomposition import FastICA, PCA, KernelPCA
except:
	print("Failed to import from sklearn.decomposition, consider conda install conda-forge::scikit-learn")
else:
	print("Success! sklearn.decomposition")
try:
	from sklearn.cross_decomposition import CCA
except:
	print("Failed to import from sklearn.cross_decomposition, consider conda install conda-forge::scikit-learn")
else:
	print("Success! sklearn.cross_decomposition")
try:
	from spectral import calc_stats, noise_from_diffs, mnf
except:
	print("Failed to import from spectral, consider conda install conda-forge::spectral")
else:
	print("Success! spectral")
try:
	import rawpy
except:
	print("Failed to import rawpy, will be necessary if plan to read raw image files, consider pip install rawpy")
else:
	print("Success! rawpy")
try:
	import pyexifinfo
except:
	print("Failed to import pyexifinfo, will be necessary to read rotation information from exif, consider pip install pyexifinfo")
else:
	print("Success! pyexifinfo")
try:
	from megaman.geometry import Geometry
except:
	print("Failed to import from megaman.geometry, only used for spectral embedding which is deprecated anyway")
else:
	print("Success! megaman.geometry")
try:
	from megaman.embedding import (Isomap, LocallyLinearEmbedding, LTSA, SpectralEmbedding)
except:
	print("Failed to import from megaman.embedding, only used for spectral embedding which is deprecated anyway")
else:
	print("Success! megaman.geometry")
