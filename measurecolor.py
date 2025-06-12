#!/usr/bin/env python
import numpy as np
from skimage import io, exposure, img_as_ubyte, color # from skimage.color import xyz2rgb, xyz2lab, deltaE_ciede2000, lab2xyz
import yaml
import argparse
import glob
import os
from mapchecker import detectMacbeth
from calibratecolor import getArguments, measureCheckerValues, XyzDict2array

"""
measuredeltae.py takes color.yaml file, outputs deltaE2000 for each patch, average of 18, and average of 24 (stdout for now)
presently wants to be run in same directory as color.yaml, can be fixed
"""

verbose = False

def showCheckerValues(checkerValues):
	for number, values in enumerate(checkerValues):
		number = str(number+1)
		values = str(np.round(values,2))
		print("Patch",number,"has LAB values",values)

def showDetail(array):
	print(
		array.dtype,
		array.shape,
		'Ch.1',
		np.min(array[:,:,0]),
		np.max(array[:,:,0]),
		'Ch.2',
		np.min(array[:,:,1]),
		np.max(array[:,:,1]),
		'Ch.3',
		np.min(array[:,:,2]),
		np.max(array[:,:,2])
	)

if __name__ == "__main__":
	args = getArguments()
	verbose = args.verbose
	print("Gathering arguments from the command line") if verbose else None
	with open(args.colorfile,'r') as unparsedyaml:
		colordata = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
	print("Loading reference data based on spectrophotometer readings") if verbose else None
	if 'filename' in colordata['checker']['xyzvalues']:
		with open(colordata['checker']['xyzvalues']['filename'],'r') as unparsedyaml:
			xyzdict = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
			xyzdict = xyzdict[colordata['checker']['xyzvalues']['reference']]
	else:
		xyzdict = colordata['checker']['xyzvalues']
	checkerReference = XyzDict2array(xyzdict)
	checkerReference = color.xyz2lab(checkerReference,illuminant='D65')
	if False:
		showCheckerValues(checkerReference)
	print("Iterating through all the Color_LAB files in the Color/ directory") if verbose else None
	for labfile in glob.glob(os.path.join('Color','*Color_LAB*')):
		print(f"{labfile=}")
		img = io.imread(labfile)
		showDetail(img) # print(f"{img.dtype=} {img.shape=} L range {np.min(img[:,:,0])} - {np.max(img[:,:,0])}, A range {np.min(img[:,:,1])} - {np.max(img[:,:,1])}, B range {np.min(img[:,:,2])} - {np.max(img[:,:,2])}")
		checkerValues = measureCheckerValues(img,colordata['checker']) 
		checkerValues = np.array(checkerValues)
		if False:
			showCheckerValues(checkerValues)
		for number, value in enumerate(checkerValues):
			deltaE = color.deltaE_ciede2000(value,checkerReference[number])
			number = str(number+1)
			deltaE = str(np.round(deltaE,2))
			print("Patch",number,"has Euclidian Distance of",deltaE)
		deltaE = color.deltaE_ciede2000(checkerValues,checkerReference)
		print("Average Euclidian Distance is",np.round(np.mean(deltaE),3))

