#!/usr/bin/env python
import numpy as np
from skimage import io, exposure, img_as_ubyte, color
import yaml
import argparse
import glob
import os
from mapchecker import detectMacbeth

"""
calibratecolor.py takes color.yaml file, outputs matrix file
"""

verbose = False

def getArguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('colorfile', help='path to color.yaml',default="color.yaml",nargs="?")
	parser.add_argument('-v','--verbose',action='store_true')
	return parser.parse_args()

def measureCheckerValues(img,checkerMap):
	if 'automap' in checkerMap:
		if __name__ == "__main__":
			if 'c' in checkerMap['automap']:
				print("Reading rgb file to create automap") if verbose else None
				tempimg = io.imread(glob.glob(os.path.join(workingdir,colordata['imageset'],colordata['basefile']+'-'+checkerMap['automap']['c']+'*.tif'))[0])
			elif 'b' in checkerMap['automap']:
				print("Reading r, g, and b, files to create automap") if verbose else None
				redimg = io.imread(glob.glob(os.path.join(workingdir,colordata['imageset'],colordata['basefile']+'-'+checkerMap['automap']['r']+'*.tif'))[0])
				greenimg = io.imread(glob.glob(os.path.join(workingdir,colordata['imageset'],colordata['basefile']+'-'+checkerMap['automap']['g']+'*.tif'))[0])
				blueimg = io.imread(glob.glob(os.path.join(workingdir,colordata['imageset'],colordata['basefile']+'-'+checkerMap['automap']['b']+'*.tif'))[0])
				tempimg = np.dstack([redimg,greenimg,blueimg])
			else:
				print("I don't know how to perform this automap")
				exit()
		else: # assuming only called from measurecolor.py or measuredeltae.py
			tempimg = color.lab2rgb(img)
		if False: # is more gamma always better even if already accurate?
			tempimg = exposure.adjust_gamma(tempimg,1/2.2)
		tempimg = exposure.rescale_intensity(tempimg)
		tempimg = img_as_ubyte(tempimg)
		checkerMap = detectMacbeth(tempimg)
	checkerValues = []
	for patch in range(1,25):
		patchCube = img[
			checkerMap[patch]['y']:checkerMap[patch]['y']+checkerMap[patch]['h'],
			checkerMap[patch]['x']:checkerMap[patch]['x']+checkerMap[patch]['w'],
			:
		]
		patchMedian = np.median(patchCube,axis=[0,1])
		checkerValues.append(patchMedian)
	return checkerValues

def XyzDict2array(dict):
	array = []
	for i in range(1,25):
		chip = [ dict[i]['X'] , dict[i]['Y'], dict[i]['Z'] ]
		array.append(chip)
	array = np.array(array,dtype=np.float64)
	return array

if __name__ == "__main__":
	args = getArguments()
	verbose = args.verbose
	workingdir = os.path.dirname(args.colorfile)
	print("Gathering arguments from the command line") if verbose else None
	with open(args.colorfile,'r') as unparsedyaml:
		colordata = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
	capturedChecker = []
	for visibleBand in colordata['visibleCaptures']:
		filelist = glob.glob(os.path.join(workingdir,colordata['imageset'],colordata['basefile']+'-'+visibleBand+'*.tif'))
		if len(filelist) > 1:
			print(f"Warning: found more than one file for {visibleBand}, taking the first, which is {filelist[0]}") if verbose else None
		elif len(filelist) < 1:
			print(f"Failed to find files matching {os.path.join(workingdir,colordata['imageset'],colordata['basefile']+'-'+visibleBand+'*.tif')}")
			exit()
		print(f"Reading {filelist[0]}") if verbose else None
		img = io.imread(filelist[0])
		capturedChecker.append(img)
	capturedChecker = np.transpose(capturedChecker,axes=[1,2,0])
	checkerValues = measureCheckerValues(capturedChecker,colordata['checker'])
	checkerValues = np.array(checkerValues)
	if 'filename' in colordata['checker']['xyzvalues']:
		with open(os.path.join(workingdir,colordata['checker']['xyzvalues']['filename']),'r') as unparsedyaml:
			xyzdict = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
			xyzdict = xyzdict[colordata['checker']['xyzvalues']['reference']]
	else:
		xyzdict = colordata['checker']['xyzvalues']
	checkerReference = XyzDict2array(xyzdict)
	print("Calculating ratio of known patch values to measured patch values") if verbose else None
	checkerRatio = np.matmul( np.transpose(checkerReference) , np.transpose(np.linalg.pinv(checkerValues)) )
	print(f"Saving {colordata['msi2xyz']}")
	np.savetxt(os.path.join(workingdir,colordata['msi2xyz']),checkerRatio,header='Matrix of XYZ x MSI Wavelengths, load with np.loadtxt()') 
