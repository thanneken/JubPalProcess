#!/usr/bin/env python
import numpy as np
from skimage import io, exposure, img_as_ubyte, color
import yaml
import argparse
import glob
import os
from calibratecolor import getArguments

"""
humancolor.py takes color.yaml, outputs LAB and sRGB
color.yaml must include entries for imageset, basefile, visibleCaptures, and msi2xyz 
(checker is for generating msi2xyz, not possible if no checker in frame)
"""

verbose = False

if __name__ == "__main__":
	args = getArguments()
	verbose = args.verbose
	print("Gathering arguments from the command line") if verbose else None
	with open(args.colorfile,'r') as unparsedyaml:
		colordata = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
	visibleCaptures = []
	workingdir = os.path.dirname(args.colorfile)
	for visibleCapture in colordata['visibleCaptures']:
		filelist = glob.glob(os.path.join(workingdir,colordata['imageset'],colordata['basefile']+'-'+visibleCapture+'*.tif'))
		if len(filelist) > 1:
			print(f"Warning: found more than one file for {visibleCapture}, taking the first, which is {filelist[0]}") if verbose else None
		if len(filelist) < 1:
			print(f"Did not find files matching glob {os.path.join(workingdir,colordata['imageset'],colordata['basefile']+'-'+visibleCapture+'*.tif')}")
			exit()
		print(f"Reading {filelist[0]}") if verbose else None
		img = io.imread(filelist[0])
		visibleCaptures.append(img)
		if '*' in colordata['basefile']:
			filename = os.path.basename(filelist[0])
			colordata['basefile'] = '-'.join(filename.split('-')[:5])
			print(f"Setting basefile to {colordata['basefile']}") if verbose else None
	visibleCaptures = np.transpose(visibleCaptures,axes=[1,2,0])
	height, width, layers = visibleCaptures.shape
	visibleCaptures = visibleCaptures.reshape(height*width,layers)
	checkerRatio = np.loadtxt(os.path.join(workingdir,colordata['msi2xyz']))
	calibratedColor = np.matmul( checkerRatio , np.transpose(visibleCaptures) )
	calibratedColor = np.transpose(calibratedColor)
	calibratedColor = calibratedColor.reshape(height,width,3)
	calibratedColor = np.clip(calibratedColor,0,1)
	srgb = color.xyz2rgb(calibratedColor)
	srgb = exposure.rescale_intensity(srgb) 
	srgb = img_as_ubyte(srgb)
	if not os.path.exists(os.path.join(workingdir,'Color')):
		os.makedirs(os.path.join(workingdir,'Color'),mode=0o755)
	if 'suffix' in colordata:
	 	colordata['suffix'] = '-'+colordata['suffix']
	else:
	 colordata['suffix'] = ''
	srgbFilePath = os.path.join(os.path.join(workingdir,'Color',colordata['basefile']+'-Color_sRGB'+colordata['suffix']+'.tif'))
	print(f"Saving sRGB {srgbFilePath}") if verbose else None
	io.imsave(srgbFilePath,srgb,check_contrast=False) 
	jpgFilePath = os.path.join(os.path.join(workingdir,'Color',colordata['basefile']+'-Color_sRGB'+colordata['suffix']+'.jpg'))
	print(f"Saving JPG {jpgFilePath}") if verbose else None
	io.imsave(jpgFilePath,srgb,check_contrast=False) 
	lab = color.xyz2lab(calibratedColor)
	lab = lab.astype('int8')
	labFilePath = os.path.join(os.path.join(workingdir,'Color',colordata['basefile']+'-Color_LAB'+colordata['suffix']+'.tif'))
	print(f"Saving LAB {labFilePath}") if verbose else None
	io.imsave(labFilePath,lab,check_contrast=False)

