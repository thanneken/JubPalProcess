#!/usr/bin/env python
from skimage import io, img_as_ubyte, exposure, color
import cv2
import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt

verbose = False

def detectMacbeth(img):
	width = img.shape[1]
	if width > 10000:
		expectedppi = 1200
	elif width > 6500:
		expectedppi = 600
	elif width > 3000:
		expectedppi = 300
	else:
		expectedppi = 150
	expectedwh = int(expectedppi / 4)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	detector = cv2.mcc.CCheckerDetector.create()
	detector.process(img,chartType=cv2.mcc.MCC24)
	checker = detector.getBestColorChecker() # when multiple checkers are in frame will be necessary to specify a mask or edit the temporary color image
	patchcoordinates = checker.getColorCharts() # Q15 fails here
	checkerMap = {"note":"Upper left coordinates calculated by OpenCV Macbeth Color Checker Module. Assuming width and height of %s for %s ppi"%(expectedwh,expectedppi)}
	if False:
		patchcoordinates = np.flip(patchcoordinates,0)
	for patch in range(24):
		x,y = np.min(patchcoordinates[patch*4:patch*4+4],axis=0)
		x = int(x)
		y = int(y)
		lrx,lry = np.partition(patchcoordinates[patch*4:patch*4+4],-2,axis=0)[-2]
		lrx = int(lrx)
		lry = int(lry)
		maxwidth = lrx - x
		maxheight = lry - y
		if maxwidth < expectedwh or maxheight < expectedwh:
			print(f"{patch+1:>2} x,y,w,h = {x:>4},{y:>4},{maxwidth},{maxheight} may not safely give {expectedwh} square sample")
			if verbose:
				print(f"coordinates are\n{patchcoordinates[patch*4:patch*4+4]}")
		w = h = expectedwh
		checkerMap[patch+1] = {'x':x,'y':y,'w':w,'h':h}
		if verbose:
			print(f"{checkerMap[patch+1]=}")
	if verbose:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		drawer = cv2.mcc.CCheckerDraw.create(checker)
		drawn = drawer.draw(img)
		plt.imshow(drawn)
		plt.show()
	return checkerMap

def getArguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('-r','--red')
	parser.add_argument('-g','--green')
	parser.add_argument('-b','--blue')
	parser.add_argument('-c','--color')
	parser.add_argument('-y','--yaml')
	parser.add_argument('-v','--verbose',action='store_true')
	return parser.parse_args()

if __name__ == "__main__":
	print("Gathering arguments from the command line") if verbose else None
	args = getArguments()
	verbose = args.verbose # this way verbose will be defined if imported
	"""
	consider handling a directory and determining how best to create a temporary color image
	"""
	if args.color:
		img = io.imread(args.color)
		if 'Color_LAB' in args.color:
			img = color.lab2rgb(img)
	elif args.red and args.green and args.blue:
		redimg = io.imread(args.red)
		greenimg = io.imread(args.green)
		blueimg = io.imread(args.blue)
		img = np.dstack([redimg,greenimg,blueimg])
	else:
		print("It is necessary to specify either one color image with -c or three channels with -r -g -b")
		exit()
	img = exposure.adjust_gamma(img,1/2.2)
	img = exposure.rescale_intensity(img)
	img = img_as_ubyte(img)
	checkerMap = detectMacbeth(img)
	if args.yaml:
		with open(args.yaml,'w') as yamlfile:
			yaml.dump(checkerMap,yamlfile,sort_keys=False)
	else:
		print(yaml.dump(checkerMap,sort_keys=False))

