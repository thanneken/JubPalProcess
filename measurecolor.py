#!/home/thanneken/python/miniconda3/bin/python
from skimage import io, img_as_uint, img_as_ubyte, exposure
from os import listdir, makedirs, path
import sys
import rawpy
import numpy 
import yaml 
import pyexifinfo
from datetime import datetime
# from colour import MSDS_CMFS,wavelength_to_XYZ,SDS_ILLUMINANTS, XYZ_to_sRGB
from skimage.color import xyz2rgb, xyz2lab, deltaE_ciede2000, lab2xyz
import csv

chartPscPath = '/storage/JubPalProj/Ambrosiana2023/Calibration/Calibration-Color/Color/MacbethAmbrosiana-PythonWhite_PSC.tif'
chartPsc32Path = '/storage/JubPalProj/Ambrosiana2023/Calibration/Calibration-Color/Color/MacbethAmbrosiana-PythonWhite_PSC_32bit.tif'
chartWavelength32Path = '/home/thanneken/Projects/Color/Calibration-Color/Color/Calibration-Color-PyWavelengthColor-LAB32.tif'
detintPath = '/home/thanneken/Projects/Color/Calibration-Color/Color/Calibration-Color-PyWavelengthColor-detint-LAB32.tif'
checkerColor = '/home/thanneken/Projects/calibratedcolor-lab.tif'
checkerMapJune2023 = {
	1 : { 'x': 4442, 'y':3340, 'w':180, 'h':180 },
	2 : { 'x': 4126, 'y':3347, 'w':180, 'h':180 },
	3 : { 'x': 3818, 'y':3352, 'w':180, 'h':180 },
	4 : { 'x': 3514, 'y':3363, 'w':180, 'h':180 },
	5 : { 'x': 3194, 'y':3368, 'w':180, 'h':180 },
	6 : { 'x': 2885, 'y':3374, 'w':180, 'h':180 },
	7 : { 'x': 4434, 'y':3025, 'w':180, 'h':180 },
	8 : { 'x': 4123, 'y':3029, 'w':180, 'h':180 },
	9 : { 'x': 3813, 'y':3039, 'w':180, 'h':180 },
	10 : { 'x': 3504, 'y':3043, 'w':180, 'h':180 },
	11 : { 'x': 3192, 'y':3057, 'w':180, 'h':180 },
	12 : { 'x': 2878, 'y':3061, 'w':180, 'h':180 },
	13 : { 'x': 4428, 'y':2712, 'w':180, 'h':180 },
	14 : { 'x': 4115, 'y':2722, 'w':180, 'h':180 },
	15 : { 'x': 3804, 'y':2729, 'w':180, 'h':180 },
	16 : { 'x': 3491, 'y':2739, 'w':180, 'h':180 },
	17 : { 'x': 3182, 'y':2752, 'w':180, 'h':180 },
	18 : { 'x': 2873, 'y':2750, 'w':180, 'h':180 },
	19 : { 'x': 4417, 'y':2403, 'w':180, 'h':180 },
	20 : { 'x': 4109, 'y':2408, 'w':180, 'h':180 },
	21 : { 'x': 3794, 'y':2425, 'w':180, 'h':180 },
	22 : { 'x': 3486, 'y':2429, 'w':180, 'h':180 },
	23 : { 'x': 3173, 'y':2430, 'w':180, 'h':180 },
	24 : { 'x': 2865, 'y':2436, 'w':180, 'h':180 },
	'brand':'MegaVision',
	'serial':'110306',
	'owner':'Gregory Heyworth, Lazarus Project',
	'data':'https://palimpsest.stmarytx.edu/Ambrosiana2023/Calibration/Calibration-Color/',
	'note':'Captured Milan June 2023 with MegaVision E7, bitwise rotation so chart appears upside down'
}

checkerXYZvfxbrain = {
	1: { 'X': 0.03111, 'Y': 0.02629, 'Z': 0.01541 },
	2: { 'X': 0.10290, 'Y': 0.09088, 'Z': 0.06234 },
	3: { 'X': 0.05270, 'Y': 0.05385, 'Z': 0.08563 },
	4: { 'X': 0.02835, 'Y': 0.03411, 'Z': 0.01497 },
	5: { 'X': 0.06893, 'Y': 0.06282, 'Z': 0.10730 },
	6: { 'X': 0.08456, 'Y': 0.10950, 'Z': 0.10450 },
	7: { 'X': 0.09765, 'Y': 0.07284, 'Z': 0.01589 },
	8: { 'X': 0.03641, 'Y': 0.03248, 'Z': 0.09210 },
	9: { 'X': 0.07415, 'Y': 0.04884, 'Z': 0.03259 },
	10: { 'X': 0.02460, 'Y': 0.01746, 'Z': 0.03928 },
	11: { 'X': 0.09227, 'Y': 0.11100, 'Z': 0.02473 },
	12: { 'X': 0.12360, 'Y': 0.10490, 'Z': 0.01807 },
	13: { 'X': 0.02283, 'Y': 0.01767, 'Z': 0.07255 },
	14: { 'X': 0.03873, 'Y': 0.05833, 'Z': 0.02233 },
	15: { 'X': 0.05931, 'Y': 0.03446, 'Z': 0.01470 },
	16: { 'X': 0.14990, 'Y': 0.14890, 'Z': 0.02116 },
	17: { 'X': 0.08241, 'Y': 0.05205, 'Z': 0.07521 },
	18: { 'X': 0.03751, 'Y': 0.05169, 'Z': 0.09082 },
	19: { 'X': 0.22450, 'Y': 0.22580, 'Z': 0.22300 },
	20: { 'X': 0.14850, 'Y': 0.14870, 'Z': 0.14940 },
	21: { 'X': 0.09288, 'Y': 0.09276, 'Z': 0.09299 },
	22: { 'X': 0.05056, 'Y': 0.05073, 'Z': 0.05049 },
	23: { 'X': 0.02416, 'Y': 0.02403, 'Z': 0.02408 },
	24: { 'X': 0.008623, 'Y': 0.008516, 'Z': 0.008604 },
	'source':'https://vfxbrain.wordpress.com/2017/08/30/macbeth-colour-checker-reference/'
}

checkerXyzMegaVision = {
	1: { 'X': 0.119367, 'Y': 0.104967, 'Z': 0.054100 },
	2: { 'X': 0.379000, 'Y': 0.337300, 'Z': 0.179300 },
	3: { 'X': 0.166800, 'Y': 0.182800, 'Z': 0.257467 },
	4: { 'X': 0.113933, 'Y': 0.137433, 'Z': 0.054700 },
	5: { 'X': 0.245400, 'Y': 0.234100, 'Z': 0.333933 },
	6: { 'X': 0.307067, 'Y': 0.422233, 'Z': 0.342833 },
	7: { 'X': 0.408400, 'Y': 0.313133, 'Z': 0.051100 },
	8: { 'X': 0.119633, 'Y': 0.110667, 'Z': 0.280967 },
	9: { 'X': 0.298600, 'Y': 0.194600, 'Z': 0.101533 },
	10: { 'X': 0.085300, 'Y': 0.066567, 'Z': 0.108167 },
	11: { 'X': 0.349100, 'Y': 0.435167, 'Z': 0.088467 },
	12: { 'X': 0.493500, 'Y': 0.437133, 'Z': 0.061000 },
	13: { 'X': 0.065733, 'Y': 0.058067, 'Z': 0.203267 },
	14: { 'X': 0.148833, 'Y': 0.231600, 'Z': 0.075867 },
	15: { 'X': 0.214867, 'Y': 0.127333, 'Z': 0.040667 },
	16: { 'X': 0.604300, 'Y': 0.604400, 'Z': 0.073700 },
	17: { 'X': 0.313533, 'Y': 0.203167, 'Z': 0.238500 },
	18: { 'X': 0.137167, 'Y': 0.195300, 'Z': 0.307933 },
	19: { 'X': 0.888133, 'Y': 0.924233, 'Z': 0.719400 },
	20: { 'X': 0.574200, 'Y': 0.597400, 'Z': 0.488833 },
	21: { 'X': 0.351867, 'Y': 0.367400, 'Z': 0.303300 },
	22: { 'X': 0.180100, 'Y': 0.187700, 'Z': 0.154967 },
	23: { 'X': 0.086867, 'Y': 0.091000, 'Z': 0.076933 },
	24: { 'X': 0.029267, 'Y': 0.030367, 'Z': 0.026333 },
	'source': 'ClrChckrClsc_130903_160413_D50_RefClrData.txt; Color Checker Classic #130903, Avg of 3 from MV Spectroscan on 2013.09.17; D50'
}

def showDetail(array):
	print(
		array.dtype,
		array.shape,
		'Ch.1',
		numpy.min(array[:,:,0]),
		numpy.max(array[:,:,0]),
		'Ch.2',
		numpy.min(array[:,:,1]),
		numpy.max(array[:,:,1]),
		'Ch.3',
		numpy.min(array[:,:,2]),
		numpy.max(array[:,:,2])
	)
def XyzDict2array(dict):
	array = []
	for i in range(1,25):
		chip = [ dict[i]['X'] , dict[i]['Y'], dict[i]['Z'] ]
		array.append(chip)
	array = numpy.array(array,dtype=numpy.float64)
	return array
def mvLab2fpLab(img): 
	print("Convert MegaVision PSC Lab to 32-bit floating point standard scale LAB")
	print("Rescaling L from 0-255 to 0-100")
	img = img.astype('float32')
	img[:,:,0] = img[:,:,0]*100/255
	print("Faster to vectorize than to iterate, but can't figure out how to subtract from a value rather than replace it.") # img[img > 127] = -255
	print("Rescaling A and B from 0—255 to -128—127")
	for i in numpy.nditer(img[:,:,1:3],op_flags=['readwrite']):
		if i > 127:
			i[...] = i-256
	print("Done converting MegaVision PSC LAB to 32-bit floating point standard scale LAB")
	return img
def measureCheckerValues(img,checkerMap):
	checkerValues = []
	for patch in range(1,25):
		patchLAB = img[checkerMap[patch]['y']:checkerMap[patch]['y']+checkerMap[patch]['h'],checkerMap[patch]['x']:checkerMap[patch]['x']+checkerMap[patch]['w'],:]
		patchMedian = numpy.median(patchLAB,axis=[0,1])
		checkerValues.append(patchMedian)
		if numpy.std(patchLAB[:,:,0]) > 2:
			print("Patch",patch,"L channel has standard deviation of",numpy.std(patchLAB[:,:,0]))
		if numpy.std(patchLAB[:,:,1]) > 2:
			print("Patch",patch,"A channel has standard deviation of",numpy.std(patchLAB[:,:,1]))
		if numpy.std(patchLAB[:,:,2]) > 2:
			print("Patch",patch,"B channel has standard deviation of",numpy.std(patchLAB[:,:,2]))
	return checkerValues
def showCheckerValues(checkerValues):
	for number, values in enumerate(checkerValues):
		number = str(number+1)
		values = str(numpy.round(values,2))
		print("Patch",number,"has LAB values",values)
def fetchPsc(path8,path32):
	if path.exists(path32):
		return io.imread(path32)
	else:
		print("Reading",path8)
		img = io.imread(path8)
		print("Rotating")
		img = numpy.rot90(img,k=2)
		img = mvLab2fpLab(img)
		print("Saving",path32)
		io.imsave(path32,img,check_contrast=False)
		return img

checkerMap = checkerMapJune2023
optionLabels = {}

print("MegaVision Spectroscan D50:")
checkerValuesXyz = XyzDict2array(checkerXyzMegaVision)
checkerValues = xyz2lab(checkerValuesXyz,illuminant='D50')
# showCheckerValues(checkerValues)
optionLabels['MegaVision Spectroscan D50'] = checkerValues

print("MegaVision PhotoShootColor:")
img = fetchPsc(chartPscPath,chartPsc32Path)
showDetail(img)
checkerValues = measureCheckerValues(img,checkerMap)
showCheckerValues(checkerValues)
optionLabels['MegaVision PhotoShootColor'] = checkerValues

print("Checker Color:")
img = io.imread(checkerColor)
showDetail(img)
checkerValues = measureCheckerValues(img,checkerMap)
showCheckerValues(checkerValues)
optionLabels['Python Checker Color'] = checkerValues

print("Wavelength Method LAB 32:")
img = io.imread(chartWavelength32Path)
showDetail(img)
checkerValues = measureCheckerValues(img,checkerMap)
showCheckerValues(checkerValues)
optionLabels['Wavelength Method No Detint'] = checkerValues

for number, label in enumerate(optionLabels.keys()):
	print(number+1,label)

for number1, option1 in enumerate(optionLabels.values()):
	for number2, option2 in enumerate(optionLabels.values()):
		if number1 >= number2:
			continue
		deltaE = deltaE_ciede2000(option1,option2)
		print(number1+1,' ~ ',number2+1,' = ',numpy.round(numpy.mean(deltaE),3))

img = fetchPsc(chartPscPath,chartPsc32Path)
showDetail(img)
xyz = lab2xyz(img)
showDetail(xyz)	

exit()

print("Internet reported ColorChecker specs:")
checkerValuesXyz = XyzDict2array(checkerXYZvfxbrain)
checkerValues = xyz2lab(checkerValuesXyz,illuminant='D50')
showCheckerValues(checkerValues)
optionLabels['Internet reported ColorChecker D50'] = checkerValues

print("Internet reported ColorChecker specs:")
checkerValuesXyz = XyzDict2array(checkerXYZvfxbrain)
checkerValues = xyz2lab(checkerValuesXyz,illuminant='D65')
showCheckerValues(checkerValues)
optionLabels['Internet reported ColorChecker D65'] = checkerValues

# array = exposure.rescale_intensity(array,out_range=(numpy.min(checkerXyzMegaVisionArray),numpy.max(checkerXyzMegaVisionArray)))
internetValuesD65 = xyz2lab(array,illuminant='D65')

options = [internetValuesD50,internetValuesD65, checkerValuesDetint]
optionLabels = {
	'PhotoShoot Color' : checkerValues1 ,
	'MegaVision Spectroscan D50' : checkerLabMegaVision 
}


print("PhotoShoot Color versus MegaVision SpectroScan D50")
deltaE = deltaE_ciede2000(checkerValues1,checkerLabMegaVision)
print(numpy.mean(deltaE),deltaE)

print("Internet ColorChecker XYZ D50 versus MegaVision Spectroscan D50")
deltaE = deltaE_ciede2000(internetValuesD50,checkerLabMegaVision)
print(numpy.mean(deltaE),deltaE)

print("PhotoShoot Color versus Wavelength Method")
deltaE = deltaE_ciede2000(checkerValues1,checkerValues2)
print(numpy.mean(deltaE),deltaE)

print("PhotoShoot Color versus Wavelength Method with Detint")
deltaE = deltaE_ciede2000(checkerValues1,checkerValuesDetint)
print(numpy.mean(deltaE),deltaE)

print("Internet ColorChecker XYZ D65 versus Wavelength Method with Detint")
deltaE = deltaE_ciede2000(internetValuesD65,checkerValuesDetint)
print(numpy.mean(deltaE),deltaE)

print("Internet ColorChecker XYZ D50 versus Wavelength Method with Detint")
deltaE = deltaE_ciede2000(internetValuesD50,checkerValuesDetint)
print(numpy.mean(deltaE),deltaE)

