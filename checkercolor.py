#!/home/thanneken/python/miniconda3/bin/python
from skimage import io, img_as_uint, img_as_ubyte, exposure, color
from os import path, listdir
import numpy 
import rawpy
import logging
import yaml 

## set variables specific to local system
inBasePath = '/storage/JubPalProj/Ambrosiana2023/Calibration/'
outBasePath = '/home/thanneken/Projects/Color/'
cachePath = '/storage/JubPalProj/cache/'
logfile = '/home/thanneken/checkercolor.log' 
sequence = 'Calibration-Color'
sequenceShort = 'Macbeth_Ambrosiana' 
sequenceShort = 'Calibration-Color'
illuminant = 'D65'
msi2xyzPath = '/home/thanneken/msi2xyz-checkercolor.txt'

## configure logging
logger = logging.getLogger(__name__) 
if path.exists(logfile): 
	logging.basicConfig(
		filename=logfile, 
		format='%(asctime)s %(levelname)s %(message)s', 
		datefmt='%Y%m%d %H:%M:%S', 
		level=logging.DEBUG # levels are DEBUG INFO WARNING ERROR CRITICAL
	) 
	logger.info("Logging configured for logfile "+logfile)
	print("Follow log file for detailed info and warnings:",logfile)
else:
	exit("Log file not specified or not found")

## going forward, logger.info can be thought of as documentation or at least the kind of explanation one might look for in comments
logger.info("Step 1: Gather reference XYZ values from spectroscan or manufacturer specifications")
logger.info("Using MegaVision spectroscan for checker reference XYZ values")
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
def XyzDict2array(dict):
	array = []
	for i in range(1,25):
		chip = [ dict[i]['X'] , dict[i]['Y'], dict[i]['Z'] ]
		array.append(chip)
	array = numpy.array(array,dtype=numpy.float64)
	return array
checkerReference = XyzDict2array(checkerXyzMegaVision)

logger.info("Step 2: Gather mean reflectance values of captured ColorChecker")
logger.info("Step 2.1: Open unflattened images as cube")

logger.info("Step 2.1.1: Open projects YAML file based on inBasePath")
projectsfile = inBasePath+inBasePath.split('/')[-2]+'.yaml'
if path.exists(projectsfile):
	with open(projectsfile,'r') as unparsedyaml:
		projects = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
else:
	exit('Unable to find '+projectsfile)

logger.info("Step 2.1.2: Gather data from projects file: location of white patch, rotation, visible bands, and flats")
try:
	whitex = projects[sequence]['white']['x']
except:
	try:
		whitex = projects['default']['white']['x']
	except:
		whitex = False
try:
	whitey = projects[sequence]['white']['y']
except:
	try:
		whitey = projects['default']['white']['y']
	except:
		whitey = False
try: 
	whitew = projects[sequence]['white']['w']
except:
	whitew = projects['default']['white']['w']
try:
	whiteh = projects[sequence]['white']['h']
except:
	whiteh = projects['default']['white']['h']
try: 
	rotation = projects[sequence]['rotation']
except:
	try: 
		rotation = projects['default']['rotation']
	except:
		rotation = 0
try:
	visibleBands = projects[sequence]['visibleBands']
except:
	visibleBands = projects['default']['visibleBands']
try:
	flatBasePath = inBasePath+projects[sequence]['flats']
except:
	flatBasePath = inBasePath+projects['default']['flats']

logger.info("Step 2.2: Flatten image cube or read flattened images from cache into cube")
capturedChecker = []
whiteLevels = []

def openrawfile(rawfile):
	with rawpy.imread(rawfile) as raw:
		return raw.raw_image.copy() 
def flatten():
	print("Opening",inBasePath+sequence+'/Raw/'+sequenceShort+'+'+visibleBand+'.dng')
	unflat = openrawfile(inBasePath+sequence+'/Raw/'+sequenceShort+'+'+visibleBand+'.dng')
	for flatFile in listdir(flatBasePath): 
		if flatFile[-7:-4] == visibleBand[-3:]:
			flatPath = flatBasePath+flatFile
	flat = openrawfile(flatPath)	
	return numpy.divide(unflat*numpy.average(flat),flat,out=numpy.zeros_like(unflat*numpy.average(flat)),where=flat!=0)
def rotate(img): 
	if rotation == 90:
		img = numpy.rot90(img,k=1)
	elif rotation == 180:
		img = numpy.rot90(img,k=2)
	elif rotation == 270:
		img = numpy.rot90(img,k=3)
	else:
		logger.info("No rotation identified")
	return img

for visibleBand in visibleBands:
	if not whitex and not whitey:
		continue
	cacheFilePath = cachePath+'flattened/'+sequenceShort+'+'+visibleBand+'.tif'
	if path.exists(cacheFilePath): 
		img = io.imread(cacheFilePath)
	else: 
		img = flatten()
		img = rotate(img)
		io.imsave(cacheFilePath,img,check_contrast=False)
	whiteSample = img[whitey:whitey+whiteh,whitex:whitex+whitew] # note y before x
	whiteLevel = numpy.percentile(whiteSample,84) # median plus 1 standard deviation is equal to 84.1 percentile
	capturedChecker.append(img)
	whiteLevels.append(whiteLevel)
capturedChecker = numpy.transpose(capturedChecker,axes=[1,2,0])

logger.info("Step 2.3: Normalize image cube")
def normalize(img):
	for i in range(img.shape[2]):
		img[:,:,i] = img[:,:,i] * numpy.max(whiteLevels) / whiteLevels[i]
	nearMax = numpy.percentile(img[whitey:whitey+whiteh,whitex:whitex+whitew,:],84)
	logger.info("Changing expected white luminance from .95 to .88 did not change ΔE. Manufacturer spec for white patch is 95%, Roy likes 88%.")
	img = img * 0.88 / nearMax 
	img = numpy.clip(img,0,1)
	return img
capturedChecker = normalize(capturedChecker)

logger.info("Step 2.4: Identify regions for 24 patches")
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
checkerMap = checkerMapJune2023

logger.info("Step 2.5: Measure mean reflectance of 24 patches for each layer in cube")
def measureCheckerValues(img,checkerMap):
	checkerValues = []
	for patch in range(1,25):
		patchCube = img[
			checkerMap[patch]['y']:checkerMap[patch]['y']+checkerMap[patch]['h'],
			checkerMap[patch]['x']:checkerMap[patch]['x']+checkerMap[patch]['w'],
			:
		]
		patchMedian = numpy.median(patchCube,axis=[0,1])
		checkerValues.append(patchMedian)
	return checkerValues
checkerValues = measureCheckerValues(capturedChecker,checkerMap)
checkerValues = numpy.array(checkerValues)

logger.info("Step 2.6: Calculate ratio of known patch values to measured patch values")
checkerRatio = numpy.matmul( numpy.transpose(checkerReference) , numpy.transpose(numpy.linalg.pinv(checkerValues)) )
# with open(msi2xyzPath,'w') as outfile:
# yaml.dump(checkerRatio.tolist(),outfile) 


numpy.savetxt(msi2xyzPath,checkerRatio,header='Matrix of XYZ x MSI Wavelengths, load with numpy.loadtxt()') #, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
# checkerRatio.tofile(msi2xyzPath,sep='\n')

exit()

logger.info("Step 3: Gather captured image data")
logger.info("Step 3.1: Open unflattened images as cube")
logger.info("Step 3.2: Flatten image cube")
logger.info("Step 3.3: Normalize image cube")
logger.warning("For now the capturedVisible and capturedChecker are one and the same")
capturedVisible = capturedChecker 

logger.info("Step 4: Derive calibrated color image")
def linearizeCameraResponse(response):
	logger.warning("These linearization values were determined for the MISHA system. A more adaptable method would be to perform a linear regression on the six neutral color checker patches.")
	response[:,0]  = 1.0465 * response[:,0]  - 0.1656
	response[:,1]  = 1.0007 * response[:,1]  - 0.0616
	response[:,2]  = 1.0158 * response[:,2]  - 0.0061
	response[:,3]  = 1.0176 * response[:,3]  - 0.0004
	response[:,4]  = 1.0162 * response[:,4]  + 0.0019
	response[:,5]  = 1.0171 * response[:,5]  + 0.0017
	response[:,6]  = 1.0178 * response[:,6]  + 0.0013
	response[:,7]  = 1.0181 * response[:,7]  + 0.0006
	response[:,8]  = 1.0186 * response[:,8]  + 0.0047
	response[:,9]  = 1.0191 * response[:,9]  + 0.0056
	response[:,10] = 1.0200 * response[:,10] + 0.0080
	response[:,11] = 1.0231 * response[:,11] + 0.0153
	return response
# checkerValues = linearizeCameraResponse(checkerValues)

logger.info("Step 4.1: Multiply captured image data by ratio of known patch values to measured patch values to produce XYZ calibrated color")
height,width,layers=capturedVisible.shape
capturedVisible = capturedVisible.reshape(height*width,layers)
# capturedVisible = linearizeCameraResponse(capturedVisible) # running linearize camera response on megavision data jumped DE from 0.848 to 4.787
calibratedColor = numpy.matmul( checkerRatio , numpy.transpose(capturedVisible))
calibratedColor = numpy.transpose(calibratedColor)
calibratedColor = calibratedColor.reshape(height,width,3)
calibratedColor = numpy.clip(calibratedColor,0,1)

def showDetail(array):
	logger.info(
		array.dtype+' '+
		array.shape+' '+
		'Ch.1 '+
		str(numpy.min(array[:,:,0]))+'-'+
		str(numpy.max(array[:,:,0]))+', '+
		'Ch.2 '+
		str(numpy.min(array[:,:,1]))+'-'+
		str(numpy.max(array[:,:,1]))+', '+
		'Ch.3 '+
		str(numpy.min(array[:,:,2]))+'-'+
		str(numpy.max(array[:,:,2]))
	)
# showDetail(calibratedColor)

logger.info("Step 4.2: Save image in sRGB color space")
srgb = color.xyz2rgb(calibratedColor)
srgb = exposure.rescale_intensity(srgb) # img = exposure.rescale_intensity(img,in_range=(0,nearMax),out_range=(0,1))
# img = numpy.clip(img,0,1) # not necessary with default rescale
srgb = img_as_ubyte(srgb)
outFilePath = '/home/thanneken/Projects/calibratedcolor.jpg'
logger.info("Saving jpeg file as "+outFilePath)
io.imsave(outFilePath,srgb,check_contrast=False)

logger.info("Step 4.3: Save image in LAB color space")
lab = color.xyz2lab(calibratedColor,illuminant=illuminant,observer='2')
# lab = lab.astype('float32')
# showDetail(lab)
# lab32Path = '/home/thanneken/Projects/calibratedcolor-lab.tif'
# io.imsave(lab32Path,lab,check_contrast=False)

logger.info("Step 5: Calculate CIEDE 2000")
def measureCheckerValues(img,checkerMap):
	checkerValues = []
	for patch in range(1,25):
		patchLAB = img[checkerMap[patch]['y']:checkerMap[patch]['y']+checkerMap[patch]['h'],checkerMap[patch]['x']:checkerMap[patch]['x']+checkerMap[patch]['w'],:]
		patchMedian = numpy.median(patchLAB,axis=[0,1])
		checkerValues.append(patchMedian)
		if numpy.std(patchLAB[:,:,0]) > 2:
			logger.warning("Patch "+str(patch)+" L channel has standard deviation of "+str(numpy.std(patchLAB[:,:,0])))
		if numpy.std(patchLAB[:,:,1]) > 2:
			logger.warning("Patch "+str(patch)+" A channel has standard deviation of "+str(numpy.std(patchLAB[:,:,1])))
		if numpy.std(patchLAB[:,:,2]) > 2:
			logger.warning("Patch "+str(patch)+" B channel has standard deviation of "+str(numpy.std(patchLAB[:,:,2])))
	return checkerValues
def showCheckerValues(checkerValues):
	for number, values in enumerate(checkerValues):
		number = str(number+1)
		values = str(numpy.round(values,2))
		logger.info("Patch "+number+" has LAB values "+values)
checkerValuesLab = measureCheckerValues(lab,checkerMap)
checkerReferenceLab = color.xyz2lab(checkerReference,illuminant=illuminant,observer='2')
# showCheckerValues(checkerReferenceLab)
# showCheckerValues(checkerValuesLab)
deltaE = color.deltaE_ciede2000(checkerReferenceLab,checkerValuesLab)
print("Average Euclidian distance between spectroscan and MSI calibrated color of 24 patches is",numpy.round(numpy.mean(deltaE),5))
logger.info("Average Euclidian distance between spectroscan and MSI calibrated color of 24 patches is "+str(numpy.round(numpy.mean(deltaE),5)))
logger.info("Calibrated Color Processing Complete\n=~=~=")

