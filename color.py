#!/home/thanneken/python/miniconda3/bin/python
from skimage import io, img_as_uint, img_as_ubyte, exposure
from os import listdir, makedirs, path
import sys
import rawpy
import numpy 
import yaml 
import pyexifinfo
from datetime import datetime

# Options (don't give boolean options variables and functions the same name)
displayStout = True
createPreviewJpg = True
createMegavisionTxt = False
createCsv = False
createColor = True
# Hard Code Paths 
inBasePath = '/storage/JubPalProj/Ambrosiana2023/Calibration/'
# inBasePath = '/storage/JubPalProj/Ambrosiana2023/Ambrosiana_F130sup/'
outBasePath = '/home/thanneken/Projects/Color/'
cachePath = '/storage/JubPalProj/cache/'
illuminant = 'D65'
observer = 'CIE 1931 2 Degree Standard Observer'
wavelengths = [400,420,450,470,505,530,560,590,615,630,655,700] 

if createColor:
	from colour import MSDS_CMFS,wavelength_to_XYZ,SDS_ILLUMINANTS, XYZ_to_sRGB
	from skimage.color import xyz2rgb
	cmfs = MSDS_CMFS[observer] # MSDS = multispectral distributions; CMFS = colour matching functions
	xyzWavelengths = wavelength_to_XYZ(wavelengths, cmfs)
	illuminantWavelenghs = SDS_ILLUMINANTS[illuminant][wavelengths] 

# Define Functions
def opentifffile(tiffile):
	img = io.imread(tiffile)
	return img
def openrawfile(rawfile):
	with rawpy.imread(rawfile) as raw:
		return raw.raw_image.copy() 
def flatten():
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
		print("No rotation identified")
	return img
def writePreviewJpg(): # save preview files of white square
	makedirs(outBasePath+sequence+'/White/',mode=0o755,exist_ok=True)
	img = exposure.rescale_intensity(whiteSample)
	img = img_as_ubyte(img)
	io.imsave(outBasePath+sequence+'/White/'+sequence+'_'+visibleBand+'.jpg',img,check_contrast=False)
def writeMegavisionTxt(): # write white balance file for MegaVision PhotoShoot or SpectraShoot
	f = open(outBasePath+sequence+'_W01-12.txt',"w") 
	f.write("WHITE-SET DATA \n    band count: "+str(len(visibleBands))+' \n  white levels:  ')
	for visibleBand in visibleBands:
		f.write('   '+dictionary[visibleBand])
	f.write('\n')
	f.close()
def writeCsv():
	import csv
	with open(outBasePath+'White.csv', 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile,fieldnames=list(dictionaries[0].keys()),dialect='excel')
		writer.writeheader()
		writer.writerows(dictionaries)
def timeFromExif():
	try:
		firstCapture = inBasePath+sequence+'/Raw/'+sequenceShort+'+RL450RB_001.dng'
		exif = pyexifinfo.get_json(firstCapture)
	except:
		firstCapture = inBasePath+sequence+'/Raw/'+listdir(inBasePath+sequence+'/Raw/')[0] 
		exif = pyexifinfo.get_json(firstCapture)
	exifTime = exif[0]["EXIF:DateTimeOriginal"]
	date = datetime.strptime(exifTime,'%Y:%m:%d %H:%M:%S')
	return date.strftime('%x %X')
def normalize(img):
	img = numpy.transpose(img,axes=[1,2,0])
	for i in range(img.shape[2]):
		img[:,:,i] = img[:,:,i] * numpy.max(whiteLevels) / whiteLevels[i]
	nearMax = numpy.mean(img[whitey:whitey+whiteh,whitex:whitex+whitew,:])
	img = img / nearMax
	img = numpy.clip(img,0,1)
	return img
def calculateSrgb(normCube):
	height,width,bands = normCube.shape
	normCube = normCube.reshape(height*width,bands) 
	print("Calculating color calibration matrix based on wavelength")
	illuminantMatrix = numpy.matmul(numpy.transpose(xyzWavelengths),(numpy.diagflat(illuminantWavelenghs)))
	imageMatrix = numpy.matmul(illuminantMatrix,(numpy.transpose(normCube)))
	XYZ = 1/100*imageMatrix
	XYZ = numpy.transpose(XYZ)
	XYZ = XYZ.reshape(height,width,3) 
	print("Converting XYZ to sRGB")
	SRGB = XYZ_to_sRGB(XYZ)
	SRGB = exposure.rescale_intensity(SRGB)
	return SRGB
def writeSrgb(img):
	print("Saving wavelength calibrated color image")
	makedirs(outBasePath+sequence+'/Color/',mode=0o755,exist_ok=True)
	jpegFilepath = outBasePath+sequence+'/Color/'+sequence+'PyλColor.jpg'
	img = img_as_ubyte(img)
	io.imsave(jpegFilepath,img,check_contrast=False)
def detint(img):
	print("Performing detint routine")
	whiteSample = img[whitey:whitey+whiteh,whitex:whitex+whitew] 
	meanWhite = numpy.mean(whiteSample[:,:,:],axis=(0,1))
	meanWhite = meanWhite.reshape(1,3)
	whiteIdeal = numpy.array([[243,243,242]])/255 # why is blue darker?
	correction = numpy.divide(whiteIdeal,meanWhite)
	correction = correction.reshape(1,1,3) # could be done with transpose?
	height, width, bands = img.shape
	pseudoImage = numpy.tile(correction,(height,width,1))
	img = img * pseudoImage
	img = numpy.clip(img,0,1)
	img = (img - numpy.min(img)) / (numpy.max(img) - numpy.min(img))
	makedirs(outBasePath+sequence+'/Color/',mode=0o755,exist_ok=True)
	jpegFilepath = outBasePath+sequence+'/Color/'+sequence+'PyλColor-detint.jpg'
	img = img_as_ubyte(img)
	io.imsave(jpegFilepath,img,check_contrast=False)

# Open YAML file based on inBasePath
projectsfile = inBasePath+inBasePath.split('/')[-2]+'.yaml'
if path.exists(projectsfile):
	with open(projectsfile,'r') as unparsedyaml:
		projects = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
else:
	exit('Unable to find '+projectsfile)

# Identify sequences to process
if len(sys.argv) > 1:
	sequences = sys.argv[1:]
else:
	sequences = []
	for directoryEntry in listdir(inBasePath): 
		try:
			projects[directoryEntry]['white']
		except:
			continue
		sequences.append(directoryEntry)

# iterate over each sequence
dictionaries = []
for sequence in sequences:
	# sequenceShort = sequence[11:] # capture filenames lack Ambrosiana_
	sequenceShort = 'Macbeth_Ambrosiana' # ad hoc for one process
	# will often be necessary if we don't enforce file names
	timeStamp = timeFromExif()
	try:
		note = projects[sequence]['white']['note']
	except:
		note = ''
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
		visibleBands = projects[sequence]['visiblebands']
	except:
		visibleBands = projects['default']['visiblebands']
	try:
		flatBasePath = inBasePath+projects[sequence]['flats']
	except:
		flatBasePath = inBasePath+projects['default']['flats']
	dictionary = {
		'TIME': timeStamp,
		'SEQUENCE': sequence,
		'NOTE': note,
		'WHITEX': whitex,
		'WHITEY': whitey,
		'WHITEW': whitew,
		'WHITEH': whiteh
	}
	visibleCube = []
	whiteLevels = []
	for visibleBand in visibleBands:
		if not whitex and not whitey:
			continue
		cacheFilePath = cachePath+'flattened/'+sequenceShort+'+'+visibleBand+'.tif'
		if path.exists(cacheFilePath): 
			img = opentifffile(cacheFilePath) 
		else:
			img = flatten()
			img = rotate(img)
			io.imsave(cacheFilePath,img,check_contrast=False)
		whiteSample = img[whitey:whitey+whiteh,whitex:whitex+whitew] # note y before x
		whiteLevel = round(numpy.median(whiteSample)+numpy.std(whiteSample),3) # tested 20 pages and median+std more consistent than median or mean
		dictionary[visibleBand] = str(whiteLevel) 
		if displayStout:
			print(sequence+'_'+visibleBand,"White Level:",whiteLevel)
		if createPreviewJpg:
			writePreviewJpg()
		if createColor:
			visibleCube.append(img)
			whiteLevels.append(whiteLevel)
	dictionaries.append(dictionary)
	if createColor:
		normCube = normalize(visibleCube)
		SRGB = calculateSrgb(normCube)
		writeSrgb(SRGB)
		detint(SRGB)
	if createMegavisionTxt and (whitex or whitey):
		writeMegavisionTxt()
if createCsv:
	writeCsv()

