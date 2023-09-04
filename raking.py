#!/home/thanneken/python/miniconda3/bin/python
from skimage import io, img_as_ubyte, exposure, color
from os import listdir, makedirs, path
import numpy
import rawpy
import sys
import yaml 

def openrawfile(rawfile):
	with rawpy.imread(rawfile) as raw:
		return raw.raw_image.copy() 
def flatten(unflat):
	try:
		flatBasePath = basePath+projects[sequence]['flats']
	except:
		flatBasePath = basePath+projects['default']['flats']
	for flatFile in listdir(flatBasePath): 
		#if flatFile[-7:-4] == unflat[-3:]: # currently unflat is an image, not a filename
		if flatFile[-7:-4] == rakingCapture[0][-3:]: 
			flatPath = flatBasePath+flatFile
	flat = openrawfile(flatPath)	
	return numpy.divide(unflat*numpy.average(flat),flat,out=numpy.zeros_like(unflat*numpy.average(flat)),where=flat!=0)
def rotate(img): 
	try: 
		rotation = projects[sequence]['rotation']
	except:
		try: 
			rotation = projects['default']['rotation']
		except:
			rotation = 0
	if rotation == 90:
		img = numpy.rot90(img,k=1)
	elif rotation == 180:
		img = numpy.rot90(img,k=2)
	elif rotation == 270:
		img = numpy.rot90(img,k=3)
	elif rotation == 0:
		print("No rotation requested")
	else:
		print("No rotation identified")
	return img
def imginfo(img):
	print(img.dtype,img.shape)
	print("Channel 1",numpy.amin(img[:,:,0]),numpy.amax(img[:,:,0]))
	print("Channel 2",numpy.amin(img[:,:,1]),numpy.amax(img[:,:,1]))
	print("Channel 3",numpy.amin(img[:,:,2]),numpy.amax(img[:,:,2]))
def lSub(raking):
	raking = raking * 100 # L scale is 0-100
	rakingLab = numpy.array([raking,diffuse[:,:,1],diffuse[:,:,2]])
	rakingLab = numpy.transpose(rakingLab,axes=[1,2,0])
	img = color.lab2rgb(rakingLab)
	return img_as_ubyte(img)
	
# Variables
gamma = 1/2.2
rakingCaptures = [
	['+RL450RB_001','Raking1'],
	['+RR940IR_002','Raking2'],
	['_003','Raking3'],
	['_004','Raking4'],
	['_005','Raking5'],
	['_006','Raking6']
]
basePath = '/storage/JubPalProj/Ambrosiana2023/Ambrosiana_F130sup/'

# Open YAML file based on basePath (needed for rotation and flat path)
projectsfile = basePath+basePath.split('/')[-2]+'.yaml'
if path.exists(projectsfile):
	with open(projectsfile,'r') as unparsedyaml:
		projects = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
else:
	exit('Unable to find '+projectsfile+' needed to identify rotation and flat path')

# Identify sequences to process
if len(sys.argv) > 1:
	sequences = sys.argv[1:]
else:
	sequences = []
	for directoryEntry in listdir(basePath): 
		diffuseRgbPath = basePath+directoryEntry+'/Color/'+directoryEntry+'_sRGB.tif'
		raking1RgbPath = basePath+directoryEntry+'/Color/'+directoryEntry+'_Raking1.tif'
		if path.exists(diffuseRgbPath) and not path.exists(raking1RgbPath):
			sequences.append(directoryEntry)
print("Working on",sequences)

for sequence in sequences: # Iterate over sequences
	print("Working on",sequence)
	diffuse = io.imread(basePath+sequence+'/Color/'+sequence+'_sRGB.tif')
	imginfo(diffuse)
	print("Converting diffuse color from RGB to LAB")
	diffuse = color.rgb2lab(diffuse)
	imginfo(diffuse)
	for rakingCapture in rakingCaptures: # Iterate over raking captures
		print("Working on",rakingCapture[0])
		if path.exists(basePath+sequence+'/Raw/'+sequence+rakingCapture[0]+'.dng'):
			raking = openrawfile(basePath+sequence+'/Raw/'+sequence+rakingCapture[0]+'.dng')
		elif path.exists(basePath+sequence+'/Raw/'+sequence[11:]+rakingCapture[0]+'.dng'):
			raking = openrawfile(basePath+sequence+'/Raw/'+sequence[11:]+rakingCapture[0]+'.dng')
		else:
			print("Couldn't find a raw file for",rakingCapture[0])
			continue
		raking = flatten(raking)
		raking = rotate(raking)
		raking = exposure.adjust_gamma(raking,gamma) 
		raking = exposure.rescale_intensity(raking)
		rakingRgb = lSub(raking)
		imginfo(rakingRgb)
		print("Saving",sequence+'_'+rakingCapture[1]+'.tif')
		io.imsave(basePath+sequence+'/Color/'+sequence+'_'+rakingCapture[1]+'.tif',rakingRgb)
		print("Saving",sequence+'_'+rakingCapture[1]+'.jpg')
		io.imsave(basePath+sequence+'/Color/'+sequence+'_'+rakingCapture[1]+'.jpg',rakingRgb)

