#!/home/thanneken/python/miniconda3/bin/python
from os import listdir, makedirs, path, scandir
from skimage import io, img_as_ubyte, exposure
import rawpy
import numpy 

basePath = '/storage/JubPalProj/Lazarus/Triv_Dante/'
basePath = '/storage/JubPalProj/Lazarus/Triv_PsalterArmorial/'
basePath = '/storage/JubPalProj/Videntes/BritishLibrary_AddMS10049/'
basePath = '/storage/JubPalProj/Videntes/Unknown_MappaMundi/'
basePath = '/storage/JubPalProj/Ambrosiana2023/Ambrosiana_F130sup/'
print("Using basePath",basePath)
flatPath = 'not applicable'
flatPath = '/storage/JubPalProj/Ambrosiana2023/Calibration/Flats_D20230615-T151702/'
print("Using flatPath",flatPath)

def opentiffile(tiffile):
	img = io.imread(tiffile)
	return img
def openrawfile(rawfile):
	with rawpy.imread(rawfile) as raw:
		return raw.raw_image.copy() #img = raw.raw_image.copy() #return img
def flatten(img,imageIndex):
	for flat in listdir(flatPath):
		if flat[-7:-4] == imageIndex:
			flatFile = flatPath+flat
	flat = openrawfile(flatFile)
	return numpy.divide(img*numpy.average(flat),flat,out=numpy.zeros_like(img*numpy.average(flat)),where=flat!=0)
def rotate(img,side):
	if side == 'r':
		print("Using rotation for rectos")
		img = numpy.rot90(img,k=3)
	elif side == 'v':
		print("Using rotation for versos")
		img = numpy.rot90(img,k=1)
	else:
		print("Not doing any rotation because can't determine recto or verso")
	return img

# Make a list of directories in the basepath
sequences = []
with scandir(basePath) as it:
	for entry in it:
		if entry.is_dir():
			sequences.append(entry.name)

for sequence in sequences:
	previewPath = basePath+sequence+'/Preview/'
	if path.exists(previewPath): 
		print(sequence,"already has a Preview directory")
		continue
	print(sequence)
	makedirs(basePath+sequence+'/Preview/',mode=0o755,exist_ok=False)
	rawdir = basePath+sequence+'/Raw/'
	flatteneddir = basePath+sequence+'/Flattened/'
	if path.exists(rawdir):
		for rawfile in listdir(rawdir):
			print(rawfile)
			img = openrawfile(basePath+sequence+'/Raw/'+rawfile)
			img = flatten(img,rawfile[-7:-4])
			img = rotate(img,sequence[-1:])
			img = exposure.rescale_intensity(img)
			img = exposure.adjust_gamma(img,1/2.2)
			img = img_as_ubyte(img)
			io.imsave(basePath+sequence+'/Preview/'+rawfile[:-3]+'jpg',img,check_contrast=False)
	elif path.exists(flatteneddir):
		print("Using Flattened Directory")
		for flattenedfile in listdir(flatteneddir):
			print(flattenedfile)
			img = opentiffile(flatteneddir+flattenedfile)
		img = exposure.rescale_intensity(img)
		img = exposure.adjust_gamma(img,1/2.2)
		img = img_as_ubyte(img)
		io.imsave(basePath+sequence+'/Preview/'+flattenedfile[:-3]+'jpg',img,check_contrast=False)
	else:
		print("Not sure what I should be previewing here")
