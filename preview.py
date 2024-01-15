#!/home/thanneken/python/miniconda3/bin/python
from os import listdir, makedirs, path, scandir
from skimage import io, img_as_ubyte, exposure
import rawpy
import numpy 
import yaml 
import multiprocessing

basePath = '/storage/JubPalProj/Lazarus/Triv_Dante/'
basePath = '/storage/JubPalProj/Lazarus/Triv_PsalterArmorial/'
basePath = '/storage/JubPalProj/Videntes/BritishLibrary_AddMS10049/'
basePath = '/storage/JubPalProj/Videntes/Unknown_MappaMundi/'
basePath = '/storage/JubPalProj/Ambrosiana2023/Ambrosiana_F130sup/'
basePath = '/storage/JubPalProj/Hereford/Hereford_Gospels/'
print("Using basePath",basePath)
flatPath = '/storage/JubPalProj/Ambrosiana2023/Calibration/Flats_D20230615-T151702/'
flatPath = '/storage/JubPalProj/Hereford/Calibration/Flat20231011/'
flatPath = 'not applicable'
flatPath = 'Derive from YAML file'
yamlPath = '/storage/JubPalProj/Hereford/Hereford_Gospels/Hereford_Gospels.yaml'
flatIndexStart = -7 # MegaVision
flatIndexEnd = -4 # MegaVision
flatIndexStart = -10 # Hereford, Misha
flatIndexEnd = -7 # Hereford, Misha
rotation = 0
print("Using flatPath",flatPath)
verbose = False

def opentiffile(tiffile):
	img = io.imread(tiffile)
	return img
def openrawfile(rawfile):
	with rawpy.imread(rawfile) as raw:
		return raw.raw_image.copy() 
def flatten(img,imageIndex):
	for flat in listdir(flatPath):
		if flat[flatIndexStart:flatIndexEnd] == imageIndex:
			flatFile = flatPath+flat
	if flatFile.endswith('.dng'):
		flat = openrawfile(flatFile) 
	else:
		flat = opentiffile(flatFile)
	return numpy.divide(img*numpy.average(flat),flat,out=numpy.zeros_like(img*numpy.average(flat)),where=flat!=0)
def rotate(img,side):
	if rotation == 0:
		if verbose:
			print("Rotation specified as 0")
	elif rotation == 90:
		if verbose:
			print("Using rotation 90")
		img = numpy.rot90(img,k=3)
	elif rotation == 180:
		if verbose:
			print("Using rotation 180")
		img = numpy.rot90(img,k=2)
	elif rotation == 270:
		if verbose:
			print("Using rotation 270")
		img = numpy.rot90(img,k=1)
	elif side == 'r':
		if verbose:
			print("Using rotation for rectos")
		img = numpy.rot90(img,k=3)
	elif side == 'v':
		if verbose:
			print("Using rotation for versos")
		img = numpy.rot90(img,k=1)
	else:
		if verbose:
			print("Not doing any rotation because can't determine recto or verso")
	return img
def previewJpeg(infile,outfile):
	print("Flattening",infile,"to",outfile)
	img = opentiffile(infile)
	if not sequence.startswith('Flat'):
		img = flatten(img,infile[flatIndexStart:flatIndexEnd])
	img = rotate(img,sequence[-1:])
	img = exposure.rescale_intensity(img)
	img = exposure.adjust_gamma(img,1/2.2)
	img = img_as_ubyte(img)
	io.imsave(outfile,img,check_contrast=False)

# Make a list of directories in the basepath
if path.exists(yamlPath): 
	with open(yamlPath,'r') as unparsedyaml:
			metadata = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
	sequences = list(metadata.keys())
	sequences.remove('default')
else:
	sequences = []
	with scandir(basePath) as it:
		for entry in it:
			if entry.is_dir():
				sequences.append(entry.name)

if False:
	metadata = {}
	metadata.update(projects['default'])
	metadata.update(projects[project])
	metadata['white'].update(projects['default']['white'])
	metadata['white'].update(projects[project]['white'])

for sequence in sequences:
	print("\nWorking on sequence",sequence)
	if metadata: # get flats and rotation from metadata
		if metadata[sequence]['flats']:
			flatPath = basePath+metadata[sequence]['flats']
			if verbose:
				print("Path to flats found in YAML metadata for sequence:",flatPath)
		elif metadata['default']['flats']:
			flatPath = basePath+metadata['default']['flats']
			if verbose:
				print("Path to flats found in YAML metadata default:",flatPath)
		else:
			if verbose:
				print("Path to flats not found in YAML metadata, using",flatPath)
		if metadata[sequence]['rotation']:
			rotation = metadata[sequence]['rotation']
			if verbose:
				print("Rotation found in YAML metadata for sequence:",rotation)
		elif metadata['default']['rotation']:
			rotation = metadata['default']['rotation']
			if verbose:
				print("Rotation found in YAML metadata default:",rotation)
		else:
			if verbose:
				print("No rotation found in YAML metadata, using",rotation)
	previewPath = basePath+sequence+'/Preview/'
	if path.exists(previewPath): 
		print(sequence,"already has a Preview directory")
		continue
	makedirs(basePath+sequence+'/Preview/',mode=0o775,exist_ok=False)
	rawdir = basePath+sequence+'/Raw/'
	flatteneddir = basePath+sequence+'/Flattened/'
	unflattenedDir = basePath+sequence+'/Unflattened/'
	if path.exists(flatteneddir):
		print("Using Flattened Directory")
		for flattenedfile in listdir(flatteneddir):
			print(flattenedfile)
			img = opentiffile(flatteneddir+flattenedfile)
			img = exposure.rescale_intensity(img)
			img = exposure.adjust_gamma(img,1/2.2)
			img = img_as_ubyte(img)
			io.imsave(basePath+sequence+'/Preview/'+flattenedfile[:-3]+'jpg',img,check_contrast=False)
	elif path.exists(rawdir):
		for rawfile in listdir(rawdir):
			print(rawfile)
			img = openrawfile(basePath+sequence+'/Raw/'+rawfile)
			img = flatten(img,rawfile[-7:-4])
			img = rotate(img,sequence[-1:])
			img = exposure.rescale_intensity(img)
			img = exposure.adjust_gamma(img,1/2.2)
			img = img_as_ubyte(img)
			io.imsave(basePath+sequence+'/Preview/'+rawfile[:-3]+'jpg',img,check_contrast=False)
	elif path.exists(unflattenedDir):
		for unflatFile in listdir(unflattenedDir): 
			infile = unflattenedDir+unflatFile
			outfile = basePath+sequence+'/Preview/'+unflatFile[:-3]+'jpg'
			p = multiprocessing.Process(target=previewJpeg,args=(infile,outfile)) 
			p.start()
		p.join() # perhaps should be inside loop but works fine, might be responsible for zombie processes
	else:
		print("Not sure what I should be previewing here")

# Not needed because not reading back from multiprocessing queue
# q = multiprocessing.Queue(maxsize=1) # 20240115 do I need a queue if I'm not reading anything back from it?
# processes = [] # 20240115
# processes.append(p)
