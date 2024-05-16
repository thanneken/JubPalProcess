#!/usr/bin/env python
from multiprocessing import Process, Queue, current_process, cpu_count
from skimage import io, img_as_float32, img_as_uint, img_as_ubyte, filters, exposure, color
from os import listdir, makedirs
from os.path import exists, join
from sklearn.decomposition import PCA, FastICA
from spectral import calc_stats, noise_from_diffs, mnf
from sys import argv
import yaml 
import numpy 
import pickle
import logging
import rawpy

savePreview = True

def main():
	getInstructions()
	logger.info('\n'+yaml.dump(instructions)+'\n')
	cacheFlattened()
	processStandardStats()
	logger.info('Concluded successfully')
	print("Concluded successfully")

def startLogging(logfile,loglevel):
	if not exists(logfile): 
		print("Creating logfile specified but does not already exist",logfile)
	global logger
	logger = logging.getLogger(__name__) 
	logLevelObject = eval('logging.'+loglevel)
	logging.basicConfig(
		filename=logfile, 
		format='%(asctime)s %(levelname)s %(message)s', 
		datefmt='%Y%m%d %H:%M:%S', 
		level = logLevelObject 
	) 
	print('Follow %s for %s progress, edit options.yaml to change log file path or log level'%(logfile,loglevel))

def getInstructions():
	global instructions
	for argument in argv:
		if 'instructions.yaml' in argument:
			instructionsPath = argument
			print('Found instructions.yaml as specified on the command line')
	with open(instructionsPath,'r') as unparsedyaml:
		instructions = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
	instructions['basepath'] = '/'.join(instructionsPath.split('/')[:-2])+'/'
	instructions['target'] = instructionsPath.split('/')[-2]
	if not instructions['settings']['logfile'].startswith('/'):
		instructions['settings']['logfile'] = instructions['basepath']+instructions['target']+'/'+instructions['settings']['logfile']
	startLogging(instructions['settings']['logfile'],instructions['settings']['loglevel'])
	if not instructions['settings']['cachepath'].startswith('/'):
		instructions['settings']['cachepath'] = instructions['basepath']+instructions['target']+'/'+instructions['settings']['cachepath']
	logger.info('Read instructions from %s'%(instructionsPath))
	logger.info('Using basepath %s'%(instructions['basepath']))
	logger.info('Using target %s'%(instructions['target']))
	logger.info('Using cache %s'%(instructions['settings']['cachepath']))
	instructions['imagesets'] = instructions['transform']['imagesets']
	instructions['options']['n_components'] = instructions['options']['n_components'][0]
	instructions['options']['skipuvbp'] = instructions['options']['skipuvbp'][0] 
	del instructions['transform']

def processStandardStats():
	logger.debug('%s Reading stack as a global object and keeping in memory for all transformations'%(current_process().name))
	global stack
	stack = readStack()
	processList = instructions['standardstats']
	standardTaskQueue = Queue()  
	standardDoneQueue = Queue()
	for processJob in processList:
		standardTaskQueue.put(processJob)
	countTransformProcesses = 2 # countTransformProcesses = len(processList) … keep this small and let histogram adjustment use max/2
	logger.info("%s Spawning %s processes for transformations based on standard stats"%(current_process().name,countTransformProcesses))
	for i in range(countTransformProcesses):
		Process(target=processStandardStatsThread,args=(standardTaskQueue,standardDoneQueue)).start()
	for i in range(len(processList)):
		logger.info(standardDoneQueue.get())
	for i in range(countTransformProcesses):
		standardTaskQueue.put('STOP')

def processStandardStatsThread(standardTaskQueue,standardDoneQueue):
	for task in iter(standardTaskQueue.get,'STOP'):
		standardTarget = '_'.join(instructions['target'].split('_')[0:2])+'_'+task.split('_')[0]
		picklePath = instructions['basepath']+standardTarget+'/stats/'+'_'.join(instructions['target'].split('_')[0:2])+'_'+task+'.pickle'
		logger.info("%s Working on stats from %s"%(current_process().name,picklePath))
		if 'mnf' in task:
			transform = processMnf(picklePath)
		elif 'ica' in task:
			transform = processFica(picklePath)
		elif 'pca' in task:
			transform = processPca(picklePath)
		histogramList = []
		for component in range(transform.shape[0]):
			layer = transform[component,:,:]
			for histogram in instructions['output']['histograms']:
				histogramList.append(['histogram',(layer,component,histogram,task)]) 
		logger.info('%s %s histogram adjustments ready to queue'%(current_process().name,len(histogramList)))
		histogramTaskQueue = Queue()
		histogramDoneQueue = Queue()
		for histogramJob in histogramList:
				histogramTaskQueue.put(histogramJob)
		histogramThreads = round(cpu_count()/2-1) # = 7 on palimpsest, 63 on incline
		logger.info('%s spawning %s sub processes for histogram adjustments'%(current_process().name,histogramThreads))
		for i in range(histogramThreads):
			Process(target=processHistogramsThread,args=(histogramTaskQueue,histogramDoneQueue)).start()
		for i in range(len(histogramList)):
			logger.debug(histogramDoneQueue.get())
		for i in range(histogramThreads):
			histogramTaskQueue.put('STOP')
		standardDoneQueue.put('%s completed all file formats for %s'%(current_process().name,task))

def readStack():
	cube = []
	for imageset in instructions['imagesets']:
		for file in listdir(instructions['basepath']+instructions['target']+'/'+imageset):
			if ((instructions['options']['skipuvbp'] == True) and (("UVB_" in file) or ("UVP_" in file))):
				logger.info('Skipping %s due to registration issues'%s(file))
				continue
			file = instructions['basepath']+instructions['target']+'/'+imageset+'/'+file
			if needsFlattening(file):
				img = io.imread(cacheEquivalent(file,'flattened'))
			else:
				img = io.imread(file)
			cube.append(img)
	return numpy.array(cube)

def openImageFile(path):
	if path.endswith('.dng'):
		with rawpy.imread(path) as raw:
			return raw.raw_image.copy() 
	else:
			return io.imread(path)

def needsFlattening(path):
	if 'NoGamma' in path:
		return False
	elif '.dng' in path:
		return True
	elif 'Unflat' in path:
		return True
	elif 'unflat' in path:
		return True
	elif 'Flat' in path:
		return False
	elif 'flat' in path:
		return False
	elif '.tif' in path:
		return False
	else:
		return False

def cacheEquivalent(inPath,derivative):
	if 'sigma' in derivative:
		derivative = 'denoise/'+derivative
	return instructions['settings']['cachepath']+derivative+'/'+inPath.split('/')[-1][:-4]+'.tif'

def cacheFlattened():
	inPaths = []
	unflatPaths = []
	for imageset in instructions['imagesets']:
		for file in listdir(instructions['basepath']+instructions['target']+'/'+imageset):
			if ((instructions['options']['skipuvbp'] == True) and (("UVB_" in file) or ("UVP_" in file))):
				logger.info('Skipping %s due to registration issues'%s(file))
				continue
			inPaths.append(instructions['basepath']+instructions['target']+'/'+imageset+'/'+file)
	for inPath in inPaths:
		if needsFlattening(inPath) and not exists(cacheEquivalent(inPath,'flattened')): 
			unflatPaths.append(inPath)
	if len(unflatPaths) > 0:
		if not exists(instructions['settings']['cachepath']+'flattened/'): 
			logger.info('Creating directory to cache flattened images, should only be necessary the first run on a machine')
			makedirs(instructions['settings']['cachepath']+'flattened/',mode=0o755,exist_ok=True)
		taskQueue = Queue()  
		doneQueue = Queue()
		threadsMax = cpu_count()
		for unflatPath in unflatPaths:
			taskQueue.put(unflatPath)
		for i in range(threadsMax):
			Process(target=cacheFlattenedThread,args=(taskQueue,doneQueue)).start()
		for i in range(len(unflatPaths)):
			logger.info(doneQueue.get())
		for i in range(threadsMax):
			taskQueue.put('STOP')

def cacheFlattenedThread(taskQueue,doneQueue):
	for unflatPath in iter(taskQueue.get,'STOP'):
		img = openImageFile(unflatPath)
		img = img_as_float32(img) 
		flatImg = findFlat(unflatPath) 
		img = flatten(img,flatImg)
		img = rotate(img) 
		img = img_as_float32(img)
		io.imsave(cacheEquivalent(unflatPath,'flattened'),img,check_contrast=False)
		logger.info(current_process().name+cacheEquivalent(unflatPath,'flattened')+' saved to cache')
		if savePreview:
			img = exposure.rescale_intensity(img)
			img = exposure.adjust_gamma(img,1/2.2)
			img = img_as_ubyte(img)
			previewPath = instructions['basepath']+instructions['target']+'/Preview/'+unflatPath.split('/')[-1][:-4]+'.jpg'
			makedirs(instructions['basepath']+instructions['target']+'/Preview/',mode=0o755,exist_ok=True)
			io.imsave(previewPath,img,check_contrast=False)
			logger.info('%s saved preview file %s'%(current_process().name,previewPath))
		doneQueue.put('%s cached flattened %s'%(current_process().name,unflatPath))
	
def findFlat(path):
	if not 'flats' in instructions:
		logger.error("It is necessary to specify relative path to flats in YAML metadata")
		exit("It is necessary to specify relative path to flats in YAML metadata")
	try: 
		exif = pyexifinfo.get_json(path)
		exifflat = exif[0]["IPTC:Keywords"][11] 
		if exifflat.endswith('.dn'):
			exifflat = exifflat+'g' 
		logger.info(current_process().name+'Found path to flatfile in MegaVision DNG header')
		match = exifflat
	except: 
		try: 
			for flatFile in listdir(instructions['basepath']+instructions['flats']):
				if flatFile[-11:] == path[-11:]:
					logger.info(current_process().name+'Found 11 character match: '+path+' ~ '+flatFile)
					match = flatFile
			assert match
		except:
			try: 
				for flatFile in listdir(instructions['basepath']+instructions['flats']):
					if flatFile[-7:] == path[-7:]:
						logger.info(current_process().name+'Found 7 character match: '+path+' ~ '+flatFile)
						match = flatFile
				assert match
			except:
				logger.error("Unable to find flatfile for this capture")
				exit("Unable find flatfile for this capture")
	return openImageFile(instructions['basepath']+instructions['flats']+match)

def flatten(img,flatImg):
	if 'blurImage' in instructions and instructions['blurImage'] == "median3":
		logger.info("Blurring image with 3x3 median")
		img = filters.median(img) # default is 3x3
	if 'blurFlat' in instructions and instructions['blurFlat'] > 0: 
		logger.info("Blurring flat with sigma "+str(instructions['blurFlat']))
		flatImg = filters.gaussian(flatImg,sigma=instructions['blurFlat'])
	return numpy.divide(img*numpy.average(flatImg),flatImg,out=numpy.zeros_like(img*numpy.average(flatImg)),where=flatImg!=0)

def rotate(img): 
	if instructions['rotation'] == 90:
		img = numpy.rot90(img,k=3)
	elif instructions['rotation'] == 180:
		img = numpy.rot90(img,k=2)
	elif instructions['rotation'] == 270:
		img = numpy.rot90(img,k=1)
	else:
		logger.info("No rotation identified")
	return img
	
def processPca(picklePath): 
	logger.info('%s Starting PCA'%(current_process().name))
	nlayers,fullh,fullw = stack.shape
	cube = stack.reshape((nlayers,fullw*fullh))
	cube = cube.transpose()
	if instructions['options']['n_components'] == 'max':
		n_components = nlayers
	else:
		n_components = instructions['options']['n_components']
	pca = pickle.load(open(picklePath,"rb"))
	cube = pca.transform(cube)
	cube = cube.transpose()
	cube = cube.reshape(n_components,fullh,fullw)
	logger.info('%s Finished PCA'%(current_process().name))
	return cube

def processFica(picklePath):
	logger.info('%s Starting ICA'%(current_process().name))
	nlayers,fullh,fullw = stack.shape
	n_components = nlayers # always use max even if user selects a lower number of components
	cube = stack.reshape((nlayers,fullw*fullh))
	cube = cube.transpose()
	fica = pickle.load(open(picklePath,"rb"))
	cube = fica.transform(cube)
	cube = img_as_float32(cube)
	cube = cube.transpose()
	cube = cube.reshape(n_components,fullh,fullw)
	logger.info('%s Finished ICA'%(current_process().name))
	return cube

def processMnf(picklePath):
	logger.info('%s Starting MNF'%(current_process().name))
	nlayers,fullh,fullw = stack.shape
	cube = stack.transpose()
	if instructions['options']['n_components'] == 'max':
		n_components = nlayers
	else:
		n_components = instructions['options']['n_components']
	mnfr = pickle.load(open(picklePath,"rb"))
	cube = mnfr.reduce(cube,num=n_components)
	cube = img_as_float32(cube)
	cube = cube.transpose()
	logger.info('%s Finished MNF'%(current_process().name))
	return cube

def processHistogramsThread(histogramTaskQueue,histogramDoneQueue):
	for task, args in iter(histogramTaskQueue.get,'STOP'):
		if task == 'histogram':
			img,component,histogram,standardstats = args
			component = str(f"{component:02d}")
			if histogram == 'equalize':
				img = exposure.equalize_hist(img)
			elif histogram == 'rescale':
				img = exposure.rescale_intensity(img)
			elif histogram == 'adaptive':
				img = exposure.rescale_intensity(img)
				img = exposure.equalize_adapthist(img,clip_limit=0.03)
			elif histogram == 'none':
				logger.warning('No histogram adjustment is a bad idea')
			else:
				logger.warning('Histogram adjustment not recognized %s'%(histogram))
			directoryPath = '%s%s/Standard/%s/'%(instructions['basepath'],instructions['target'],standardstats) 
			filename = '%s_standard_%s_%s_c%s'%(instructions['target'],standardstats,histogram,component) 
			for fileFormat in instructions['output']['fileformats']:
				finalDirectoryPath = directoryPath # finalDirectoryPath = directoryPath+fileFormat+'/'
				makedirs(finalDirectoryPath,mode=0o755,exist_ok=True)
				finalFilePath = finalDirectoryPath+filename+'.'+fileFormat
				if fileFormat == 'tif':
					img32 = img_as_float32(img)
					io.imsave(finalFilePath,img32,check_contrast=False)
				elif fileFormat == 'png':
					img16 = img_as_uint(img)
					io.imsave(finalFilePath,img16,check_contrast=False)
				elif fileFormat == 'jpg':
					img8 = img_as_ubyte(img)
					io.imsave(finalFilePath,img8,check_contrast=False)
			histogramDoneQueue.put('%s completed all file formats for %s'%(current_process().name,filename))

if __name__ == '__main__':
	main()

