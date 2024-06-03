#!/usr/bin/env python
from multiprocessing import Process, Queue, current_process 
from skimage import io, img_as_float32, img_as_uint, img_as_ubyte, filters, exposure, color
from os import listdir, makedirs
from os.path import exists, join
from sklearn.decomposition import PCA, FastICA
from spectral import calc_stats, noise_from_diffs, mnf
from sys import argv
from math import floor
from psutil import virtual_memory, cpu_count, cpu_times, cpu_percent
from time import sleep
import logging
import yaml 
import inquirer
import numpy 
import rawpy
try:
	import pyexifinfo
except:
	print('Unable to load pyexifinfo, only useful if intend to read flatpath from dng metadata (MegaVision)')
try:
	import pickle
except: 
	print('You will want to install the pickle if you want to pickle or unpickle noise and stats from other sequences')
else:
	saveStats = True

def main():
	getInstructions()
	logger.info('\n'+yaml.dump(instructions)+'\n')
	if 'logresources' in instructions['settings'] and instructions['settings']['logresources'] > 0:
		logResourcesProcess = Process(target=logResourcesFunction,args=[instructions['settings']['logresources']])
		logResourcesProcess.start()
	cacheFlattened()
	estimateResources()
	if any(x in instructions['options']['methods'] for x in ['kpca','pca','mnf','fica']):
		cacheBlurDivide()
	processMethods()
	if instructions['settings']['logresources'] > 0:
		logResourcesProcess.terminate() # see also join, terminate, kill, and close
	logger.info('Concluded successfully')
	exit("Concluded successfully")

def logResourcesFunction(sleepInterval):
	while True:
		memAvail = round(virtual_memory()[1]/2**30,1) 
		logger.info('Resources %s%% CPU used, %s%% RAM used, %s GB RAM available'%(cpu_percent(interval=1),virtual_memory()[2],memAvail))
		sleep(sleepInterval)

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

def findOptionsFile():
	for argument in argv:
		if 'options.yaml' in argument:
			if exists(argument):
				print('Found options.yaml as specified on the command line')
				return argument
			else:
				print('Found options.yaml in command-line arguments, but the file does not exist')
				continue
	if exists('options.yaml'):
		print('Found options.yaml in present working directory')
		return 'options.yaml'
	elif exists('git/JubPalProcess/options.yaml'):
		print('Found options.yaml in git/JubPalProcess/')
		return 'git/JubPalProcess/options.yaml'
	else:
	 	exit('Cannot continue without specifying path to options.yaml')

def readInstructionsFile(instructionsPath):
	global instructions
	with open(instructionsPath,'r') as unparsedyaml:
		instructions = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
	instructions['basepath'] = '/'.join(instructionsPath.split('/')[:-2])+'/'
	instructions['target'] = instructionsPath.split('/')[-2]
	if not instructions['settings']['logfile'].startswith('/'):
		instructions['settings']['logfile'] = instructions['basepath']+instructions['target']+'/'+instructions['settings']['logfile']
	if not instructions['settings']['cachepath'].startswith('/'):
		instructions['settings']['cachepath'] = instructions['basepath']+instructions['target']+'/'+instructions['settings']['cachepath']
	startLogging(instructions['settings']['logfile'],instructions['settings']['loglevel'])
	logger.info('Read instructions from %s'%(instructionsPath))
	logger.info('Using basepath %s'%(instructions['basepath']))
	logger.info('Using target %s'%(instructions['target']))
	logger.info('Using cache %s'%(instructions['settings']['cachepath']))
	instructions['imagesets'] = instructions['transform']['imagesets']
	instructions['roi'] = instructions['transform']['rois'][list(instructions['transform']['rois'].keys())[0]] 
	instructions['noisesample'] = instructions['transform']['noisesamples'][list(instructions['transform']['noisesamples'].keys())[0]] 
	instructions['options']['n_components'] = instructions['options']['n_components'][0]
	instructions['options']['skipuvbp'] = instructions['options']['skipuvbp'][0] 
	del instructions['transform']

def getInstructions():
	global instructions
	for argument in argv:
		if 'instructions.yaml' in argument:
			if exists(argument):
				print('Found instructions.yaml as specified on the command line')
				readInstructionsFile(argument)
				return
			else:
				print('Found instructions.yaml in command-line arguments, but the file does not exist')
				continue
	optionsfile = findOptionsFile()
	with open(optionsfile,'r') as unparsedyaml:
		instructions = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
	startLogging(instructions['settings']['logfile'],instructions['settings']['loglevel'])
	del instructions['document'] , instructions['tip']
	logger.info('Read instructions from %s'%(optionsfile))
	if 'noninteractive' in argv:
		instructions['options']['interactive'] = False
		logger.info('Using non-interactive mode as instructed by commandline argument')
	elif len(instructions['options']['interactive']) > 1:
		questions = [inquirer.List('interactive','Proceed with interactive choices?',choices=instructions['options']['interactive'])]
		selections = inquirer.prompt(questions)
		instructions['options']['interactive'] = selections['interactive']
		logger.info('User selected interactive mode %s'%(instructions['options']['interactive']))
	else:
		instructions['options']['interactive']	= instructions["options"]["interactive"][0]
		logger.info('Interactive mode is %s because that is the only uncommented option in options file'%(instructions['options']['interactive']))
	if instructions['options']['interactive']:
		logger.info('Asking user to provide instructions')
		askUser()
	else:
		logger.info('Reading instructions based on default values in project file')
		readProjectDefaults()

def askUser():
	if len(instructions['basepaths']) > 1:
		questions = [inquirer.List('basepath','Select basepath for source data',choices=instructions['basepaths'])]
		selections = inquirer.prompt(questions)
		instructions['basepath'] = selections['basepath']
		logger.info('User selected basepath %s'%(instructions['basepath']))
	else:
		instructions['basepath'] = instructions['basepaths'][0]
		logger.info('Only uncommented basepath in options file is %s'%(instructions['basepath']))
	del instructions['basepaths']
	logger.info('Looking for projectFile in selected basepath')
	projectFile = instructions['basepath']+instructions['basepath'].split('/')[-2]+'.yaml'
	if exists(projectFile):
		with open(projectFile,'r') as unparsedyaml:
			targets = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
		logger.info('Read project file %s'%(projectFile))
	else:
	 	exit('Unable to find '+projectFile)
	targetChoices = list(targets.keys())
	targetChoices.remove('default')
	if len(targetChoices) > 1:
		questions = [inquirer.List('target','Select target',choices=targetChoices)]
		selections = inquirer.prompt(questions)
		target = selections['target']
		logger.info('User selected target %s'%(target))
	else:
		target = targetChoices[0]
		logger.info('Only offered target is %s'%(target))
	instructions['target'] = target
	instructions.update(targets['default'])
	instructions.update(targets[target])
	if 'white' in instructions:
		instructions['white'].update(targets['default']['white'])
		instructions['white'].update(targets[target]['white'])
	if len(instructions['options']['methods']) > 1:
		questions = [ inquirer.Checkbox('methods','Select Process',choices=instructions['options']['methods']) ]
		methods = []
		while len(methods) < 1:
			selections = inquirer.prompt(questions)
			methods = selections['methods']
		instructions['options']['methods'] = selections['methods']
		logger.info('User selected methods are %s'%(instructions['options']['methods']))
	else:
		logger.info('Only method uncommented in options file is %s'%(instructions['options']['methods']))
	if any(x in instructions['options']['methods'] for x in ['kpca','pca','mnf','fica']):
		askUserTransformationOptions()
	else:
		logger.info('Cleaning up instructions for the sake the log')
		del instructions['options']['n_components'],instructions['noisesamples'],instructions['rois'],instructions['output']['histograms']
	if len(instructions['output']['fileformats']) > 1:
		questions = [ inquirer.Checkbox('fileformats','Select file format(s) to output',choices=instructions['output']['fileformats']) ]
		fileformats = []
		while len(fileformats) < 1:
			selections = inquirer.prompt(questions)
			fileformats = selections['fileformats']
		instructions['output']['fileformats'] = fileformats
	else:
	 	logger.info('Only fileformat uncommented in options file is %s'%(instructions['output']['fileformats'][0]))

def askUserTransformationOptions():
	if len(instructions['imagesets']) > 1:
		questions = [inquirer.Checkbox('imagesets','Select one or more image sets',choices=instructions['imagesets'])]
		imagesets = []
		while len(imagesets) < 1:
			selections = inquirer.prompt(questions)
			imagesets = selections['imagesets']
		instructions['imagesets'] = imagesets
		logger.info('User selected imagesets %s'%(instructions['imagesets']))
	else:
		logger.info('Only option available for imagesets is %s'%(instructions['imagesets']))
	if len(instructions['options']['sigmas']) > 1:
		questions = [ inquirer.Checkbox('sigmas','Sigma for RLE blur and divide?',choices=instructions['options']['sigmas']) ]
		sigmas = []
		while len(sigmas) < 1:
			selections = inquirer.prompt(questions)
			sigmas = selections['sigmas']
		instructions['options']['sigmas'] = sigmas 
		logger.info('User selected sigma options %s'%(instructions['options']['sigmas']))
	else:
		logger.info('Only sigma value uncommented in options file is %s'%(instructions['options']['sigmas']))
	if len(instructions['options']['skipuvbp']) > 1:
		questions = [ inquirer.List('skipuvbp','Skip files with UVB_ or UVP_ in filename?',choices=instructions['options']['skipuvbp']) ]
		selections = inquirer.prompt(questions)
		instructions['options']['skipuvbp'] = selections['skipuvbp']
		logger.info('User selected Skip files with UVB_ or UVP_ in filename %s'%(instructions['options']['skipuvbp']))
	else:
		instructions['options']['skipuvbp'] = instructions['options']['skipuvbp'][0] 
		logger.info('Only option available for skip files with UVB_ or UVP_ in filename is %s'%(instructions['options']['skipuvbp']))
	
	if len(instructions['rois']) > 1:
		questions = [inquirer.List('roi','Select Region Of Interest (ROI)',choices=instructions['rois'].keys())]
		selections = inquirer.prompt(questions)
		instructions['roi'] = selections['roi']
	else:
		instructions['roi'] = instructions['rois'][list(instructions['rois'].keys())[0]] 
	if 'mnf' in instructions['options']['methods']:
		if len(instructions['noisesamples'].keys()) > 1:
			questions = [inquirer.List('noisesample','Select Noise Region',choices=instructions['noisesamples'].keys())]
			selections = inquirer.prompt(questions)
			instructions['noisesample'] = selections['noisesample']
		else:
			instructions['noisesample'] = instructions['noisesamples'][list(instructions['noisesamples'].keys())[0]] 
	del instructions['rois'],instructions['noisesamples']
	if len(instructions['options']['n_components']) > 1:
		questions = [ inquirer.List('n_components','How many components to generate for PCA and MNF? (ICA is always max)',choices=instructions['options']['n_components']) ]
		selections = inquirer.prompt(questions)
		instructions['options']['n_components'] = selections['n_components']
		logger.info('User selected number of components %s'%(instructions['options']['n_components']))
	else:
		instructions['options']['n_components'] = instructions['options']['n_components'][0] 
		logger.info('Only option available for number of components is %s'%(instructions['options']['n_components']))
	if len(instructions['output']['histograms']) > 1:
		questions = [ inquirer.Checkbox('histograms','Select histogram adjustment(s) for final product',choices=instructions['output']['histograms']) ]
		histograms = []
		while len(histograms) < 1:
			selections = inquirer.prompt(questions)
			histograms = selections['histograms']
		instructions['output']['histograms'] = histograms
	else:
		instructions['output']['histograms'] = instructions['output']['histograms'][0]
	if 'multilayer' in instructions['output']:
		logger.warning('The option to output a stack/cube rather than a directory of images has been deprecated')

def readProjectDefaults():
	instructions['basepath'] = instructions['basepaths'][0]
	del instructions['basepaths']
	logger.info('Looking for projectFile in selected basepath')
	projectFile = instructions['basepath']+instructions['basepath'].split('/')[-2]+'.yaml'
	if exists(projectFile):
		with open(projectFile,'r') as unparsedyaml:
			targets = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
		logger.info('Read project file %s'%(projectFile))
	else:
	 	exit('Unable to find '+projectFile)
	target = nextNeededTarget(targets.keys())
	if target == None:
		logger.info("No projects defined in the first named basepath lack a Transform directory")
		exit("No projects defined in the first named basepath lack a Transform directory")
	instructions['target'] = target
	instructions.update(targets['default'])
	instructions.update(targets[target])
	if 'white' in instructions:
		instructions['white'].update(targets['default']['white'])
		instructions['white'].update(targets[target]['white'])
	if any(x in instructions['options']['methods'] for x in ['kpca','pca','mnf','fica']):
		instructions['options']['skipuvbp'] = instructions['options']['skipuvbp'][0] 
		instructions['options']['n_components'] = instructions['options']['n_components'][0] 
		instructions['roi'] = instructions['rois'][list(instructions['rois'].keys())[0]]
		instructions['noisesample'] = instructions['noisesamples'][list(instructions['noisesamples'].keys())[0]] 
		del instructions['rois'], instructions['noisesamples']
	else:
		del instructions['imagesets'],instructions['options']['sigmas'],instructions['options']['skipuvbp']
		del instructions['options']['n_components'],instructions['noisesamples'],instructions['rois'],instructions['output']['histograms']

def nextNeededTarget(targets):
	matches = (f for f in targets if exists(instructions['basepath']+f) and not exists(instructions['basepath']+f+'/Transform'))
	return next(matches,None)

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
	global countInput
	countInput = len(inPaths)
	for inPath in inPaths:
		if needsFlattening(inPath) and not exists(cacheEquivalent(inPath,'flattened')): 
			unflatPaths.append(inPath)
	if len(unflatPaths) > 0:
		if not exists(instructions['settings']['cachepath']+'flattened/'): 
			logger.info('Creating directory to cache flattened images, should only be necessary the first run on a machine')
			makedirs(instructions['settings']['cachepath']+'flattened/',mode=0o755,exist_ok=True)
		taskQueue = Queue()  
		doneQueue = Queue()
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
		logger.info(cacheEquivalent(unflatPath,'flattened')+' saved to cache')
		doneQueue.put('%s cached flattened %s'%(current_process().name,unflatPath))
		
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

def findFlat(path):
	if not 'flats' in instructions:
		logger.error("It is necessary to specify relative path to flats in YAML metadata")
		exit("It is necessary to specify relative path to flats in YAML metadata")
	try: 
		exif = pyexifinfo.get_json(path)
		exifflat = exif[0]["IPTC:Keywords"][11] 
		if exifflat.endswith('.dn'):
			exifflat = exifflat+'g' 
		assert(exists(exifflat))
		logger.info('Found path to flatfile in MegaVision DNG header')
		match = exifflat
	except: 
		try: 
			for flatFile in listdir(instructions['basepath']+instructions['flats']):
				if flatFile[-11:] == path[-11:]:
					logger.info('Found 11 character match: '+path+' ~ '+flatFile)
					match = flatFile
			assert match
		except:
			try: 
				for flatFile in listdir(instructions['basepath']+instructions['flats']):
					if flatFile[-7:] == path[-7:]:
						logger.info('Found 7 character match: '+path+' ~ '+flatFile)
						match = flatFile
				assert match
			except:
				logger.error("Unable to find flatfile for %s"%(path))
				exit("Unable find flatfile for %s"%(path))
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
	if 'rotation' in instructions and instructions['rotation'] == 90:
		img = numpy.rot90(img,k=3)
	elif 'rotation' in instructions and  instructions['rotation'] == 180:
		img = numpy.rot90(img,k=2)
	elif 'rotation' in instructions and instructions['rotation'] == 270:
		img = numpy.rot90(img,k=1)
	else:
		logger.info("No rotation identified")
	return img
	
def estimateResources():
	memAvail = round(virtual_memory()[1]/2**30,1) 
	logger.info('%s threads and %s GB RAM available'%(threadsMax,memAvail))
	sampleDirectory = instructions['basepath']+instructions['target']+'/'+instructions['imagesets'][0]+'/'
	sampleFile = listdir(sampleDirectory)[0]
	img = openImageFile(join(sampleDirectory,sampleFile))
	countPixels = img.shape[0] * img.shape[1]
	logger.info('32 bits per pixel, %s pixels per layer, %s layers occupies %s GB RAM per cube'%(countPixels,countInput,round(countPixels*countInput*4/2**30,1)))
	countMethods = len(instructions['options']['methods'])
	countSigmas = len(instructions['options']['sigmas'])
	if 'color' in instructions['options']['methods']:
		countTransforms = (countMethods-1)*countSigmas+1
	else:
		countTransforms = countMethods * countSigmas
	logger.info('%s transformations for %s methods with %s blur divide sigmas plus color'%(countTransforms,countMethods,countSigmas))
	threadsTransformRound = round(virtual_memory()[1] / (countPixels*countInput*4*4),2)
	global threadsTransformFloor
	threadsTransformFloor = floor(threadsTransformRound)
	logger.info('%s (%s) parallel transformations possible assuming 4x input cube size for each transformation'%(threadsTransformRound,threadsTransformFloor))
	if threadsTransformFloor == 0:
		logger.warning('Estimates suggest not enough RAM to complete a transformation, overriding to give it a try')
		threadsTransformFloor = 1

def cacheBlurDivide():
	inPaths = []
	for imageset in instructions['imagesets']:
		for file in listdir(instructions['basepath']+instructions['target']+'/'+imageset):
			if ((instructions['options']['skipuvbp'] == True) and (("UVB_" in file) or ("UVP_" in file))):
				logger.info('Skipping %s due to registration issues'%s(file))
				continue
			inPaths.append(instructions['basepath']+instructions['target']+'/'+imageset+'/'+file)
	bdCache = []
	for sigma in instructions['options']['sigmas']:
		if sigma == 0:
			continue
		for inPath in inPaths:
			if not exists(cacheEquivalent(inPath,'sigma'+str(sigma))): 
				bdCache.append([inPath,sigma])
		if not exists(instructions['settings']['cachepath']+'denoise/sigma'+str(sigma)): 
			logger.info('Creating directory to cache blurred and divided images, should only be necessary the first run on a machine')
			makedirs(instructions['settings']['cachepath']+'denoise/sigma'+str(sigma),mode=0o755,exist_ok=True)
	if len(bdCache) > 0:
		taskQueue = Queue()  
		doneQueue = Queue()
		for bd in bdCache:
			taskQueue.put(bd)
		for i in range(threadsMax):
			Process(target=cacheBlurDivideThread,args=(taskQueue,doneQueue)).start()
		for i in range(len(bdCache)):
			logger.info(doneQueue.get())
		for i in range(threadsMax):
			taskQueue.put('STOP')

def cacheBlurDivideThread(taskQueue,doneQueue):
	for inPath, sigma in iter(taskQueue.get,'STOP'):
		bdPath = cacheEquivalent(inPath,'sigma'+str(sigma)) 
		if exists(cacheEquivalent(inPath,'flattened')):
			img = openImageFile(cacheEquivalent(inPath,'flattened'))
		else:
			img = openImageFile(inPath)
		if not img.dtype == "float32":
			img = img_as_float32(img)
		numerator = filters.median(img) # default is 3x3, same as RLE suggested
		denominator = filters.gaussian(img,sigma=sigma)
		img = numpy.divide(numerator,denominator,out=numpy.zeros_like(numerator),where=denominator!=0)
		io.imsave(bdPath,img,check_contrast=False)
		doneQueue.put('%s cached denoise %s'%(current_process().name,bdPath))

def processMethods():
	processList = []
	for method in instructions['options']['methods']:
		if method == 'color':
			processList.append(['transform',(method,0)])
		else:
			for sigma in instructions['options']['sigmas']:
				processList.append(['transform',(method,sigma)])
	transformTaskQueue = Queue()  
	transformDoneQueue = Queue()
	for processJob in processList:
		transformTaskQueue.put(processJob)
	for i in range(threadsTransformFloor):
		Process(target=processMethodsThread,args=(transformTaskQueue,transformDoneQueue)).start()
	for i in range(len(processList)):
		logger.info(transformDoneQueue.get())
	for i in range(threadsTransformFloor):
		transformTaskQueue.put('STOP')

def processMethodsThread(transformTaskQueue,transformDoneQueue):
	for task, args in iter(transformTaskQueue.get,'STOP'):
		if task == 'transform':
			method = args[0]
			sigma = args[1]
			if method == 'color':
				startColor()
				transformDoneQueue.put('%s completed color processing'%(current_process().name))
				return
			elif method == 'pca':
				stack = processPca(sigma)
			elif method == 'mnf':
				stack = processMnf(sigma)
			elif method == 'fica':
				stack = processFica(sigma)
			histogramList = []
			for component in range(stack.shape[0]):
				layer = stack[component,:,:]
				for histogram in instructions['output']['histograms']:
					histogramList.append(['histogram',(layer,sigma,method,component,histogram)]) 
			logger.info('%s %s histogram adjustments ready to queue'%(current_process().name,len(histogramList)))
			histogramTaskQueue = Queue()
			histogramDoneQueue = Queue()
			for histogramJob in histogramList:
			    histogramTaskQueue.put(histogramJob)
			# if len(histogramList) < threadsMax/4: histogramThreads = len(histogramList)
			if threadsTransformFloor == 1:
			    histogramThreads = threadsMax
			elif threadsTransformFloor > 1:
			    histogramThreads = threadsTransformFloor # round((threadsMax-threadsTransformFloor)/threadsTransformFloor,None)-1
			else:
			    histogramThreads = threadsMax 
			logger.info('%s spawning %s sub processes for histogram adjustments'%(current_process().name,histogramThreads))
			for i in range(histogramThreads):
				Process(target=processHistogramsThread,args=(histogramTaskQueue,histogramDoneQueue)).start()
			for i in range(len(histogramList)):
				logger.debug(histogramDoneQueue.get())
			for i in range(histogramThreads):
				histogramTaskQueue.put('STOP')
			transformDoneQueue.put('%s completed all file formats for %s %s'%(current_process().name,method,sigma))

def processHistogramsThread(histogramTaskQueue,histogramDoneQueue):
	for task, args in iter(histogramTaskQueue.get,'STOP'):
		if task == 'histogram':
			img,sigma,transform,component,histogram = args
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
			roix,roiy,roiw,roih = instructions['roi']['x'], instructions['roi']['y'], instructions['roi']['w'], instructions['roi']['h']
			roistring = "x"+str(roix)+"y"+str(roiy)+"w"+str(roiw)+"h"+str(roih) 
			if transform == 'mnf':
				noisex,noisey,noisew,noiseh = instructions['noisesample']['x'], instructions['noisesample']['y'], instructions['noisesample']['w'], instructions['noisesample']['h']
				noisestring = "nx"+str(noisex)+"y"+str(noisey)+"w"+str(noisew)+"h"+str(noiseh) 
			else:
				noisestring = ''
			directoryPath = '%s%s/Transform/r%sbd%s/%s_%s%s/%s/'%(instructions['basepath'],instructions['target'],countInput,sigma,transform,roistring,noisestring,histogram) 
			filename = '%s_r%s_bd%s_%s_%s%s_%s_c%s'%(instructions['target'],countInput,sigma,transform,roistring,noisestring,histogram,component) 
			for fileFormat in instructions['output']['fileformats']:
				finalDirectoryPath = directoryPath+fileFormat+'/'
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

def readStack(sigma):
	stack = []
	for imageset in instructions['imagesets']:
		for file in listdir(instructions['basepath']+instructions['target']+'/'+imageset):
			if ((instructions['options']['skipuvbp'] == True) and (("UVB_" in file) or ("UVP_" in file))):
				logger.info('Skipping %s due to registration issues'%s(file))
				continue
			file = instructions['basepath']+instructions['target']+'/'+imageset+'/'+file
			if sigma > 0:
				img = io.imread(cacheEquivalent(file,'sigma'+str(sigma)))
			elif needsFlattening(file):
				img = io.imread(cacheEquivalent(file,'flattened'))
			else:
				img = io.imread(file)
			stack.append(img)
	return numpy.array(stack)

def processPca(sigma): 
	memAvail = round(virtual_memory()[1]/2**30,1) 
	logger.info('%s PCA starting with %s GB available'%(current_process().name,memAvail))
	stack = readStack(sigma)
	nlayers,fullh,fullw = stack.shape
	roix,roiy,roiw,roih = instructions['roi']['x'], instructions['roi']['y'], instructions['roi']['w'], instructions['roi']['h']
	roi = stack[:,roiy:roiy+roih,roix:roix+roiw] 
	roi = roi.reshape((nlayers,roiw*roih))
	roi = roi.transpose()
	stack = stack.reshape((nlayers,fullw*fullh))
	stack = stack.transpose()
	if instructions['options']['n_components'] == 'max':
		n_components = nlayers
	else:
		n_components = instructions['options']['n_components']
	pca = PCA(n_components=n_components)
	pca.fit(roi)
	if saveStats:
		roistring = "x"+str(roix)+"y"+str(roiy)+"w"+str(roiw)+"h"+str(roih) 
		picklePath = instructions['basepath']+instructions['target']+'/stats/'+instructions['target']+'_r'+str(countInput)+'_bd'+str(sigma)+'_pca_'+roistring+'.pickle'
		logger.info("%s Saving PCA stats to %s"%(current_process().name,picklePath))
		makedirs(instructions['basepath']+instructions['target']+'/stats',mode=0o755,exist_ok=True)
		pickle.dump(pca,open(picklePath,"wb"))
	stack = pca.transform(stack)
	stack = stack.transpose()
	stack = stack.reshape(n_components,fullh,fullw)
	logger.info('%s PCA finished'%(current_process().name))
	return stack
	
def processFica(sigma):
	memAvail = round(virtual_memory()[1]/2**30,1) 
	if 'fica_max_iter' in instructions['settings']:
		max_iter = instructions['settings']['fica_max_iter']
	else:
		max_iter = 100
	if 'fica_tol' in instructions['settings']:
		tol = instructions['settings']['fica_tol']
	else:
		tol = 0.0001 
	logger.info('%s ICA starting with %s GB available, %s max iterations, %s tolerance'%(current_process().name,memAvail,max_iter,tol))
	stack = readStack(sigma)
	nlayers,fullh,fullw = stack.shape
	roix,roiy,roiw,roih = instructions['roi']['x'], instructions['roi']['y'], instructions['roi']['w'], instructions['roi']['h']
	roi = stack[:,roiy:roiy+roih,roix:roix+roiw] 
	roi = roi.reshape((nlayers,roiw*roih))
	roi = roi.transpose()
	stack = stack.reshape((nlayers,fullw*fullh))
	stack = stack.transpose()
	n_components = nlayers # always use max even if user selects a lower number of components
	fica = FastICA(n_components=n_components,max_iter=max_iter,tol=tol)
	fica.fit(roi)
	if saveStats:
		roistring = "x"+str(roix)+"y"+str(roiy)+"w"+str(roiw)+"h"+str(roih) 
		picklePath = instructions['basepath']+instructions['target']+'/stats/'+instructions['target']+'_r'+str(countInput)+'_bd'+str(sigma)+'_fica_'+roistring+'.pickle'
		logger.info("%s Saving ICA stats to %s"%(current_process().name,picklePath))
		makedirs(instructions['basepath']+instructions['target']+'/stats',mode=0o755,exist_ok=True)
		pickle.dump(fica,open(picklePath,"wb"))
	stack = fica.transform(stack)
	stack = img_as_float32(stack)
	stack = stack.transpose()
	stack = stack.reshape(n_components,fullh,fullw)
	logger.info('%s ICA finished'%(current_process().name))
	return stack

def processMnf(sigma):
	memAvail = round(virtual_memory()[1]/2**30,1) 
	logger.info('%s MNF starting with %s GB available'%(current_process().name,memAvail))
	stack = readStack(sigma)
	nlayers,fullh,fullw = stack.shape
	if instructions['options']['n_components'] == 'max':
		n_components = nlayers
	else:
		n_components = instructions['options']['n_components']
	roix,roiy,roiw,roih = instructions['roi']['x'], instructions['roi']['y'], instructions['roi']['w'], instructions['roi']['h']
	noisex,noisey,noisew,noiseh = instructions['noisesample']['x'], instructions['noisesample']['y'], instructions['noisesample']['w'], instructions['noisesample']['h']
	stack = stack.transpose()
	signal = calc_stats(stack[roix:roix+roiw,roiy:roiy+roih,:]) 
	noise = noise_from_diffs(stack[noisex:noisex+noisew,noisey:noisey+noiseh,:]) 
	mnfr = mnf(signal,noise)
	if saveStats:
		roistring = "x"+str(roix)+"y"+str(roiy)+"w"+str(roiw)+"h"+str(roih) 
		noisestring = "nx"+str(noisex)+"y"+str(noisey)+"w"+str(noisew)+"h"+str(noiseh) 
		picklePath = instructions['basepath']+instructions['target']+'/stats/'+instructions['target']+'_r'+str(countInput)+'_bd'+str(sigma)+'_mnf_'+roistring+noisestring+'.pickle'
		logger.info("%s Saving MNF stats to %s"%(current_process().name,picklePath))
		makedirs(instructions['basepath']+instructions['target']+'/stats',mode=0o755,exist_ok=True)
		pickle.dump(mnfr,open(picklePath,"wb"))
	stack = mnfr.reduce(stack,num=n_components)
	stack = img_as_float32(stack)
	stack = stack.transpose()
	logger.info('%s MNF finished'%(current_process().name))
	return stack

def startColor():
	msi2xyzFile = checkColorReady()
	if msi2xyzFile:
		logger.info("Confirmed ready to process color with msi2xyzFile "+msi2xyzFile)
		processColor(msi2xyzFile)
	else:
		logger.info("Not doing color processing")

def checkColorReady():
	if not 'color' in instructions['options']['methods']:
		logger.info("Color processing not selected")
		return False
	if 'white' in instructions and 'x' in instructions['white']:
		logger.info("White patch is defined for this page")
	else:
		logger.info("White patch is not defined for this page")
		return False
	if exists(instructions['basepath']+instructions['target']+'/Color'):
		logger.info("Color directory already exists, not repeating labor")
		return False
	else:
	 	logger.info("Color directory does not already exist, continuing")
	if 'msi2xyzFile' in instructions:
		if exists(instructions['basepath']+instructions['msi2xyzFile']):
			logger.info("Found cached msi2xyz.txt as specified in instructions")
			return instructions['basepath']+instructions['msi2xyzFile']
		else:
		 	logger.info("msi2xyzFile specified in metadata but file does not yet exist... need to check if have resources to generate it")
	else:
		logger.info("msi2xyzFile not specificed in metadata... necessary to specify path where file should go even if it does not yet exist")
		return False
	if exists(instructions['basepath']+'msi2xyz.txt'):
		logger.info("Found cached msi2xyz.txt in instructions['basepath']")
		return instructions['basepath']+'msi2xyz.txt'
	if exists(instructions['basepath']+'Calibration/msi2xyz.txt'):
		logger.info("Found cached msi2xyz.txt in Calibration directory")
		return instructions['basepath']+'Calibration/msi2xyz.txt'
	if exists(instructions['basepath']+'Calibration/Color/msi2xyz.txt'):
		logger.info("Found cached msi2xyz.txt in Calibration/Color directory")
		return instructions['basepath']+'Calibration/Color/msi2xyz.txt'
	logger.info("Looking for checker metadata in %s"%(instructions['basepath']+instructions['checkerMetadata']))
	if exists(instructions['basepath']+instructions['checkerMetadata']):
		instructions['msi2xyzFile']
		logger.info("Found the ingredients to generate an msi2xyz file")
		createMsi2Xyz()
		return instructions['basepath']+instructions['msi2xyzFile']
	else:
		logger.info("Don't have the ingredients to generate an msi2xyz file")
		return False

def createMsi2Xyz():
	with open(instructions['basepath']+instructions['checkerMetadata'],'r') as unparsedyaml:
		calibration = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
	if 'rotation' not in calibration['Calibration-Color']:
 		calibration['Calibration-Color']['rotation'] = 0
	capturedChecker = []
	whiteLevels = []
	for visibleBand in calibration['Calibration-Color']['visibleBands']:
		if 'sequenceShort' in calibration['Calibration-Color']:
			sequenceName = calibration['Calibration-Color']['sequenceShort']
		elif 'shortFilenameBase' in calibration['Calibration-Color']:
			sequenceName = calibration['Calibration-Color']['shortFilenameBase']
		else:
			sequenceName = calibration['Calibration-Color']['checkerCaptureDirectory']

		if needsFlattening(instructions['basepath']+sequenceName+'/'+calibration['Calibration-Color']['imagesets'][0]+'/'+sequenceName+'+'+visibleBand+'.tif'):
			logger.info("Flattening %s"%(instructions['basepath']+sequenceName+'_'+visibleBand+'.tif'))
			cacheFilePath = instructions['settings']['cachepath']+'flattened/'+sequenceName+'_'+visibleBand+'.tif'
			if exists(cacheFilePath): 
				img = io.imread(cacheFilePath)
			else: 
				for filename in listdir(instructions['basepath']+sequenceName+'/'+calibration['Calibration-Color']['imagesets'][0]):
					if visibleBand in filename:
						unflatPath = instructions['basepath']+sequenceName+'/'+calibration['Calibration-Color']['imagesets'][0]+'/'+filename
				img = openImageFile(unflatPath)
				for flatFile in listdir(instructions['basepath']+calibration['Calibration-Color']['flats']):
					if visibleBand in flatFile:
						flatPath = instructions['basepath']+calibration['Calibration-Color']['flats']+flatFile
				flat = openImageFile(flatPath)
				img = flatten(img,flat)
				img = rotate(img)
				io.imsave(cacheFilePath,img,check_contrast=False)
		else:
			logger.info("Opening already flattened image file %s"%(instructions['basepath']+sequenceName+'_'+visibleBand+'.tif'))
			img = openImageFile(instructions['basepath']+sequenceName+'/'+calibration['Calibration-Color']['imagesets'][0]+'/'+sequenceName+'+'+visibleBand+'.tif')

		whiteSample = img[
			calibration['Calibration-Color']['white']['y']:calibration['Calibration-Color']['white']['y']+calibration['Calibration-Color']['white']['h'],
			calibration['Calibration-Color']['white']['x']:calibration['Calibration-Color']['white']['x']+calibration['Calibration-Color']['white']['w'] ]
		whiteLevel = numpy.percentile(whiteSample,84) # median plus 1 standard deviation is equal to 84.1 percentile
		capturedChecker.append(img)
		whiteLevels.append(whiteLevel)
	capturedChecker = numpy.transpose(capturedChecker,axes=[1,2,0])
	nearMax = numpy.percentile(
		capturedChecker[
			calibration['Calibration-Color']['white']['y']:calibration['Calibration-Color']['white']['y']+calibration['Calibration-Color']['white']['h'],
			calibration['Calibration-Color']['white']['x']:calibration['Calibration-Color']['white']['x']+calibration['Calibration-Color']['white']['w'],
			:
		],
	84)
	capturedChecker = normalize(capturedChecker,whiteLevels,nearMax)
	checkerValues = measureCheckerValues(capturedChecker,calibration['Calibration-Color']['checkerMap'])
	checkerValues = numpy.array(checkerValues)
	checkerReference = XyzDict2array(calibration['Calibration-Color']['checkerReference'])
	logger.info("Calculating ratio of known patch values to measured patch values")
	checkerRatio = numpy.matmul( numpy.transpose(checkerReference) , numpy.transpose(numpy.linalg.pinv(checkerValues)) )
	numpy.savetxt(instructions['basepath']+instructions['msi2xyzFile'],checkerRatio,header='Matrix of XYZ x MSI Wavelengths, load with numpy.loadtxt()') 

def processColor(msi2xyzFile):
	makedirs(instructions['basepath']+instructions['target']+'/Color',mode=0o755,exist_ok=False)
	imgCube = []
	whiteLevels = []
	for visibleBand in instructions['visibleBands']:
		if 'sequenceShort' in instructions:
			sequenceName = instructions['sequenceShort']
		else:
			sequenceName = instructions['target']
		if 'shortFilenameBase' in instructions:
			filenameBase = instructions['shortFilenameBase']
		else:
			filenameBase = sequenceName
		if needsFlattening(instructions['basepath']+sequenceName+'/'+instructions['imagesets'][0]+'/'+filenameBase+'+'+visibleBand+'.tif'):
			cacheFilePath = instructions['settings']['cachepath']+'flattened/'+sequenceName+'+'+visibleBand+'.tif'
			if exists(cacheFilePath): 
				img = io.imread(cacheFilePath)
			else: 
				if 'Unflattened' in instructions['imagesets']:
					unflatPath = instructions['basepath']+instructions['target']+'/Unflattened/'
				elif 'Raw' in instructions['imagesets']:
					unflatPath = instructions['basepath']+instructions['target']+'/Raw/'
				elif 'Reflectance' in instructions['imagesets']:
					unflatPath = instructions['basepath']+instructions['target']+'/Reflectance/'
				else:
					exit("Need more info to know what image data to use for color calibration")
				for filename in listdir(unflatPath):
					if visibleBand in filename:
						unflatFile = unflatPath+filename
				img = openImageFile(unflatFile)
				for filename in listdir(instructions['basepath']+instructions['flats']): 
					if visibleBand in filename:
						flatFile = instructions['basepath']+instructions['flats']+filename
				flat = openImageFile(flatFile)
				img = flatten(img,flat)
				if not 'rotation' in instructions:
					instructions['rotation'] = 0
				img = rotate(img)
				io.imsave(cacheFilePath,img,check_contrast=False)
		else:
			img = openImageFile(instructions['basepath']+sequenceName+'/'+instructions['imagesets'][0]+'/'+filenameBase+'+'+visibleBand+'.tif')

		whiteSample = img[
			instructions['white']['y']:instructions['white']['y']+instructions['white']['h'],
			instructions['white']['x']:instructions['white']['x']+instructions['white']['w']
		] # note y before x
		whiteLevel = round(numpy.percentile(whiteSample,84),3) # median plus 1 standard deviation is equal to 84.1 percentile
		imgCube.append(img)
		whiteLevels.append(whiteLevel)
	imgCube = numpy.transpose(imgCube,axes=[1,2,0])
	nearMax = numpy.percentile(
		imgCube[
			instructions['white']['y']:instructions['white']['y']+instructions['white']['h'],
			instructions['white']['x']:instructions['white']['x']+instructions['white']['w'],
			:
		],
	84)
	imgCube = normalize(imgCube,whiteLevels,nearMax)
	height,width,layers=imgCube.shape
	imgCube = imgCube.reshape(height*width,layers)
	checkerRatio = numpy.loadtxt(msi2xyzFile)
	calibratedColor = numpy.matmul( checkerRatio , numpy.transpose(imgCube))
	calibratedColor = numpy.transpose(calibratedColor)
	calibratedColor = calibratedColor.reshape(height,width,3)
	calibratedColor = numpy.clip(calibratedColor,0,1)
	srgb = color.xyz2rgb(calibratedColor)
	srgb = exposure.rescale_intensity(srgb) 
	srgb = img_as_ubyte(srgb)
	srgbFilePath = instructions['basepath']+instructions['target']+'/Color/'+instructions['target']+'_sRGB.tif'
	logger.info("Saving sRGB tiff "+srgbFilePath)
	io.imsave(srgbFilePath,srgb,check_contrast=False) 
	jpgFilePath = instructions['basepath']+instructions['target']+'/Color/'+instructions['target']+'.jpg'
	logger.info("Saving jpeg file as "+jpgFilePath)
	io.imsave(jpgFilePath,srgb,check_contrast=False) 
	lab = color.xyz2lab(calibratedColor)
	lab = lab.astype('int8')
	labFilePath = instructions['basepath']+instructions['target']+'/Color/'+instructions['target']+'_LAB.tif'
	logger.info("Saving LAB tiff "+labFilePath)
	io.imsave(labFilePath,lab,check_contrast=False)

def normalize(img,whiteLevels,nearMax):
	logger.warning('Not actually sure that normalization is necessary with the checker matrix transformation method')
	for i in range(img.shape[2]):
		img[:,:,i] = img[:,:,i] * numpy.max(whiteLevels) / whiteLevels[i]
	logger.info("Changing expected white luminance from .95 to .88 did not change ΔE. Manufacturer spec for white patch is 95%, Roy likes 88%.")
	img = img * 0.88 / nearMax 
	img = numpy.clip(img,0,1)
	return img

def XyzDict2array(dict):
	array = []
	for i in range(1,25):
		chip = [ dict[i]['X'] , dict[i]['Y'], dict[i]['Z'] ]
		array.append(chip)
	array = numpy.array(array,dtype=numpy.float64)
	return array

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

if __name__ == '__main__':
	threadsMax = cpu_count()
	main()

