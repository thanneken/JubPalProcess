#!/usr/bin/env python
import numpy as np
import logging
from os import listdir, makedirs
from os.path import exists
from skimage import io, img_as_ubyte, img_as_uint, exposure
from sklearn.cross_decomposition import CCA
from sys import argv
import yaml 
from multiprocessing import Process, Queue, current_process 
from psutil import cpu_count

def findOptionsFile():
	for argument in argv:
		if 'cca.yaml' in argument:
			if exists(argument):
				print('Found cca.yaml as specified on the command line')
				return argument
			else:
				print('Found cca.yaml in command-line arguments, but the file does not exist')
				continue
	if exists('cca.yaml'):
		print('Found cca.yaml in present working directory')
		return 'cca.yaml'
	elif exists('git/JubPalProcess/cca.yaml'):
		print('Found cca.yaml in git/JubPalProcess/')
		return 'git/JubPalProcess/cca.yaml'
	else:
	 	exit('Cannot continue without specifying path to options.yaml')
def readStack(target):
	images = []
	for directory in inputDirectories:
		logger.info("Reading %s/%s"%(target,directory))
		for file in listdir(basepath+target+'/'+directory+'/'):
			img = io.imread(basepath+target+'/'+directory+'/'+file)
			images.append(img)
	img = np.stack(images)
	return img
def addTraining(img,observations):
	global training
	global labels
	ys, xs = np.nonzero(observations[:,:,3])
	newTraining = np.zeros((len(xs),img.shape[0]),dtype=np.uint16)
	newLabels = np.zeros((len(xs),3),dtype=np.uint8) 
	for i in range(len(xs)):
		newTraining[i] = img[:,ys[i],xs[i]]
		newLabels[i] = observations[ys[i],xs[i],:-1]
	if np.any(training):
		training = np.append(training,newTraining,axis=0)
	else:
		training = newTraining
	if np.any(labels):
		labels = np.append(labels,newLabels,axis=0)
	else:
		labels = newLabels

training = None
labels = None
optionsfile = findOptionsFile()
with open(optionsfile,'r') as unparsedyaml:
	instructions = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)

basepath = instructions['basepath']
target = instructions['target']
includeObservations = instructions['includeObservations']
inputDirectories = instructions['inputDirectories']
loglevel = instructions['loglevel']

if not exists(basepath+target+'/CCA/'): 
	print('Creating %s/CCA/ Directory'%(target))
	makedirs(basepath+target+'/CCA/',mode=0o755,exist_ok=False)
logfile = basepath+target+'/CCA/cca.log'
logger = logging.getLogger(__name__) 
logLevelObject = eval('logging.'+loglevel)
logging.basicConfig(filename=logfile,format='%(asctime)s %(levelname)s %(message)s',datefmt='%Y%m%d %H:%M:%S',level=logLevelObject) 
print("Follow logfile %s"%(logfile))
logger.info(" ~= Starting New Run =~")

if includeObservations:
	for include in includeObservations:
		observations = basepath+include+'/CCA/'+include+'_Observations.png'
		logger.info("Reading observations file %s"%(observations))
		observations = io.imread(observations)
		height,width,channels = observations.shape
		logger.info("Observations image is %s pixels wide, %s pixels high, and %s channels deep"%(width,height,channels))
		if not channels == 4:
			logger.warning("Expecting a four-channel observations image")
		logger.info("Loading stack of measurements for %s"%(include))
		img = readStack(include)
		logger.info("Measurements stack is %s channels deep, %s pixels high, and %s pixels wide"%(img.shape))
		logger.info("Assembling training set")
		addTraining(img,observations)
	logger.info("Labels has shape %s and dtype %s"%(labels.shape,labels.dtype))
	logger.info("Unique labels based on previous targets are %s"%(np.unique(labels,axis=0)))
logger.info("Loading stack of measurements for target %s"%(target))
img = readStack(target)
logger.info("Measurements stack is %s channels deep, %s pixels high, and %s pixels wide"%(img.shape))

if exists(basepath+target+'/CCA/'+target+'_Observations.png'):
	logger.info('Reading observations for target')
	observations = io.imread(basepath+target+'/CCA/'+target+'_Observations.png')
	height,width,channels = observations.shape
	logger.info("Observations image is %s pixels wide, %s pixels high, and %s channels deep"%(width,height,channels))
	if not channels == 4:
		logger.warning("Expecting a four-channel observations image")
	logger.info("Assembling training set from target")
	addTraining(img,observations)

logger.info("Labels has shape %s and dtype %s"%(labels.shape,labels.dtype))
logger.info("Unique labels are %s"%(np.unique(labels,axis=0)))

if len(np.unique(labels,axis=0)) > 2:
	n_components = 3
else:
	n_components = len(np.unique(labels,axis=0))

logger.info("Fitting training samples shape %s to labels shape %s with %s components"%(len(training),len(labels),n_components))
logger.debug("I'm not sure n_components is what I think it is. On Ambrosiana_C73inf_053 had the same result when n_components was 2 and 3")
cca = CCA(n_components=n_components)
cca.fit(training,labels)

logger.info("Reshaping image")
logger.info("Image has dtype, shape, min, and max %s %s %s %s"%(img.dtype,img.shape,np.min(img),np.max(img)))
layers,height,width = img.shape
img = np.transpose(img)
img = img.reshape((height*width,layers))
logger.info("Image has dtype, shape, min, and max %s %s %s %s"%(img.dtype,img.shape,np.min(img),np.max(img)))

logger.info("Predicting for full image")
img = cca.predict(img)

logger.info("Reshaping image")
logger.info("Image has dtype, shape, min, and max %s %s %s %s"%(img.dtype,img.shape,np.min(img),np.max(img)))
img = img.reshape(width,height,3)
img = np.transpose(img,(1,0,2))
logger.info("Image has dtype, shape, min, and max %s %s %s %s"%(img.dtype,img.shape,np.min(img),np.max(img)))

maskx,masky,maskw,maskh = int(width/3),int(height/3),int(width/3),int(height/3)
logger.info("Adjusting histogram with mask x,y,w,h = %s,%s,%s,%s"%(maskx,masky,maskw,maskh))
mask = np.zeros((height,width),bool)
mask[masky:masky+maskh,maskx:maskx+maskw] = 1
img = exposure.equalize_hist(img,mask=mask)
logger.info("Image has dtype, shape, min, and max %s %s %s %s"%(img.dtype,img.shape,np.min(img),np.max(img)))

img = img_as_uint(img)
outfile = "%s%s/CCA/%s_cca_%s.tif"%(basepath,target,target,len(training))
logger.info("Saving %s"%(outfile))
io.imsave(outfile,img)
outfile = outfile[:-3]+'jpg'
img = img_as_ubyte(img)
logger.info("Saving %s"%(outfile))
io.imsave(outfile,img)

logger.info("Concluded successfully\n")
print("Concluded successfully\n")

