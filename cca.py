#!/usr/bin/env python
import numpy as np
import logging
from os import listdir, makedirs
from os.path import exists
from skimage import io, img_as_ubyte, img_as_uint, exposure
from sklearn.cross_decomposition import CCA

basepath = '/storage/JubPalProj/AmbrosianaArchive/Ambrosiana_C73inf/'
target = 'Ambrosiana_C73inf_055' 
includeObservations = [ 
	'Ambrosiana_C73inf_054'
]
ignore = [
	'Ambrosiana_C73inf_053' 
	,
	'Ambrosiana_C73inf_049' 
]
inputDirectories = [ 'Captures-Fluorescence-NoGamma' , 'Captures-Narrowband-NoGamma' , 'Captures-Transmissive-NoGamma' ]
loglevel = 'INFO'

if not exists(basepath+target+'/CCA/'): 
	print('Creating %s/CCA/ Directory'%(target))
	makedirs(basepath+target+'/CCA/',mode=0o755,exist_ok=False)
logfile = basepath+target+'/CCA/cca.log'
logger = logging.getLogger(__name__) 
logLevelObject = eval('logging.'+loglevel)
logging.basicConfig(filename=logfile,format='%(asctime)s %(levelname)s %(message)s',datefmt='%Y%m%d %H:%M:%S',level=logLevelObject) 
print("Follow logfile %s"%(logfile))
logger.info(" ~= Starting New Run =~")

def readStack(target):
	images = []
	for directory in inputDirectories:
		logger.info("Reading %s/%s"%(target,directory))
		for file in listdir(basepath+target+'/'+directory+'/'):
			img = io.imread(basepath+target+'/'+directory+'/'+file)
			images.append(img)
	img = np.stack(images)
	return img

training = []
labels = []
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
	logger.info("Assembling training set may take a few minutes (not currently multithreaded)")
	for y in range(height):
		if y%1000 == 0:
			logger.info("At line %s"%(y))
		for x in range(width):
			if observations[y,x,3] > 0: # expecting 255 or 1 for opaque pixels, 0 for transparent pixels
				training.append(img[:,y,x]) 
				labels.append(observations[y,x,:-1])

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
	logger.info("Assembling training set may take a few minutes")
	for y in range(height):
		if y%1000 == 0:
			logger.info("At line %s"%(y))
		for x in range(width):
			if observations[y,x,3] > 0: 
				training.append(img[:,y,x]) 
				labels.append(observations[y,x,:-1])

logger.info("Unique labels are %s"%(np.unique(labels,axis=0)))
n_components = len(np.unique(labels,axis=0))
if n_components != 3:
	logger.warning('Are you sure you wish to decompose to %s components?'%(n_components))

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

