#!/home/thanneken/python/miniconda3/bin/python
import multiprocessing
from skimage import io, img_as_float32, img_as_uint, img_as_ubyte, filters, exposure, color
from os import listdir, makedirs
from os.path import exists, join
import time 
import yaml 
import inquirer
import numpy 
import rawpy
import pyexifinfo
import logging
import sys

# DEFINE OUR FUNCTIONS
def openImageFile(filePath):
	if filePath.endswith('.dng'):
		with rawpy.imread(filePath) as raw:
			return raw.raw_image.copy() 
	else:
			return io.imread(filePath)
def openrawfile(rawfile):
	with rawpy.imread(rawfile) as raw:
		return raw.raw_image.copy() 
def normalize(img,whiteLevels,nearMax):
	for i in range(img.shape[2]):
		img[:,:,i] = img[:,:,i] * numpy.max(whiteLevels) / whiteLevels[i]
	logger.info("Changing expected white luminance from .95 to .88 did not change ΔE. Manufacturer spec for white patch is 95%, Roy likes 88%.")
	img = img * 0.88 / nearMax 
	img = numpy.clip(img,0,1)
	return img
def processColor(msi2xyzFile):
	makedirs(basepath+project+'/Color',mode=0o755,exist_ok=False)
	imgCube = []
	whiteLevels = []
	for visibleBand in metadata['visibleBands']:
		if 'sequenceShort' in metadata:
			sequenceName = metadata['sequenceShort']
		else:
			sequenceName = project
		cacheFilePath = cachepath+'flattened/'+sequenceName+'+'+visibleBand+'.tif'
		if exists(cacheFilePath): 
			img = io.imread(cacheFilePath)
		else: 
			if 'Unflattened' in metadata['imagesets']:
				unflatPath = basepath+project+'/Unflattened/'
			elif 'Raw' in metadata['imagesets']:
				unflatPath = basepath+project+'/Raw/'
			elif 'Reflectance' in metadata['imagesets']:
				unflatPath = basepath+project+'/Reflectance/'
			else:
				exit("Need more info to know what image data to use for color calibration")
			for filename in listdir(unflatPath):
				if visibleBand in filename:
					unflatFile = unflatPath+filename
			img = openImageFile(unflatFile)
			for filename in listdir(basepath+metadata['flats']): 
				if visibleBand in filename:
					flatFile = basepath+metadata['flats']+filename
			flat = openImageFile(flatFile)
			img = flatten(img,flat)
			if not 'rotation' in metadata:
			 	metadata['rotation'] = 0
			img = rotate(img,metadata['rotation'])
			io.imsave(cacheFilePath,img,check_contrast=False)
		whiteSample = img[
			metadata['white']['y']:metadata['white']['y']+metadata['white']['h'],
			metadata['white']['x']:metadata['white']['x']+metadata['white']['w']
		] # note y before x
		whiteLevel = round(numpy.percentile(whiteSample,84),3) # median plus 1 standard deviation is equal to 84.1 percentile
		imgCube.append(img)
		whiteLevels.append(whiteLevel)
	imgCube = numpy.transpose(imgCube,axes=[1,2,0])
	nearMax = numpy.percentile(
		imgCube[
			metadata['white']['y']:metadata['white']['y']+metadata['white']['h'],
			metadata['white']['x']:metadata['white']['x']+metadata['white']['w'],
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
	srgbFilePath = basepath+project+'/Color/'+project+'_sRGB.tif'
	logger.info("Saving sRGB tiff "+srgbFilePath)
	io.imsave(srgbFilePath,srgb,check_contrast=False) 
	jpgFilePath = basepath+project+'/Color/'+project+'.jpg'
	logger.info("Saving jpeg file as "+jpgFilePath)
	io.imsave(jpgFilePath,srgb,check_contrast=False) 
	lab = color.xyz2lab(calibratedColor)
	lab = lab.astype('int8')
	labFilePath = basepath+project+'/Color/'+project+'_LAB.tif'
	logger.info("Saving LAB tiff "+labFilePath)
	io.imsave(labFilePath,lab,check_contrast=False)
def checkColorReady():
	if not 'color' in methods:
		logger.info("Color processing not selected")
		return False
	if 'x' in metadata['white']:
		logger.info("White patch is defined for this page")
	else:
		logger.info("White patch is not defined for this page")
		return False
	if exists(basepath+project+'/Color'):
		logger.info("Color directory already exists, not repeating labor")
		return False
	else:
	 	logger.info("Color directory does not already exist, continuing")
	if 'msi2xyzFile' in metadata:
		if exists(basepath+metadata['msi2xyzFile']):
			logger.info("Found cached msi2xyz.txt as specified in metadata")
			return basepath+metadata['msi2xyzFile']
		else:
		 	logger.info("msi2xyzFile specified in metadata but file does not yet exist... need to check if have resources to generate it")
	else:
		logger.info("msi2xyzFile not specificed in metadata... necessary to specify path where file should go even if it does not yet exist")
		return False
	if exists(basepath+'msi2xyz.txt'):
		logger.info("Found cached msi2xyz.txt in basepath")
		return basepath+'msi2xyz.txt'
	if exists(basepath+'Calibration/msi2xyz.txt'):
		logger.info("Found cached msi2xyz.txt in Calibration directory")
		return basepath+'Calibration/msi2xyz.txt'
	if exists(basepath+'Calibration/Color/msi2xyz.txt'):
		logger.info("Found cached msi2xyz.txt in Calibration/Color directory")
		return basepath+'Calibration/Color/msi2xyz.txt'
	if exists(basepath+metadata['checkerMetadata']):
		metadata['msi2xyzFile']
		logger.info("Found the ingredients to generate an msi2xyz file")
		createMsi2Xyz()
		return basepath+metadata['msi2xyzFile']
	else:
		logger.info("Don't have the ingredients to generate an msi2xyz file")
		return False
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
def createMsi2Xyz():
	with open(basepath+metadata['checkerMetadata'],'r') as unparsedyaml:
		calibration = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
	if 'rotation' not in calibration['Calibration-Color']:
 		calibration['Calibration-Color']['rotation'] = 0
	capturedChecker = []
	whiteLevels = []
	for visibleBand in calibration['Calibration-Color']['visibleBands']:
		if 'sequenceShort' in calibration['Calibration-Color']:
			sequenceName = calibration['Calibration-Color']['sequenceShort']
		else:
			sequenceName = calibration['Calibration-Color']['checkerCaptureDirectory']
		cacheFilePath = cachepath+'flattened/'+sequenceName+'_'+visibleBand+'.tif'
		if exists(cacheFilePath): 
			img = io.imread(cacheFilePath)
		else: 
			for filename in listdir(basepath+sequenceName+'/'+calibration['Calibration-Color']['imagesets'][0]):
				if visibleBand in filename:
					unflatPath = basepath+sequenceName+'/'+calibration['Calibration-Color']['imagesets'][0]+'/'+filename
			# img = openrawfile(basepath+'../Calibration/Calibration-Color/'+checkerMetadata['imagesets'][0]+'/'+checkerMetadata['shortFilenameBase']+'+'+visibleBand+'.dng')
			img = openImageFile(unflatPath)
			for flatFile in listdir(basepath+calibration['Calibration-Color']['flats']):
				if visibleBand in flatFile:
					flatPath = basepath+calibration['Calibration-Color']['flats']+flatFile
			flat = openImageFile(flatPath)
			img = flatten(img,flat)
			img = rotate(img,calibration['Calibration-Color']['rotation'])
			io.imsave(cacheFilePath,img,check_contrast=False)
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
	numpy.savetxt(basepath+metadata['msi2xyzFile'],checkerRatio,header='Matrix of XYZ x MSI Wavelengths, load with numpy.loadtxt()') 
def nextNeededProject(projects):
	# matches = (f for f in projects if exists(basepath+f) and not exists(basepath+f+'/Transform'))
	matches = (f for f in projects if exists(basepath+f) and not exists(basepath+f+'/Color'))
	return next(matches,None)
def blurDivide(img,sigma):
	if not img.dtype == "float32":
		img = img_as_float32(img)
	numerator = filters.median(img) # default is 3x3, same as RLE suggested
	denominator = filters.gaussian(img,sigma=sigma)
	ratio = numpy.divide(numerator,denominator,out=numpy.zeros_like(numerator),where=denominator!=0)
	return ratio
def flatten(unflat,flat):
	if metadata['blurImage'] == "median3":
		logger.info("Blurring image with 3x3 median")
		unflat = filters.median(unflat) # default is 3x3
	if metadata['blurFlat'] > 0: 
		logger.info("Blurring flat with sigma "+str(metadata['blurFlat']))
		flat = filters.gaussian(flat,sigma=metadata['blurFlat'])
	# logger.warning('Temporary blur of unflat is probably not sustainable')
	# unflat = filters.gaussian(unflat,sigma=metadata['blurFlat'])
	return numpy.divide(unflat*numpy.average(flat),flat,out=numpy.zeros_like(unflat*numpy.average(flat)),where=flat!=0)
def rotate(img,rotation): 
	if rotation == 90:
		img = numpy.rot90(img,k=3)
	elif rotation == 180:
		img = numpy.rot90(img,k=2)
	elif rotation == 270:
		img = numpy.rot90(img,k=1)
	else:
		logger.info("No rotation identified")
	return img

def findFlat(filename,fullpath): # replaces major portion of flattenrotate
	if not 'flats' in metadata:
		logger.error("It is necessary to specify relative path to flats in YAML metadata")
		exit("It is necessary to specify relative path to flats in YAML metadata")
	try: # find exact file in DNG header
		exif = pyexifinfo.get_json(fullpath)
		exifflat = exif[0]["IPTC:Keywords"][11] # specific to MegaVision
		if exifflat.endswith('.dn'):
			exifflat = exifflat+'g' 
		match = exifflat
	except: 
		try: # find last 11 character match in filenames
			for flatFile in listdir(basepath+metadata['flats']):
				if flatFile[-11:] == fullpath[-11:]:
					logger.info('Found 11 character match: '+filename+' ~ '+flatFile)
					match = flatFile
			assert match
		except:
			try: # find last 7 character match in filenames
				for flatFile in listdir(basepath+metadata['flats']):
					print("Looking for",fullpath[-7:],"at end of",flatFile)
					if flatFile[-7:] == fullpath[-7:]:
						logger.info('Found 7 character match: '+filename+' ~ '+flatFile)
						match = flatFile
				assert match
			except:
				logger.error("Unable to find flatfile for this capture")
				exit("Unable find flatfile for this capture")
	img = openImageFile(basepath+metadata['flats']+match)
	return img
def flattenrotate(fullpath): # deprecated and replced with findFlat(), flatten(), findRotation(), and rotate()
		file = fullpath.split('/')[-1][:-4] # filename is everything after the last forward slash, and remove the extension too
		if exists(cachepath+'flattened/'+file+'.tif'): # check cache
				logger.info("Found in cache: flattened/"+file+'.tif')
				flattenedfloat = io.imread(cachepath+'flattened/'+file+'.tif')
		else:
				with rawpy.imread(fullpath) as raw:
						capture = raw.raw_image.copy()
				flatsdir = metadata['flats']
				flatpath = basepath+flatsdir+exifflat
				if not exists(flatpath): # if metadata doesn't work look for file in directory with right index number
						logger.info("According to EXIF, flat is "+flatpath) 
						for flatfile in listdir(basepath+flatsdir): 
								if flatfile[-7:] == fullpath[-7:]:
										flatpath = basepath+flatsdir+flatfile
										logger.info("EXIF identified flat not found, flat with same index number is "+flatpath)
				with rawpy.imread(flatpath) as raw:
						flat = raw.raw_image.copy()
				# flattenedfloat = capture*numpy.average(flat)/flat
				flattenedfloat = numpy.divide(capture*numpy.average(flat),flat,out=numpy.zeros_like(capture*numpy.average(flat)),where=flat!=0)
				# might make sense to split flatten and rotate into separate functions
				# look for rotation in yaml, exif, or filename
				if 'rotation' in metadata:
					print("I think something got lost or duplicated here but ignoring because deprecated anyway")
				else:
					exif = pyexifinfo.get_json(fullpath)
					exifflat = exif[0]["IPTC:Keywords"][11] # was index 11 n array of keywords in 2017, likely to be different in 2023
					if exifflat.endswith('.dn'):
						exifflat = exifflat+'g' # the last letter got cut off in 2017, likely to be different in 2023
					exiforientation = exif[0]["EXIF:Orientation"]
				if exiforientation == "Rotate 90 CW": # counter-intuitive, read as rotated 90 CW and rotate 270 to correct
						flattenedfloat = numpy.rot90(flattenedfloat,k=3)
				elif exiforientation == "Rotate 180":
					if file[11:12] == 'r': # 2023 Ambrosiana is not coded correctly in metadata, go by filename
						logger.info("Using rotation for rectos")
						flattenedfloat = numpy.rot90(flattenedfloat,k=3) 
					elif file[11:12] == 'v':
						logger.info("Using rotation for versos")
						flattenedfloat = numpy.rot90(flattenedfloat,k=1) 
					else:
						flattenedfloat = numpy.rot90(flattenedfloat,k=2) 
				elif exiforientation == "Rotate 90 CCW":
						flattenedfloat = numpy.rot90(flattenedfloat)
				elif exiforientation == "Rotate 270 CW":
						flattenedfloat = numpy.rot90(flattenedfloat)
				# save flat to cache
				flattenedfloat = img_as_float32(flattenedfloat)
				makedirs(cachepath+'flattened/',mode=0o755,exist_ok=True)
				io.imsave(cachepath+'flattened/'+file+'.tif',flattenedfloat,check_contrast=False)
		return flattenedfloat
def findRotation(filename,fullpath): # replaces major portion of flattenrotate
	if 'rotation' in metadata:
		rotation = metadata['rotation']
		logger.info('Using rotation of '+str(rotation)+'° based on yaml metadata')
	else:
		exif = pyexifinfo.get_json(fullpath)
		exiforientation = exif[0]["EXIF:Orientation"]
		if exiforientation == 'Rotate 90 CW':
			rotation = 90
		elif exiforientation == 'Rotate 180':
			rotation = 180
		elif exiforientation == 'Rotate 270 CW':
			rotation = 270
		elif exiforientation == 'Rotate 90 CCW':
			rotation = 270
	return rotation
def blurDivide(img,sigma): 
	if not img.dtype == "float32":
		img = img_as_float32(img)
	numerator = filters.median(img) # default is 3x3, same as RLE suggested
	denominator = filters.gaussian(img,sigma=sigma)
	ratio = numpy.divide(numerator,denominator,out=numpy.zeros_like(numerator),where=denominator!=0)
	return ratio
def prepImg(q,fullpath,sigma): # replaces readnblur
	filename = fullpath.split('/')[-1][:-4]
	blurCacheFile = cachepath+'denoise/sigma'+str(sigma)+'/'+filename+'.tif'
	flattenedCacheFile = cachepath+'flattened/'+filename+'.tif'
	if exists(blurCacheFile): # Already got to blur and divide for this sigma, just need to return and move on
		img = io.imread(blurCacheFile)
		logger.info(blurCacheFile+' found in cache')
	else:
		if 'NoGamma' in fullpath: # if image is already flattened just open and put in queue
			logger.info(fullpath+' does not need to be flattened')
			img = io.imread(fullpath)
		elif exists(flattenedCacheFile):
			logger.info(flattenedCacheFile+' found in cache')
			img =  io.imread(flattenedCacheFile)
		else: 
			img = openImageFile(fullpath)
			img = img_as_float32(img) 
			flatImg = findFlat(filename,fullpath) 
			img = flatten(img,flatImg)
			rotation = findRotation(filename,fullpath) 
			img = rotate(img,rotation) 
			img = img_as_float32(img)
			io.imsave(flattenedCacheFile,img,check_contrast=False)
			logger.info(flattenedCacheFile+' saved to cache')
		if sigma > 0:
			img = blurDivide(img,sigma) 
			img = exposure.rescale_intensity(img)
			img = img_as_float32(img)
			io.imsave(blurCacheFile,img,check_contrast=False)
			logger.info(blurCacheFile+' saved to cache')
	q.put(img)
def readnblur(q,fullpath,sigma): # deprecated and replaced with prepImg
		file = fullpath.split('/')[-1][:-4] 
		if exists(cachepath+'denoise/sigma'+str(sigma)+'/'+file+'.tif'): 
				logger.info(file+".tif found in cache")
				img = io.imread(cachepath+'denoise/sigma'+str(sigma)+'/'+file+'.tif')
		else:
				logger.info("Reading "+fullpath)
				if (fullpath.endswith('.tif')): # if ends in tif then read
						img = io.imread(fullpath)
						img = img_as_float32(img)
				elif (fullpath.endswith('.dng')): # if ends in dng then flatten and rotate
						img = flattenrotate(fullpath)
				if (sigma > 0):
						img = blurDivide(img,sigma)
						img = exposure.rescale_intensity(img)
						makedirs(cachepath+'denoise/sigma'+str(sigma)+'/',mode=0o755,exist_ok=True)
						io.imsave(cachepath+'denoise/sigma'+str(sigma)+'/'+file+'.tif',img,check_contrast=False)
		q.put(img)
def stacker(sigma):
	countinput = 0 
	stack = []
	q = multiprocessing.Queue(maxsize=1)
	processes = []
	for imageset in imagesets:
		for file in listdir(basepath+project+'/'+imageset):
			#skip this file if matches regex UVB_|UVP_
			if ((skipuvbp == True) and (("UVB_" in file) or ("UVP_" in file))):
				continue
			fullpath = basepath+project+'/'+imageset+'/'+file
			countinput += 1
			p = multiprocessing.Process(target=prepImg,args=(q,fullpath,sigma)) # replace readnblur() with prepImg()
			processes.append(p)
			p.start()
	for process in processes: 
		stack.append(q.get())
	for process in processes: 
		process.join() # added 6/17/2023, fixed 3/2/2024
	stack = numpy.array(stack)
	return stack, countinput
# EXPOSURE
def deprecated_histogram_adjust(outpath,outfile,histograms,d3_processed,fileformats,multilayer,n_components): # deprecate 6/17/2023
	processes = []
	for histogram in histograms:
		p = multiprocessing.Process(target=histogram_adjust_thread,args=(outpath,outfile,histogram,d3_processed,fileformats,multilayer,n_components))
		processes.append(p)
		p.start()
		p.join() # added 6/17/2023
def histogram_adjust(outpath,outfile,histograms,d3_processed,fileformats,multilayer,n_components):
	for histogram in histograms:
		histogram_adjust_thread(outpath,outfile,histogram,d3_processed,fileformats,multilayer,n_components)
		if False:
			p = multiprocessing.Process(target=histogram_adjust_thread,args=(outpath,outfile,histogram,d3_processed,fileformats,multilayer,n_components)) 
			p.start()
def histogram_adjust_thread(outpath,outfile,histogram,d3_processed,fileformats,multilayer,n_components):
	if histogram == 'equalize':
		logger.info("Performing histogram equalization")
		adjusted_eq = d3_processed
		for i in range (0,n_components):
			adjusted_eq[i,:,:]	= exposure.equalize_hist(adjusted_eq[i,:,:])
		outpath_h = join(outpath,histogram)
		outfile_h = outfile+'_'+histogram
		save_all_formats(adjusted=adjusted_eq,histogram=histogram,outpath=outpath_h,outfile=outfile_h,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
	elif histogram == 'rescale':
		logger.info("Performing histogram rescale")
		adjusted_rs = d3_processed
		for i in range (0,n_components):
			adjusted_rs[i,:,:]	= exposure.rescale_intensity(adjusted_rs[i,:,:])
		outpath_h = join(outpath,histogram)
		outfile_h = outfile+'_'+histogram
		save_all_formats(adjusted=adjusted_rs,histogram=histogram,outpath=outpath_h,outfile=outfile_h,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
	elif histogram == 'adaptive':
		logger.info("Performing adaptive histogram equalization")
		adjusted_ad = d3_processed
		for i in range (0,n_components):
			adjusted_ad[i,:,:] = exposure.rescale_intensity(adjusted_ad[i,:,:])
			adjusted_ad[i,:,:] = exposure.equalize_adapthist(adjusted_ad[i,:,:], clip_limit=0.03)
		outpath_h = join(outpath,histogram)
		outfile_h = outfile+'_'+histogram
		save_all_formats(adjusted=adjusted_ad,histogram=histogram,outpath=outpath_h,outfile=outfile_h,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
	else:
		logger.info("Passing with no histogram adjustment")
		adjusted_n = d3_processed
		outpath_h = join(outpath,histogram)
		outfile_h = outfile+'_'+histogram
		save_all_formats(adjusted=adjusted_n,histogram=histogram,outpath=outpath_h,outfile=outfile_h,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
# OUTPUT 
def save_all_formats(adjusted,histogram,outpath,outfile,fileformats,multilayer,n_components):
	for fileformat in fileformats:
		## create path and filename
		outpath_current = join(outpath,fileformat)
		outfile_current = outfile
		makedirs(outpath_current,mode=0o755,exist_ok=True)
		outfile_current = join(outpath_current,outfile_current)
		## write to disk as requested
		if multilayer == 'stack' and fileformat == 'tif':
			outfile_current = outfile_current+'.'+fileformat
			logger.info("Saving "+outfile_current)
			img32 = img_as_float32(adjusted)
			io.imsave(outfile_current,img32,check_contrast=False)
		elif n_components == 1:
			outfile_current = outfile_current+'.'+fileformat
			component = adjusted[0,:,:]
			logger.info("Saving "+outfile_current)
			if fileformat == 'tif':
				img32 = img_as_float32(component)
				io.imsave(outfile_current,img32,check_contrast=False)
			elif fileformat == 'png':
				img16 = img_as_uint(component)
				io.imsave(outfile_current,img16,check_contrast=False)
			elif fileformat == 'jpg':
				img8 = img_as_ubyte(component)
				io.imsave(outfile_current,img8,check_contrast=False)
		elif multilayer == 'separate files':
			for i in range (0,n_components):
				outfilec = outfile_current+'_c'+str(f"{i:02d}")+'.'+fileformat
				logger.info("Saving "+outfilec)
				component = adjusted[i,:,:]
				if fileformat == 'tif':
					img32 = img_as_float32(component)
					io.imsave(outfilec,img32,check_contrast=False)
				elif fileformat == 'png':
					img16 = img_as_uint(component)
					io.imsave(outfilec,img16,check_contrast=False)
				elif (fileformat=='jpg' and histogram=='none'):
					logger.warn("Can't save floating point to jpeg without at least some histogram adjustment")
				elif fileformat == 'jpg':
					img8 = img_as_ubyte(component)
					io.imsave(outfilec,img8,check_contrast=False)

# GATHER INPUT
datafile = '/home/thanneken/git/JubPalProcess/options.yaml'
print("Reading options from",datafile)
with open(datafile,'r') as unparsedyaml:
		jubpaloptions = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
## read non-interactive options
cachepath = jubpaloptions["settings"]["cachepath"]
fica_max_iter = jubpaloptions["settings"]["fica_max_iter"]
fica_tol = jubpaloptions["settings"]["fica_tol"]
logfile = jubpaloptions["settings"]["logfile"]
illuminant = jubpaloptions['settings']['color']['illuminant']
## configure logging
logger = logging.getLogger(__name__) # necessary to instantiate?
if exists(logfile): 
	logging.basicConfig(
		filename=logfile, 
		format='%(asctime)s %(levelname)s %(message)s', 
		datefmt='%Y%m%d %H:%M:%S', 
		level=logging.DEBUG # levels are DEBUG INFO WARNING CRITICAL ERROR CRITICAL
	) 
else:
	print("Log file not specified or not found")

## Determine whether in use interactive mode
if 'noninteractive' in sys.argv:
	interactive = False
elif len(jubpaloptions["options"]["interactive"]) > 1:
	questions = [inquirer.List('interactive',"Proceed with interactive choices?",choices=jubpaloptions["options"]["interactive"])]
	selections = inquirer.prompt(questions)
	interactive = selections["interactive"]
else:
	interactive = jubpaloptions["options"]["interactive"][0]

if interactive:
	## select one basepath
	if len(jubpaloptions["basepaths"]) > 1:
		questions = [inquirer.List('basepath',"Select basepath for source data",choices=jubpaloptions["basepaths"])]
		selections = inquirer.prompt(questions)
		basepath = selections["basepath"]
	else:
		basepath = jubpaloptions["basepaths"][0]
	## look for project list in selected basepath
	projectsfile = basepath+basepath.split('/')[-2]+'.yaml'
	if exists(projectsfile):
		with open(projectsfile,'r') as unparsedyaml:
				projects = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
	else:
	 	exit('Unable to find '+projectsfile)

	## select one project
	projectChoices = list(projects.keys())
	projectChoices.remove('default') 
	if len(projectChoices) > 1:
		questions = [ inquirer.List('project',"Select project",choices=projectChoices) ]
		selections = inquirer.prompt(questions)
		project = selections["project"]
	else:
		project = projectChoices[0]
	metadata = {}
	metadata.update(projects['default'])
	metadata.update(projects[project])
	if 'white' in metadata: 
		metadata['white'].update(projects['default']['white'])
		metadata['white'].update(projects[project]['white'])

	## select one or more methods
	if len(jubpaloptions["options"]["methods"]) > 1:
		questions = [ inquirer.Checkbox('methods',"Select Process",choices=jubpaloptions["options"]["methods"]) ]
		methods = []
		while len(methods) < 1:
			selections = inquirer.prompt(questions)
			methods = selections["methods"]
		methods = selections["methods"]
	else:
		methods = jubpaloptions["options"]["methods"][0]

	if any(x in methods for x in ['kpca','pca','mnf','fica']):
		linearTransformations = True
	else:
		linearTransformations = False
		sigmas = []

	if linearTransformations:
		## select one or more sigma for blur and divide
		if len(jubpaloptions["options"]["sigmas"]) > 1:
			questions = [ inquirer.Checkbox('sigmas',"Sigma for RLE blur and divide?",choices=jubpaloptions["options"]["sigmas"]) ]
			sigmas = []
			while len(sigmas) < 1:
				selections = inquirer.prompt(questions)
				sigmas = selections["sigmas"] 
		else:
			sigmas = jubpaloptions["options"]["sigmas"][0]

		## select one of skipuvbp boolean
		if len(jubpaloptions["options"]["skipuvbp"]) > 1:
			questions = [ inquirer.List('skipuvbp',"Skip files with UVB_ or UVP_ in filename?",choices=jubpaloptions["options"]["skipuvbp"]) ]
			selections = inquirer.prompt(questions)
			skipuvbp = selections["skipuvbp"]
		else:
			skipuvbp = jubpaloptions["options"]["skipuvbp"][0]

		## select one number of components
		if len(jubpaloptions["options"]["n_components"]) > 1:
			questions = [ inquirer.List('n_components',"How many components to generate?",choices=jubpaloptions["options"]["n_components"]) ]
			selections = inquirer.prompt(questions)
			n_components = selections["n_components"]
		else:
			n_components = jubpaloptions["options"]["n_components"][0]
		## select one or more image sets
		imagesetOptions = metadata['imagesets']
		if len(imagesetOptions) > 1:
			questions = [ inquirer.Checkbox('imagesets',"Select one or more image sets",choices=imagesetOptions) ]
			imagesets = []
			while len(imagesets) < 1:
				selections = inquirer.prompt(questions)
				imagesets = selections["imagesets"]
		else:
			imagesets = imagesetOptions
		## select one roi, eventually one or more
		if len(metadata['rois'].keys()) > 1: 
			questions = [
				inquirer.List('roi',"Select region of interest",choices=metadata["rois"].keys())
			]
			selections = inquirer.prompt(questions)
			roi = selections["roi"]
		else:
			roi = list(metadata["rois"].keys())[0] 
		roix = metadata["rois"][roi]["x"]
		roiy = metadata["rois"][roi]["y"]
		roiw = metadata["rois"][roi]["w"]
		roih = metadata["rois"][roi]["h"]
		roilabel = metadata["rois"][roi]["label"]
		## select noise sample for mnf 
		if ('mnf' in methods):
				if len(metadata["noisesamples"].keys()) > 1:
					questions = [
						inquirer.List('noisesample',"Select Noise Region",choices=metadata["noisesamples"].keys())
					]
					selections = inquirer.prompt(questions)
					noisesample = selections["noisesample"]
				else:
					noisesample = list(metadata["noisesamples"].keys())[0] 
				noisesamplex = metadata["noisesamples"][noisesample]["x"]
				noisesampley = metadata["noisesamples"][noisesample]["y"]
				noisesamplew = metadata["noisesamples"][noisesample]["w"]
				noisesampleh = metadata["noisesamples"][noisesample]["h"]
				noisesamplelabel = metadata["noisesamples"][noisesample]["label"]
				noisestring = 'x'+str(noisesamplex)+'y'+str(noisesampley)+'w'+str(noisesamplew)+'h'+str(noisesampleh)
		## select one or more histogram adjustments
		if len(jubpaloptions["output"]["histograms"]) > 1:
			questions = [ inquirer.Checkbox('histograms',"Select histogram adjustment(s) for final product",choices=jubpaloptions["output"]["histograms"]) ]
			histograms = []
			while len(histograms) < 1:
				selections = inquirer.prompt(questions)
				histograms = selections["histograms"]
		else:
			histograms = jubpaloptions["output"]["histogram"][0]
		## select multilayer as stack or separate files
		if len(jubpaloptions["output"]["multilayer"]) > 1:
			questions = [ inquirer.List('multilayer',"Select what to do with multiple layers",choices=jubpaloptions["output"]["multilayer"]) ]
			selections = inquirer.prompt(questions)
			multilayer = selections["multilayer"]
		else:
			multilayer = jubpaloptions["output"]["multilayer"][0]
		basepathout = basepath	# put output in the same directory 
		## select one or more fileformat
		if len(jubpaloptions["output"]["fileformats"]) > 1:
			questions = [ inquirer.Checkbox('fileformats',"Select file format(s) to output",choices=jubpaloptions["output"]["fileformats"]) ]
			fileformats = []
			while len(fileformats) < 1:
				selections = inquirer.prompt(questions)
				fileformats = selections["fileformats"]
		else:
			fileformats = jubpaloptions["output"]["fileformats"][0] 
else: # make non-interactive choices
	basepath = jubpaloptions["basepaths"][0] # first listed basepath 
	projectsfile = basepath+basepath.split('/')[-2]+'.yaml'
	if exists(projectsfile):
		with open(projectsfile,'r') as unparsedyaml:
				projects = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
	else:
	 	exit('Unable to find '+projectsfile)
	project = nextNeededProject(projects.keys())
	if project == None:
		print("No projects defined in",projectsfile,"(the first named basepath) lack a Transform directory")
		logger.info("No projects defined in "+projectsfile+" (the first named basepath) lack a Transform directory")
		exit()
	sigmas = jubpaloptions["options"]["sigmas"] # all listed sigmas
	skipuvbp = jubpaloptions["options"]["skipuvbp"][0] # top option for Skip UVB and UVP
	methods = jubpaloptions["options"]["methods"] # all listed methods
	n_components = jubpaloptions["options"]["n_components"][0] # first named number of components (max)
	metadata = {}
	metadata.update(projects['default'])
	metadata.update(projects[project])
	metadata['white'].update(projects['default']['white'])
	metadata['white'].update(projects[project]['white'])
	imagesets  = metadata["imagesets"]
	if any(x in methods for x in ['kpca','pca','mnf','fica']):
		linearTransformations = True
	else:
		linearTransformations = False
		sigmas = []

	if linearTransformations:
		roi = list(metadata["rois"].keys())[0] # use first named roi (multiple roi in a single pass not yet supported)
		roix = metadata["rois"][roi]["x"]
		roiy = metadata["rois"][roi]["y"]
		roiw = metadata["rois"][roi]["w"]
		roih = metadata["rois"][roi]["h"]
		roilabel = metadata["rois"][roi]["label"]
		if ('mnf' in methods):
			noisesample = list(metadata["noisesamples"].keys())[0] # use first named region of noise
			noisesamplex = metadata["noisesamples"][noisesample]["x"]
			noisesampley = metadata["noisesamples"][noisesample]["y"]
			noisesamplew = metadata["noisesamples"][noisesample]["w"]
			noisesampleh = metadata["noisesamples"][noisesample]["h"]
			noisesamplelabel = metadata["noisesamples"][noisesample]["label"]
			noisestring = 'x'+str(noisesamplex)+'y'+str(noisesampley)+'w'+str(noisesamplew)+'h'+str(noisesampleh)
		histograms = ['equalize','adaptive'] # produce equalized and adaptive histograms, not rescale or none
		multilayer = 'separate files'
		basepathout = basepath	# put it in the same directory in non-interactive mode
		fileformats = ['jpg']

if exists(logfile): 
	print("All information gathered, progress info available in",logfile)
else:
	print("All information gathered, processing underway, if you would like progress info please create a logfile and specify it in jubpaloptions.yaml")

# Summarize Choices
logger.info("Basepath is "+basepath)
logger.info("Project is "+project)
for method in methods:
	logger.info("Process is "+method)
if linearTransformations:
	for sigma in sigmas:
		logger.info("Sigma is "+str(sigma))
	logger.info("nLayers is "+str(n_components))
	for imageset in imagesets:
		logger.info("Imageset is "+imageset)
	logger.info("ROI is "+roi+' '+str(roilabel)+' '+str(roix)+' '+str(roiy)+' '+str(roiw)+' '+str(roih))
	if ('mnf' in methods):
		logger.info("Noise sample is "+noisesample+' '+noisesamplelabel+' '+noisestring)
	for histogram in histograms:
		logger.info("Histogram adjustment is "+histogram)

start = time.time()


msi2xyzFile = checkColorReady()
if msi2xyzFile:
	logger.info("Confirmed ready to process color with msi2xyzFile "+msi2xyzFile)
	processColor(msi2xyzFile)
else:
	logger.info("Not doing color processing")

for sigma in sigmas:
	if not exists(cachepath+'flattened/'): # should only be necessary first run on a machine
		makedirs(cachepath+'flattened/',mode=0o755,exist_ok=True)
	if not exists(cachepath+'denoise/sigma'+str(sigma)+'/'): # should only be necessary first run on a machine
		makedirs(cachepath+'denoise/sigma'+str(sigma)+'/',mode=0o755,exist_ok=True)
	stack, countinput = stacker(sigma)
	# turn image cube into a long rectangle
	nlayers,fullh,fullw = stack.shape
	if n_components == "max":
		n_components = nlayers
	n_components_fica = nlayers
	capture2d = stack.reshape((nlayers,fullw*fullh))
	capture2d = capture2d.transpose()
	# turn region of interest cube into a long rectangle
	roistring = "x"+str(roix)+"y"+str(roiy)+"w"+str(roiw)+"h"+str(roih)
	roi3d = stack[:,roiy:roiy+roih,roix:roix+roiw] # note that y before x
	roi2d = roi3d.reshape((nlayers,roiw*roih))
	roi2d = roi2d.transpose()
	outpath = join(basepathout,project,'Transform/r'+str(countinput)+'bd'+str(sigma))
	outfile = project+'_r'+str(countinput)+'_bd'+str(sigma)
	if ('fica' in methods):
			method = 'fica'
			from sklearn.decomposition import FastICA
			# UserWarning: FastICA did not converge. Consider increasing tolerance or the maximum number of iterations.
			max_iter = fica_max_iter
			tol = fica_tol
			fica = FastICA(n_components=n_components_fica,max_iter=max_iter,tol=tol)
			logger.info("Starting ICA fit with tolerance "+str(tol))
			fica.fit(roi2d)
			logger.info("Starting transform")
			d2_processed = fica.transform(capture2d)
			d2_processed = img_as_float32(d2_processed)
			logger.info("Processed 2d is "+str(d2_processed.shape)+' '+str(d2_processed.dtype))
			d2_processed = d2_processed.transpose()
			logger.info("Transposed to "+str(d2_processed.shape)+' '+str(d2_processed.dtype))
			d3_processed = d2_processed.reshape(n_components_fica,fullh,fullw)
			logger.info("Reshaped to "+str(d3_processed.shape)+' '+str(d3_processed.dtype))
			outpath_fica = join(outpath,method+'_'+roistring)
			outfile_fica = outfile+'_'+method+'_'+roistring
			histogram_adjust(outpath=outpath_fica,outfile=outfile_fica,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components_fica)
			del d2_processed, d3_processed # 3/2/2024 effort to reduce memory errors
	if ('mnf' in methods):
			method = 'mnf'
			from spectral import calc_stats, noise_from_diffs, mnf
			stack = stack.transpose()
			logger.info("Transposed stack has shape"+' '+str(stack.shape)+' '+"and dtype"+' '+str(stack.dtype))
			logger.info("Calculating signal...")
			signal = calc_stats(stack[roix:roix+roiw,roiy:roiy+roih,:])
			logger.info("Calculating noise...")
			noise = noise_from_diffs(stack[noisesamplex:noisesamplex+noisesamplew,noisesampley:noisesampley+noisesampleh,:])
			logger.info("Calculating ratio...")
			mnfr = mnf(signal,noise) # d3_processed = mnf(signal,noise)
			# denoised = mnfr.denoise(stack,snr=10) # not sure this is doing anything
			# d3_processed = mnfr.denoise(stack,snr=10) 
			d3_processed = mnfr.reduce(stack,num=n_components) # no need to create an extra object mnfr d3_processed = d3_processed.reduce(stack,num=n_components)
			d3_processed = d3_processed.transpose()
			d3_processed = img_as_float32(d3_processed)
			logger.info("Reshaped to "+str(d3_processed.shape)+' '+str(d3_processed.dtype))
			outpath_mnf = join(outpath,method+'_'+roistring+'n'+noisestring)
			outfile_mnf = outfile+'_'+method+'_'+roistring+'n'+noisestring
			histogram_adjust(outpath=outpath_mnf,outfile=outfile_mnf,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
			del d3_processed # 3/2/2024 effort to reduce memory errors
	if ('pca' in methods):
			method = 'pca'
			from sklearn.decomposition import PCA
			pca = PCA(n_components=n_components)
			logger.info("Starting fit")
			pca.fit(roi2d)
			logger.info("Starting transform")
			d2_processed = pca.transform(capture2d)
			d2_processed = d2_processed.transpose()
			d3_processed = d2_processed.reshape(n_components,fullh,fullw)
			logger.info("Processed cube is "+str(d3_processed.shape)+' '+str(d3_processed.dtype))
			outpath_pca = join(outpath,method+'_'+roistring)
			outfile_pca = outfile+'_'+method+'_'+roistring
			# jubpalfunctions.histogram_adjust(outpath=outpath_pca,outfile=outfile_pca,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
			histogram_adjust(outpath=outpath_pca,outfile=outfile_pca,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
			del d2_processed, d3_processed # 3/2/2024 effort to reduce memory errors
	if ('cca' in methods):
			method = 'cca'
			logger.info("starting method cca")
			#classification = io.imread("/home/thanneken/Projects/Ambrosiana_C73inf_052/classification.tif") # temporary hard wired
			classification = io.imread("/home/thanneken/Projects/USCAntiphonary/train-32bit-rgb-chip.tif") # temporary hard wired
			classification = io.imread("/home/thanneken/Projects/Ambrosiana_A79inf_001v/train-32bit-rgb.tif")
			classification = img_as_float32(classification)
			#classification = exposure.rescale_intensity(classification) 
			logger.info("Classification is "+str(classification.shape)+' '+str(classification.dtype))
			classh,classw,classlayers = classification.shape
			class2d = classification.reshape(classw*classh,classlayers) 
			#class2d = classification.reshape(classlayers,classw*classh) 
			#class2d = class2d.transpose()
			logger.info("Classification 2d is "+str(class2d.shape)+' '+str(class2d.dtype))
			from sklearn.cross_decomposition import CCA
			cca = CCA(n_components=n_components,max_iter=5000)
			logger.info("Starting fit")
			cca.fit(roi2d,class2d)
			logger.info("Starting transform")
			d2_processed = cca.transform(capture2d)
			logger.info("Processed 2d is "+str(d2_processed.shape)+' '+str(d2_processed.dtype))
			d2_processed = d2_processed.transpose()
			logger.info("Transposed to "+str(d2_processed.shape)+' '+str(d2_processed.dtype))
			d3_processed = d2_processed.reshape(n_components,fullh,fullw)
			logger.info("Reshaped to "+str(d3_processed.shape)+' '+str(d3_processed.dtype))
			outpath_cca = join(outpath,method+'_'+roistring)
			outfile_cca = outfile+'_'+method+'_'+roistring
			# jubpalfunctions.histogram_adjust(outpath=outpath_cca,outfile=outfile_cca,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
			histogram_adjust(outpath=outpath_cca,outfile=outfile_cca,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
	if ('kpca' in methods):
			method = 'kpca'
			from sklearn.decomposition import KernelPCA
			import numpy
			kernel="rbf" # rbf|linear|cosine|sigmoid|poly|precomputed
			eigen_solver="dense" # auto|dense|arpack
			n_jobs=-1 # -1 means all cores
			kpca = KernelPCA(n_components=n_components,kernel=kernel,eigen_solver=eigen_solver,n_jobs=n_jobs)
			logger.info("starting fit")
			kpca.fit(roi2d)
			logger.info("done with fit, starting transform")
			# transform a certain number of lines at a time
			linesatatime = 8 # higher is faster, must be a factor of number of lines
			for x in range(0,fullh,linesatatime): # for each line index (zero to height - 1)
					line_processed = kpca.transform(capture2d[fullw*x:fullw*(x+linesatatime)])
					if x == 0:# for first line
							d2_processed = line_processed
					else: # for subsequent lines
							d2_processed = numpy.concatenate((d2_processed,line_processed))
					logger.info("Processed 2d is "+str(d2_processed.shape)+' '+str(d2_processed.dtype)+' '+str(fullw*x)+' '+str(fullw*(x+linesatatime)))
			logger.info("Processed 2d is "+str(d2_processed.shape)+' '+str(d2_processed.dtype))
			d2_processed = d2_processed.transpose()
			logger.info("Transposed to "+str(d2_processed.shape)+' '+str(d2_processed.dtype))
			d3_processed = d2_processed.reshape(n_components,fullh,fullw)
			logger.info("Reshaped to "+str(d3_processed.shape)+' '+str(d3_processed.dtype))
			outpath_kpca = join(outpath,method+'_'+roistring)
			outfile_kpca = outfile+'_'+method+'_'+roistring
			# jubpalfunctions.histogram_adjust(outpath=outpath_kpca,outfile=outfile_kpca,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
			histogram_adjust(outpath=outpath_kpca,outfile=outfile_kpca,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
	if ('specembed' in methods):
			method = 'specembed'
			from megaman.geometry import Geometry
			from megaman.embedding import (Isomap, LocallyLinearEmbedding, LTSA, SpectralEmbedding)
			radius = 40
			eigen_solver="arpack"
			eigen_solver="dense" 
			eigen_solver="amg" # arpack|dense (amg is default with specembed)
			adjacency_method = 'cyflann'
			adjacency_kwds = {'radius':radius} # ignore distances above this radius
			affinity_method = 'gaussian'
			affinity_kwds = {'radius':radius} # A = exp(-||x - y||/radius^2)
			laplacian_method = 'geometric'
			laplacian_kwds = {'scaling_epps':radius} # scaling ensures convergence to Laplace-Beltrami operator
			geom = Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds,
											affinity_method=affinity_method, affinity_kwds=affinity_kwds,
											laplacian_method=laplacian_method, laplacian_kwds=laplacian_kwds)
			logger.info("Ready to compute adjancency matrix")
			geom.set_data_matrix(roi2d)
			adjacency_matrix = geom.compute_adjacency_matrix()
			logger.info("Computed adjancency matrix, ready to fit_transform")
			spec = SpectralEmbedding(n_components=n_components, eigen_solver=eigen_solver,geom=geom, drop_first=False)
			d2_processed = spec.fit_transform(roi2d)
			logger.info("Processed 2d is "+str(d2_processed.shape)+' '+str(d2_processed.dtype))
			d2_processed = d2_processed.transpose()
			logger.info("Transposed to "+str(d2_processed.shape)+' '+str(d2_processed.dtype))
			d3_processed = d2_processed.reshape(n_components,roih,roiw)
			logger.info("Reshaped to "+str(d3_processed.shape)+' '+str(d3_processed.dtype))
			outpath_specembed = join(outpath,method+'_'+roistring)
			outfile_specembed = outfile+'_'+method+'_'+roistring
			# jubpalfunctions.histogram_adjust(outpath=outpath_specembed,outfile=outfile_specembed,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
			histogram_adjust(outpath=outpath_specembed,outfile=outfile_specembed,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)

# REPORT DURATION
end = time.time()
duration = end - start
logger.info("Completed after "+str(duration)+" seconds")

