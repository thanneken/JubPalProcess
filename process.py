#!/home/thanneken/python/miniconda3/bin/python
import multiprocessing
from skimage import io, img_as_float32, img_as_uint, img_as_ubyte, filters, exposure
from os import listdir, makedirs
from os.path import exists, join
import time 
import yaml 
import inquirer
# import jubpalfunctions ## just put all the functions in this file, avoids having to send variables
import numpy 
import rawpy
import pyexifinfo
import logging
import sys

# DEFINE OUR FUNCTIONS
def nextNeededProject(projects):
	matches = (f for f in projects if exists(basepath+f) and not exists(basepath+f+'/Transform'))
	return next(matches,None)
def blurdivide(img,sigma):
	if not img.dtype == "float32":
		img = img_as_float32(img)
	# print("Creating a numerator as 3x3 median")
	numerator = filters.median(img) # default is 3x3, same as RLE suggested
	# print("Creating a denominator with radius/sigma = 50 Gaussian blur on denominator (RLE does 101x101 box blur)")
	denominator = filters.gaussian(img,sigma=sigma)
	ratio = numerator / denominator
	return ratio
def flattenrotate(fullpath):
		file = fullpath.split('/')[-1][:-4] # filename is everything after the last forward slash, and remove the extension too
		if exists(cachepath+'flattened/'+file+'.tif'): # check cache
				logger.info("Found in cache: flattened/"+file+'.tif')
				flattenedfloat = io.imread(cachepath+'flattened/'+file+'.tif')
		else:
				with rawpy.imread(fullpath) as raw:
						capture = raw.raw_image.copy()
				exif = pyexifinfo.get_json(fullpath)
				exifflat = exif[0]["IPTC:Keywords"][11] # was index 11 n array of keywords in 2017, likely to be different in 2023
				if exifflat.endswith('.dn'):
					exifflat = exifflat+'g' # the last letter got cut off in 2017, likely to be different in 2023
				exiforientation = exif[0]["EXIF:Orientation"]
				flatsdir = jubpaloptions["projects"][project]["flats"] 
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
				# np.divide(a, b, out=np.zeros_like(a), where=b!=0)
				flattenedfloat = numpy.divide(capture*numpy.average(flat),flat,out=numpy.zeros_like(capture*numpy.average(flat)),where=flat!=0)
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
def readnblur(q,fullpath,sigma):
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
						img = blurdivide(img,sigma)
						img = exposure.rescale_intensity(img)
						makedirs(cachepath+'denoise/sigma'+str(sigma)+'/',mode=0o755,exist_ok=True)
						io.imsave(cachepath+'denoise/sigma'+str(sigma)+'/'+file+'.tif',img,check_contrast=False)
		q.put(img)
#def stacker(basepath,project,imagesets,sigma,skipuvbp,cachepath):
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
			p = multiprocessing.Process(target=readnblur,args=(q,fullpath,sigma))
			processes.append(p)
			p.start()
	for process in processes: 
		stack.append(q.get())
	stack = numpy.array(stack)
	p.join() # added 6/17/2023
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

if interactive == True:
	## select one basepath
	if len(jubpaloptions["basepaths"]) > 1:
		questions = [inquirer.List('basepath',"Select basepath for source data",choices=jubpaloptions["basepaths"])]
		selections = inquirer.prompt(questions)
		basepath = selections["basepath"]
	else:
		basepath = jubpaloptions["basepaths"][0]
	## select one project
	if len(jubpaloptions["projects"].keys()) > 1:
		questions = [ inquirer.List('project',"Select project",choices=jubpaloptions["projects"].keys()) ]
		selections = inquirer.prompt(questions)
		project = selections["project"]
	else:
		project = jubpaloptions["projects"].keys()[0]
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
	## select one number of components
	if len(jubpaloptions["options"]["n_components"]) > 1:
		questions = [ inquirer.List('n_components',"How many components to generate?",choices=jubpaloptions["options"]["n_components"]) ]
		selections = inquirer.prompt(questions)
		n_components = selections["n_components"]
	else:
		n_components = jubpaloptions["options"]["n_components"][0]
	## select one or more image sets
	if len(jubpaloptions["projects"][project]["imagesets"]) > 1:
		questions = [ inquirer.Checkbox('imagesets',"Select one or more image sets",choices=jubpaloptions["projects"][project]["imagesets"]) ]
		imagesets = []
		while len(imagesets) < 1:
			selections = inquirer.prompt(questions)
			imagesets = selections["imagesets"]
	else:
		imagesets = jubpaloptions["projects"][project]["imagesets"]
	## select one roi, eventually one or more
	if len(jubpaloptions["projects"][project]["rois"].keys()) > 1:
		questions = [
			inquirer.List('roi',"Select region of interest",choices=jubpaloptions["projects"][project]["rois"].keys())
		]
		selections = inquirer.prompt(questions)
		roi = selections["roi"]
	else:
		roi = list(jubpaloptions["projects"][project]["rois"].keys())[0] 
	roix = jubpaloptions["projects"][project]["rois"][roi]["x"]
	roiy = jubpaloptions["projects"][project]["rois"][roi]["y"]
	roiw = jubpaloptions["projects"][project]["rois"][roi]["w"]
	roih = jubpaloptions["projects"][project]["rois"][roi]["h"]
	roilabel = jubpaloptions["projects"][project]["rois"][roi]["label"]
	## select noise sample for mnf 
	if ('mnf' in methods):
			if len(jubpaloptions["projects"][project]["noisesamples"].keys()) > 1:
				questions = [
					inquirer.List('noisesample',"Select Noise Region",choices=jubpaloptions["projects"][project]["noisesamples"].keys())
				]
				selections = inquirer.prompt(questions)
				noisesample = selections["noisesample"]
			else:
				noisesample = list(jubpaloptions["projects"][project]["noisesamples"].keys())[0] 
			noisesamplex = jubpaloptions["projects"][project]["noisesamples"][noisesample]["x"]
			noisesampley = jubpaloptions["projects"][project]["noisesamples"][noisesample]["y"]
			noisesamplew = jubpaloptions["projects"][project]["noisesamples"][noisesample]["w"]
			noisesampleh = jubpaloptions["projects"][project]["noisesamples"][noisesample]["h"]
			noisesamplelabel = jubpaloptions["projects"][project]["noisesamples"][noisesample]["label"]
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
	## select one output path
	if len(jubpaloptions["basepaths"]) > 1:
		questions = [ inquirer.List('basepathout',"Select basepath for output (project name is implicit)",choices=jubpaloptions["basepaths"]) ]
		selections = inquirer.prompt(questions)
		basepathout = selections["basepathout"] 
	else:
		basepathout = jubpaloptions["basepaths"][0]
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
	project = nextNeededProject(jubpaloptions["projects"].keys())
	if project == None:
		print("No projects defined in",datafile,"in the first named basepath lack a Transform directory")
		logger.info("No projects defined in "+datafile+" in the first named basepath lack a Transform directory")
		exit()
	sigmas = jubpaloptions["options"]["sigmas"] # all listed sigmas
	skipuvbp = True # skip files with filter distortion
	# methods = 'fica'
	methods = jubpaloptions["options"]["methods"] # all listed methods
	n_components = jubpaloptions["options"]["n_components"][0] # first named number of components (max)
	imagesets = jubpaloptions["projects"][project]["imagesets"] # use all image sets listed in the options file
	roi = list(jubpaloptions["projects"][project]["rois"].keys())[0] # use first named roi (multiple roi in a single pass not yet supported)
	roix = jubpaloptions["projects"][project]["rois"][roi]["x"]
	roiy = jubpaloptions["projects"][project]["rois"][roi]["y"]
	roiw = jubpaloptions["projects"][project]["rois"][roi]["w"]
	roih = jubpaloptions["projects"][project]["rois"][roi]["h"]
	roilabel = jubpaloptions["projects"][project]["rois"][roi]["label"]
	if ('mnf' in methods):
		noisesample = list(jubpaloptions["projects"][project]["noisesamples"].keys())[0] # use first named region of noise
		noisesamplex = jubpaloptions["projects"][project]["noisesamples"][noisesample]["x"]
		noisesampley = jubpaloptions["projects"][project]["noisesamples"][noisesample]["y"]
		noisesamplew = jubpaloptions["projects"][project]["noisesamples"][noisesample]["w"]
		noisesampleh = jubpaloptions["projects"][project]["noisesamples"][noisesample]["h"]
		noisesamplelabel = jubpaloptions["projects"][project]["noisesamples"][noisesample]["label"]
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
for sigma in sigmas:
	logger.info("Sigma is "+str(sigma))
for method in methods:
	logger.info("Process is "+method)
logger.info("nLayers is "+str(n_components))
for imageset in imagesets:
	logger.info("Imageset is "+imageset)
logger.info("ROI is "+roi+' '+str(roilabel)+' '+str(roix)+' '+str(roiy)+' '+str(roiw)+' '+str(roih))
if ('mnf' in methods):
	logger.info("Noise sample is "+noisesample+' '+noisesamplelabel+' '+noisestring)
for histogram in histograms:
	logger.info("Histogram adjustment is "+histogram)

# jubpalfunctions.jubpaloptions = jubpaloptions
# jubpalfunctions.project = project
# jubpalfunctions.basepath = basepath
# jubpalfunctions.imagesets = imagesets
# jubpalfunctions.skipuvbp = skipuvbp
# jubpalfunctions.cachepath = cachepath
# jubpalfunctions.logger = logger

start = time.time()
for sigma in sigmas:
	#stack, countinput = jubpalfunctions.stacker(basepath,project,imagesets,sigma,skipuvbp,cachepath)
	# stack, countinput = jubpalfunctions.stacker(sigma)
	stack, countinput = stacker(sigma)
	# turn image cube into a long rectangle
	nlayers,fullh,fullw = stack.shape
	if n_components == "max":
		n_components = nlayers
	capture2d = stack.reshape((nlayers,fullw*fullh))
	capture2d = capture2d.transpose()
	# turn region of interest cube into a long rectangle
	roistring = "x"+str(roix)+"y"+str(roiy)+"w"+str(roiw)+"h"+str(roih)
	roi3d = stack[:,roiy:roiy+roih,roix:roix+roiw] # note that y before x
	roi2d = roi3d.reshape((nlayers,roiw*roih))
	roi2d = roi2d.transpose()
	outpath = join(basepathout,project,'Transform/r'+str(countinput)+'bd'+str(sigma))
	outfile = project+'_r'+str(countinput)+'_bd'+str(sigma)
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
			if n_components > 12: # limit mnf to 12 in effort to address memory problem
				logger.info("Limiting MNF to 12 components")
				n_components == 20
			d3_processed = mnfr.reduce(stack,num=n_components) # no need to create an extra object mnfr d3_processed = d3_processed.reduce(stack,num=n_components)
			d3_processed = d3_processed.transpose()
			d3_processed = img_as_float32(d3_processed)
			logger.info("Reshaped to "+str(d3_processed.shape)+' '+str(d3_processed.dtype))
			outpath_mnf = join(outpath,method+'_'+roistring+'n'+noisestring)
			outfile_mnf = outfile+'_'+method+'_'+roistring+'n'+noisestring
			# jubpalfunctions.histogram_adjust(outpath=outpath_mnf,outfile=outfile_mnf,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
			histogram_adjust(outpath=outpath_mnf,outfile=outfile_mnf,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
	if ('fica' in methods):
			method = 'fica'
			from sklearn.decomposition import FastICA
			# UserWarning: FastICA did not converge. Consider increasing tolerance or the maximum number of iterations.
			max_iter = fica_max_iter
			tol = fica_tol
			fica = FastICA(n_components=n_components,max_iter=max_iter,tol=tol)
			logger.info("Starting ICA fit with tolerance "+str(tol))
			fica.fit(roi2d)
			logger.info("Starting transform")
			d2_processed = fica.transform(capture2d)
			d2_processed = img_as_float32(d2_processed)
			logger.info("Processed 2d is "+str(d2_processed.shape)+' '+str(d2_processed.dtype))
			d2_processed = d2_processed.transpose()
			logger.info("Transposed to "+str(d2_processed.shape)+' '+str(d2_processed.dtype))
			d3_processed = d2_processed.reshape(n_components,fullh,fullw)
			logger.info("Reshaped to "+str(d3_processed.shape)+' '+str(d3_processed.dtype))
			outpath_fica = join(outpath,method+'_'+roistring)
			outfile_fica = outfile+'_'+method+'_'+roistring
			# jubpalfunctions.histogram_adjust(outpath=outpath_fica,outfile=outfile_fica,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
			histogram_adjust(outpath=outpath_fica,outfile=outfile_fica,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
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

