#!/home/thanneken/python/miniconda3/bin/python
from skimage import io, img_as_float32
from os.path import join
import time 
import yaml 
import inquirer
import jubpalfunctions

# GATHER INPUT
datafile = 'jubpaloptions.yaml'
print("Reading options from",datafile)
with open(datafile,'r') as unparsedyaml:
	jubpaloptions = yaml.load(unparsedyaml,Loader=yaml.SafeLoader)
## read non-interactive options
cachepath = jubpaloptions["settings"]["cachepath"]
fica_max_iter = jubpaloptions["settings"]["fica_max_iter"]
fica_tol = jubpaloptions["settings"]["fica_tol"]
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

# Summarize Choices
print("Basepath is",basepath)
print("Project is",project)
for sigma in sigmas:
	print("Sigma is",sigma)
for method in methods:
	print("Process is",method)
print("nLayers is",n_components)
for imageset in imagesets:
	print("Imageset is",imageset)
print("ROI is",roi,roilabel,roix,roiy,roiw,roih)
if ('mnf' in methods):
	print("Noise sample is",noisesample,noisesamplelabel,noisesamplex,noisesampley,noisesamplew,noisesampleh)
for histogram in histograms:
	print("Histogram adjustment is",histogram)

start = time.time()
for sigma in sigmas:
	stack, countinput = jubpalfunctions.stacker(basepath,project,imagesets,sigma,skipuvbp,cachepath)
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
	outpath = join(basepathout,project,'r'+str(countinput)+'bd'+str(sigma))
	outfile = project+'_r'+str(countinput)+'_bd'+str(sigma)
	if ('pca' in methods):
			method = 'pca'
			from sklearn.decomposition import PCA
			pca = PCA(n_components=n_components)
			print("Starting fit")
			pca.fit(roi2d)
			print("Starting transform")
			d2_processed = pca.transform(capture2d)
			d2_processed = d2_processed.transpose()
			d3_processed = d2_processed.reshape(n_components,fullh,fullw)
			print("Processed cube is",d3_processed.shape,d3_processed.dtype)
			outpath_pca = join(outpath,method+'_'+roistring)
			outfile_pca = outfile+'_'+method+'_'+roistring
			jubpalfunctions.histogram_adjust(outpath=outpath_pca,outfile=outfile_pca,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
	if ('mnf' in methods):
			method = 'mnf'
			from spectral import calc_stats, noise_from_diffs, mnf
			stack = stack.transpose()
			print("Transposed stack has shape",stack.shape,"and dtype",stack.dtype)
			print("Calculating signal...")
			signal = calc_stats(stack[roix:roix+roiw,roiy:roiy+roih,:])
			print("Calculating noise...")
			noise = noise_from_diffs(stack[noisesamplex:noisesamplex+noisesamplew,noisesampley:noisesampley+noisesampleh,:])
			print("Calculating ratio...")
			mnfr = mnf(signal,noise)
			# denoised = mnfr.denoise(stack,snr=10) # not sure this is doing anything
			# d3_processed = mnfr.denoise(stack,snr=10) 
			d3_processed = mnfr.reduce(stack,num=n_components)
			d3_processed = d3_processed.transpose()
			d3_processed = img_as_float32(d3_processed)
			print("Reshaped to",d3_processed.shape,d3_processed.dtype)
			outpath_mnf = join(outpath,method+'_'+roistring+'n'+noisestring)
			outfile_mnf = outfile+'_'+method+'_'+roistring+'n'+noisestring
			jubpalfunctions.histogram_adjust(outpath=outpath_mnf,outfile=outfile_mnf,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
	if ('fica' in methods):
			method = 'fica'
			from sklearn.decomposition import FastICA
			# UserWarning: FastICA did not converge. Consider increasing tolerance or the maximum number of iterations.
			max_iter = fica_max_iter
			tol = fica_tol
			fica = FastICA(n_components=n_components,max_iter=max_iter,tol=tol)
			print("Starting fit")
			fica.fit(roi2d)
			print("Starting transform")
			d2_processed = fica.transform(capture2d)
			d2_processed = img_as_float32(d2_processed)
			print("Processed 2d is",d2_processed.shape,d2_processed.dtype)
			d2_processed = d2_processed.transpose()
			print("Transposed to",d2_processed.shape,d2_processed.dtype)
			d3_processed = d2_processed.reshape(n_components,fullh,fullw)
			print("Reshaped to",d3_processed.shape,d3_processed.dtype)
			outpath_fica = join(outpath,method+'_'+roistring)
			outfile_fica = outfile+'_'+method+'_'+roistring
			jubpalfunctions.histogram_adjust(outpath=outpath_fica,outfile=outfile_fica,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
	if ('cca' in methods):
			method = 'cca'
			print("starting method cca")
			#classification = io.imread("/home/thanneken/Projects/Ambrosiana_C73inf_052/classification.tif") # temporary hard wired
			classification = io.imread("/home/thanneken/Projects/USCAntiphonary/train-32bit-rgb-chip.tif") # temporary hard wired
			classification = io.imread("/home/thanneken/Projects/Ambrosiana_A79inf_001v/train-32bit-rgb.tif")
			classification = img_as_float32(classification)
			#classification = exposure.rescale_intensity(classification) 
			print("Classification is ",classification.shape,classification.dtype)
			classh,classw,classlayers = classification.shape
			class2d = classification.reshape(classw*classh,classlayers) 
			#class2d = classification.reshape(classlayers,classw*classh) 
			#class2d = class2d.transpose()
			print("Classification 2d is ",class2d.shape,class2d.dtype)
			from sklearn.cross_decomposition import CCA
			cca = CCA(n_components=n_components,max_iter=5000)
			print("Starting fit")
			cca.fit(roi2d,class2d)
			print("Starting transform")
			d2_processed = cca.transform(capture2d)
			print("Processed 2d is",d2_processed.shape,d2_processed.dtype)
			d2_processed = d2_processed.transpose()
			print("Transposed to",d2_processed.shape,d2_processed.dtype)
			d3_processed = d2_processed.reshape(n_components,fullh,fullw)
			print("Reshaped to",d3_processed.shape,d3_processed.dtype)
			outpath_cca = join(outpath,method+'_'+roistring)
			outfile_cca = outfile+'_'+method+'_'+roistring
			jubpalfunctions.histogram_adjust(outpath=outpath_cca,outfile=outfile_cca,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
	if ('kpca' in methods):
			method = 'kpca'
			from sklearn.decomposition import KernelPCA
			import numpy
			kernel="rbf" # rbf|linear|cosine|sigmoid|poly|precomputed
			eigen_solver="dense" # auto|dense|arpack
			n_jobs=-1 # -1 means all cores
			kpca = KernelPCA(n_components=n_components,kernel=kernel,eigen_solver=eigen_solver,n_jobs=n_jobs)
			print("starting fit")
			kpca.fit(roi2d)
			print("done with fit, starting transform")
			# transform a certain number of lines at a time
			linesatatime = 8 # higher is faster, must be a factor of number of lines
			for x in range(0,fullh,linesatatime): # for each line index (zero to height - 1)
					line_processed = kpca.transform(capture2d[fullw*x:fullw*(x+linesatatime)])
					if x == 0:# for first line
							d2_processed = line_processed
					else: # for subsequent lines
							d2_processed = numpy.concatenate((d2_processed,line_processed))
					print("Processed 2d is",d2_processed.shape,d2_processed.dtype,fullw*x,fullw*(x+linesatatime))
			print("Processed 2d is",d2_processed.shape,d2_processed.dtype)
			d2_processed = d2_processed.transpose()
			print("Transposed to",d2_processed.shape,d2_processed.dtype)
			d3_processed = d2_processed.reshape(n_components,fullh,fullw)
			print("Reshaped to",d3_processed.shape,d3_processed.dtype)
			outpath_kpca = join(outpath,method+'_'+roistring)
			outfile_kpca = outfile+'_'+method+'_'+roistring
			jubpalfunctions.histogram_adjust(outpath=outpath_kpca,outfile=outfile_kpca,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
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
			print("Ready to compute adjancency matrix")
			geom.set_data_matrix(roi2d)
			adjacency_matrix = geom.compute_adjacency_matrix()
			print("Computed adjancency matrix, ready to fit_transform")
			spec = SpectralEmbedding(n_components=n_components, eigen_solver=eigen_solver,geom=geom, drop_first=False)
			d2_processed = spec.fit_transform(roi2d)
			print("Processed 2d is",d2_processed.shape,d2_processed.dtype)
			d2_processed = d2_processed.transpose()
			print("Transposed to",d2_processed.shape,d2_processed.dtype)
			d3_processed = d2_processed.reshape(n_components,roih,roiw)
			print("Reshaped to",d3_processed.shape,d3_processed.dtype)
			outpath_specembed = join(outpath,method+'_'+roistring)
			outfile_specembed = outfile+'_'+method+'_'+roistring
			jubpalfunctions.histogram_adjust(outpath=outpath_specembed,outfile=outfile_specembed,histograms=histograms,d3_processed=d3_processed,fileformats=fileformats,multilayer=multilayer,n_components=n_components)

# REPORT DURATION
end = time.time()
duration = end - start
print("Completed after ",duration," seconds")

