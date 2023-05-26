#!/home/thanneken/python/miniconda3/bin/python
import multiprocessing
from os import listdir, makedirs
from os.path import exists, join
from skimage import io, img_as_float32, img_as_uint, img_as_ubyte, filters, exposure
import numpy 
import rawpy
import pyexifinfo
import logging

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
            flattenedfloat = numpy.rot90(flattenedfloat,k=2)
        elif exiforientation == "Rotate 90 CCW":
            flattenedfloat = numpy.rot90(flattenedfloat)
        elif exiforientation == "Rotate 270 CW":
            flattenedfloat = numpy.rot90(flattenedfloat)
        # save flat to cache
        flattenedfloat = img_as_float32(flattenedfloat)
        makedirs(cachepath+'flattened/',mode=0o755,exist_ok=True)
        io.imsave(cachepath+'flattened/'+file+'.tif',flattenedfloat)
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
            io.imsave(cachepath+'denoise/sigma'+str(sigma)+'/'+file+'.tif',img)
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
	return stack, countinput
# EXPOSURE
def histogram_adjust(outpath,outfile,histograms,d3_processed,fileformats,multilayer,n_components):
	processes = []
	for histogram in histograms:
		p = multiprocessing.Process(target=histogram_adjust_thread,args=(outpath,outfile,histogram,d3_processed,fileformats,multilayer,n_components))
		processes.append(p)
		p.start()
def histogram_adjust_thread(outpath,outfile,histogram,d3_processed,fileformats,multilayer,n_components):
	if histogram == 'equalize':
		logger.info("Performing histogram equalization")
		adjusted_eq = d3_processed
		for i in range (0,n_components):
			adjusted_eq[i,:,:]  = exposure.equalize_hist(adjusted_eq[i,:,:])
		outpath_h = join(outpath,histogram)
		outfile_h = outfile+'_'+histogram
		save_all_formats(adjusted=adjusted_eq,histogram=histogram,outpath=outpath_h,outfile=outfile_h,fileformats=fileformats,multilayer=multilayer,n_components=n_components)
	elif histogram == 'rescale':
		logger.info("Performing histogram rescale")
		adjusted_rs = d3_processed
		for i in range (0,n_components):
			adjusted_rs[i,:,:]  = exposure.rescale_intensity(adjusted_rs[i,:,:])
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
			io.imsave(outfile_current,img32)
		elif n_components == 1:
			outfile_current = outfile_current+'.'+fileformat
			component = adjusted[0,:,:]
			logger.info("Saving "+outfile_current)
			if fileformat == 'tif':
				img32 = img_as_float32(component)
				io.imsave(outfile_current,img32)
			elif fileformat == 'png':
				img16 = img_as_uint(component)
				io.imsave(outfile_current,img16)
			elif fileformat == 'jpg':
				img8 = img_as_ubyte(component)
				io.imsave(outfile_current,img8)
		elif multilayer == 'separate files':
			for i in range (0,n_components):
				outfilec = outfile_current+'_c'+str(f"{i:02d}")+'.'+fileformat
				logger.info("Saving "+outfilec)
				component = adjusted[i,:,:]
				if fileformat == 'tif':
					img32 = img_as_float32(component)
					io.imsave(outfilec,img32)
				elif fileformat == 'png':
					img16 = img_as_uint(component)
					io.imsave(outfilec,img16)
				elif (fileformat=='jpg' and histogram=='none'):
					logger.warn("Can't save floating point to jpeg without at least some histogram adjustment")
				elif fileformat == 'jpg':
					img8 = img_as_ubyte(component)
					io.imsave(outfilec,img8)
