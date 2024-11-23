#!/usr/bin/env python
from skimage import io, img_as_float32, img_as_uint, img_as_ubyte, exposure
from sys import argv
import numpy 

kernel_size = 128
clip_limit = 0.1

if len(argv) < 4: 
	print("This command requires three arguments.\nThe first is the filename of the numerator.\nThe second filename of the denominator.\nThe third is the name of the file to output.\nChanges to adaptive histogram equalization must be made in the code.")
	exit()
numerator = argv[1]
denominator = argv[2]
outfilepath = argv[3]
print("Reading %s"%(numerator))
numerator = io.imread(numerator)
numerator = img_as_float32(numerator)
print("Reading %s"%(denominator))
denominator = io.imread(denominator)
denominator = img_as_float32(denominator)
numerator = numerator + 1
denominator = denominator + 1
img = numpy.divide(numerator,denominator)

print("Ratio had shape %s and dtype %s range %s-%s"%(img.shape,img.dtype,numpy.min(img),numpy.max(img)))

print("Performing adaptive histogram equalization with kernel size %s and clip limit %s"%(kernel_size,clip_limit)) 
img = exposure.rescale_intensity(img) 
img = exposure.equalize_adapthist(img,kernel_size=kernel_size,clip_limit=clip_limit)

print("Saving ratio as %s"%(outfilepath))
if outfilepath.endswith('.tif') or outfilepath.endswith('.png'):
	img = img_as_uint(img)	
elif outfilepath.endswith('.jpg'):
	img = img_as_ubyte(img)
else:
	print("Not sure about file format for %s"%(outfilepath))
io.imsave(outfilepath,img)

