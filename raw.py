#!/usr/bin/env python
import rawpy
from skimage import io
import sys
from os import path

if len(sys.argv) < 2:
    exit("This is a simple tool that takes filenames of raw Bayer pattern image files as command line arguments and outputs three components without interpolation")
else:
    half_size=True # each 2x2 block becomes one pixel in each of three channels without interpolation
    no_auto_bright=True # see https://letmaik.github.io/rawpy/api/rawpy.Params.html and https://www.libraw.org/docs/API-datastruct-eng.html
    no_auto_scale=True
    gamma=(1,0) # None for default setting of power = 2.222 and slope = 4.5; (1,0) for linear
    output_bps=16 # 8 or 16 bits per sample
    print("Settings:\n\thalf_size: %s\n\tno_auto_bright: %s\n\tno_auto_scale: %s\n\tgamma (power, slope): %s\n\toutput_bps: %s"%(half_size,no_auto_bright,no_auto_scale,gamma,output_bps))
for infilepath in sys.argv[1:]:
    print("Reading %s"%(infilepath))
    img = rawpy.imread(infilepath)
    io.imsave(infilepath[:-4]+'-veryraw.tif',img.raw_image.copy(),check_contrast=False)
    img = img.postprocess(half_size=half_size,no_auto_bright=no_auto_bright,gamma=gamma,no_auto_scale=no_auto_scale,output_bps=output_bps)
    height,width,channels = img.shape
    print("Processed image is %s pixels high, %s pixels wide, and %s channels deep with each pixel described with %s data"%(height,width,channels,img.dtype))
    for channel in range(channels):
        outfilepath = infilepath[:-4]+'-c'+str(channel)+'.tif'
        if path.exists(outfilepath):
            print("Overwriting %s"%(outfilepath))
        else:
            print("Saving %s"%(outfilepath))
        io.imsave(outfilepath,img[:,:,channel],check_contrast=False)

