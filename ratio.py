#!/usr/bin/env python
from skimage import io, img_as_float32, img_as_uint, img_as_ubyte, exposure
import argparse
import numpy 

parser = argparse.ArgumentParser(prog="ratios.py")
parser.add_argument('-n','--numerator',required=True,help='filename for the numerator')
parser.add_argument('-d','--denominator',required=True,help='filename for the denominator')
parser.add_argument('-a','--adaptive',help='filename for a monochrome ratio with adaptive histogram adjustment')
parser.add_argument('-t','--triple',help='filename for a contrast triple pseudocolor with R=gamma, G=adaptive, B=equalize')
parser.add_argument('-b','--bits',default=8,type=int,help='bits per pixel, defaults to 8')
parser.add_argument('-k','--kernel',default=128,type=int,help='kernel size for adaptive histogram adjustment, defaults to 128')
parser.add_argument('-c','--clip',default=0.1,type=float,help='clip limit for adaptive histogram adjustment, defaults to 0.1')
parser.add_argument('-g','--gamma',default=2.2,type=float,help='gamma adjustment for contrast triples, defaults to 2.2')
args = parser.parse_args()

def contrastTriple(img,gamma,kernel_size,clip_limit):
    gamma = 1 / gamma
    img = exposure.rescale_intensity(img) 
    ch1 = exposure.adjust_gamma(img, 0.45) 
    ch2 = exposure.equalize_adapthist(img,kernel_size=kernel_size,clip_limit=clip_limit) 
    ch3 = exposure.equalize_hist(img) 
    return numpy.stack((ch1,ch2,ch3),2)

def saveImage(img,outfilepath,bpp):
    print("Saving ratio as %s"%(outfilepath))
    if outfilepath.endswith('.jpg') or bpp == 8:
        img = img_as_ubyte(img)
    else:
        img = img_as_uint(img)	
    io.imsave(outfilepath,img)

print("Reading %s"%(args.numerator))
numerator = io.imread(args.numerator)
numerator = img_as_float32(numerator)
print("Reading %s"%(args.denominator))
denominator = io.imread(args.denominator)
denominator = img_as_float32(denominator)
numerator = numerator + 1
denominator = denominator + 1
img = numpy.divide(numerator,denominator)
print("Ratio had shape %s and dtype %s range %s - %s"%(img.shape,img.dtype,numpy.min(img),numpy.max(img)))

if args.adaptive:
    print("Performing adaptive histogram equalization with kernel size %s and clip limit %s"%(args.kernel,args.clip)) 
    adaptive = exposure.rescale_intensity(img) 
    adaptive = exposure.equalize_adapthist(adaptive,kernel_size=args.kernel,clip_limit=args.clip)
    saveImage(adaptive,args.adaptive,args.bits)
if args.triple:
    print("Creating contrast triple of ratio with R based on gamma = %s, G based on adaptive with kernel size %s and clip limit %s, B based on histogram equalization"%(args.gamma,args.kernel,args.clip)) 
    contrastTriple = contrastTriple(img,args.gamma,args.kernel,args.clip)
    saveImage(contrastTriple,args.triple,args.bits)

