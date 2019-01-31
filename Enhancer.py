# @Author: Atul Sahay <atul>
# @Date:   2018-11-22T14:22:47+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2018-11-22T19:23:03+05:30

import cv2
import numpy as np
import PIL as p
from PIL import ImageEnhance

def butter_lowpass(cutoff, fs, order=5):
     nyq = 0.5 * fs
     normal_cutoff = cutoff / nyq
     b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
     return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
     b, a = butter_lowpass(cutoff, fs, order=order)
     y = signal.lfilter(b, a, data)
     return y

def sharpness_value(PILimage):
    gray = PILimage.convert('L') # grayscale
    pixels = np.asarray(gray, dtype=np.int32)
    gy, gx = np.gradient(pixels)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness_messure = np.average(gnorm)

    return sharpness_messure

def sharpness(ImageName,OUTPUT):
    image = p.Image.open(ImageName)
    enhancer = ImageEnhance.Sharpness(image)
    factor = 11 # how much shaprness we needed
    ## TODO:  Will make it adaptive
    factor/=sharpness_value(image)
    enhancedImage = enhancer.enhance(factor)
    enhancedImage.save(OUTPUT)


def gamma_correction(ImageName,OUTPUT,sensitivity=0.4):
    image_v = cv2.imread(ImageName)
    image_v = image_v/255.0
    gamma_corrected_image = cv2.pow(image_v,sensitivity)
    gamma_corrected_image*=255
    gamma_corrected_image = gamma_corrected_image.astype(np.uint8)
    # cv2.imshow('Original Image',image_v)
    # cv2.imshow('Power Law Transformation',gamma_corrected_image)
    # cv2.waitKey(0)
    cv2.imwrite(OUTPUT,gamma_corrected_image)
