# @Author: Atul Sahay <atul>
# @Date:   2018-11-22T14:28:53+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   nilanjan
# @Last modified time: 2018-11-22T19:01:37+05:30

import os
import matplotlib.pyplot as plt
# from scipy import signal
# from scipy.io import wavfile
import numpy as np
import Enhancer as enh


# Note : First Create a folder name [] under the  directory where the python
#        file is kept

def image_enhancement(filename,dlen):
    #Output for gamma_correction
    # OUTPUT = "gamma_corrected_train/"+filename[dLen+1:-4]+".png"
    # print("OUTPUT : ",OUTPUT)
    # enh.gamma_correction(filename,OUTPUT,0.6)
    OUTPUT = "adaptive_sharpened_train/"+filename[dLen+1:-4]+".png"
    print("OUTPUT : ",OUTPUT)
    enh.sharpness(filename,OUTPUT)




directory = 'validation'
dLen = len(directory)

for filename in os.listdir(directory):
    print("filename"+str(filename))
    if filename.endswith(".png"):
        # print("dir: "+str(directory))
        path = directory+"/"+filename
        # print("path "+str(path))
        image_enhancement(path,dLen)
    else:
        pass
