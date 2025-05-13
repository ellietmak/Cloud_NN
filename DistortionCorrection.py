###############################################################################################
#Contact roktem@lbl.gov for questions
#Author: modified by Rusen Oktem from opencv examples
#Date : December 2022
#Usage: python DistortionCorrection.py config_file_name file_prefix
#---Reads intrinsic parameters (image size, camera matrix, and distortion coefficients) 
#from config_file (xml format)
#---Reads list of jpeg images from the path defined by file_prefix 
#(i.e. files in the form file_prefix.jpg)
#---Writes the distortion corrected output to the file named file_prefix_c.jpg
#########################################################################################

import numpy as np
import cv2 as cv
import glob
import sys
import os
import re
from lxml import objectify


#Read calibration parameters from the config file
def ReadConfigFile(in_file):
    print ('\nReading parameters from configuration file....')
    with open(in_file, 'r') as f:
            xml = f.read()
    root = objectify.fromstring(xml)
    width = root.image_Width
    height = root.image_Height

    cameraMat = xml_to_numpy(root.Camera_Matrix)
    distCoeffs = xml_to_numpy(root.Distortion_Coefficients)
    sizeim = (int(width), int(height))
    print('read config file')
    return sizeim, cameraMat, distCoeffs
    

#Extract parameters from xml file
def xml_to_numpy(xml):
    xml_data_str = re.split(r'[\s]+', str(xml.data))
    xml_data_str = [s for s in xml_data_str if s] # remove blank values
    out = np.array(xml_data_str, dtype=float).reshape((xml.rows, xml.cols))
    print('done xml to np')
    return out 

#Use opencv libraries to generate distortion correction maps from camera parameters
def GenerateCorrectionMaps(sizeim, cameraMatrix, distCoeffs):
    out , _= cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, sizeim, 1, sizeim, 0)
    map1, map2 = cv.initUndistortRectifyMap(cameraMatrix, distCoeffs, np.array([]), out, sizeim, cv.CV_16SC2)
    print('generated correction map')
    return map1, map2

#USe distortion correction maps and opencv remap function to apply distortion correction
def CorrectAFrame(img, map1, map2):
        print(img)
        temp = img.copy()
        print('corrected aframe')
        return cv.remap(temp, map1, map2, cv.INTER_LINEAR)


################MAIN BODY#########
#################################################################################
def main():
    #calibration paths
    calibration_path = {'/Users/ellie/Desktop/cloud_nn/data/s5_set': '/Users/ellie/Desktop/cloud_nn/config_files/calibrationOutHOUa.xml',
               '/Users/ellie/Desktop/cloud_nn/data/e43_set': '/Users/ellie/Desktop/cloud_nn/config_files/calibrationOutSGP43a.xml',
               '/Users/ellie/Desktop/cloud_nn/data/e44_set': '/Users/ellie/Desktop/cloud_nn/config_files/calibrationOutSGP44a.xml',
               '/Users/ellie/Desktop/cloud_nn/data/e45_set': '/Users/ellie/Desktop/cloud_nn/config_files/calibrationOutSGP45a.xml'}

    
    path = '/Users/ellie/Desktop/cloud_nn/data/'
    data_path = os.listdir(path)
  

    # total = 0

    for image_folder in data_path:

        #skip images at COR due to not having the calibration file
        # if image_folder == 'COR': 
        #     continue
        if 'corrected' in image_folder:
            continue
       
        print('image_folder:', image_folder)
        tmp_path = path + image_folder
        print('tmp_path:', tmp_path, '\n')

        confFile = calibration_path[tmp_path]
        print('confile:', confFile)
        
        imsize, cameraMatrix, distCoeffs = ReadConfigFile(confFile)
        newmat = cv.getOptimalNewCameraMatrix(cameraMatrix,distCoeffs,imsize,1,imsize)
        print("Camera Matrix: ", newmat) 
        #Generate distortion correction maps
        map1, map2 = GenerateCorrectionMaps(imsize,cameraMatrix,distCoeffs)
        
        count = 0

        for image in os.listdir(tmp_path):
            # if count > 0:
            #      count -= 1
            #      continue
            # else:
            #      count = 20
            #      total += 1
            #      print(f'Number of images: {total}')
                 
            image_path = tmp_path + '/' + image
            print(f'Working on {image_path}...')
            
            #Read an image from the list
            inimg = cv.imread(image_path)

            #Apply distortion correction
            outimg = CorrectAFrame(inimg,map1,map2)
            print('applied distortion correction')
            
            #downsize by 2^2
            ncol = outimg.shape[1]>>2
            nrow = outimg.shape[0]>>2
            outimg_ds = cv.resize(outimg, (ncol, nrow), interpolation=cv.INTER_CUBIC)
            print('downsize by 2 ^ 2')
            
            #write to output file
            out_path = '/Users/ellie/Desktop/cloud_nn/data/' + image_folder + '_corrected/'
            print('out_path:', out_path)
            
            full_path = out_path + image[:-4] + '_c.jpg'
            print(full_path)

            cv.imwrite(full_path, outimg_ds)
            if not cv.imwrite(full_path, outimg_ds):
                 raise Exception('not writing')


    print('All done.....')
    

main()
