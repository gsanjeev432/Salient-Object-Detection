# read, display, and save the images
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# for color space conversion
from skimage.color import rgb2lab
# for some simple profiling
import time
from imutils import paths
import cv2
import numpy as np
# the actual functions
from saliencyMap import getSaliencyMapNumpy
import os
from matplotlib import interactive
import morphsnakes

path = "D:\\Sanjeev\\Saliency Object Detection\\SaliencyMap\\MSRA-B\\images\\"
outPath = "D:\\Sanjeev\\Saliency Object Detection\\SaliencyMap\\a"

def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]


def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum(grid.T**2, 0))
    u = np.float_(phi > 0)
    return u




def main():
    for image_path in os.listdir(path):
        input_path = os.path.join(path, image_path)
        # read image
        rgbImage = mpimg.imread(input_path)
    
        # convert to lab
        labImage = rgb2lab(rgbImage)
        # TODO: Matlab scales/shifts values, so we do the same in order to compare results
        labImage[:, :, 0] = labImage[:, :, 0] * 2.55
        labImage[:, :, 1] = labImage[:, :, 1] + 128
        labImage[:, :, 2] = labImage[:, :, 2] + 128
    
        start = time.clock()
        # calculate saliency map
        sm3 = getSaliencyMapNumpy(labImage)
        end = time.clock()
        print ("getSaliencyMapNumpy() took", (end - start), " seconds")
    
        output = sm3
    
        # save output
        fullpath = os.path.join(outPath, 'crop'+image_path)
        mpimg.imsave(fullpath, output)
        print('Calculating Active Contour from Saliency Map')
#        os.system("python tests.py")
    #    os.system("python spectral.py")



if __name__ == "__main__":
    main()
