from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure import FeedForwardNetwork

import glob
import matplotlib.image as mpimg
import numpy as np
import time

def extractRFromImages(files):

    image = []
    images = []
    
    for file in files:
        image = mpimg.imread(file)
        image = extractRed(image)
        images.append(image)
        image = []
    
    return images
    
def extractRed(image):

    row = []
    rImage = []
    
    for pixels in image:
        for pixel in pixels:
            row.append(pixel[0])
        rImage.append(row)
        row = []
    
    return np.array(rImage)

files = glob.glob('/home/jvujjini/Desktop/train/*.png')

files = sorted(files, key=lambda x: int(x.split("/")[-1][:-4]))

start = time.time()

print "Running..."
extractRFromImages(files)

end = time.time()

print "Task took " + str((end-start)) + " secs"