'''

kaggle.py

Processes the CIFAR10 DataSet

Jagan Vujjini

'''

import numpy as np
from glob import glob
import matplotlib.image as mpimg

class CIFAR10:
    
    '''This class converts the dataset to our needs'''

    def __init__(self, phase, datapath=None):
        
        self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']
        self.n_classes = len(self.label_names)
        self.data_path = datapath

        if phase == 'train':
            files = glob(self.data_path + 'train/*.png')
        else:
            files = glob(self.data_path + 'test/*.png')

        # Sorting the files
        files = sorted(files, key=lambda x: int(x.split("/")[-1][:-4]))
        
        #Reading images by RGB values
        images = np.array([mpimg.imread(f) for f in files])
        
        '''intermediate step just to test the network. 
        Cannot convert a 32*32*3 image to a 32 attribute 
        feature vector. Taking means of Columns and 
        converting the image into a single line obviously 
        wont give better accuracy!'''
        
        images = images.mean(axis=1).mean(axis=2)
        
        self.images = images
        
        self.label = {k: v for k, v in zip(self.label_names, range(self.n_classes))}

        if phase == 'train':
            self.y = self.generateLabels(self.label)
        else:
            print "Testing Labels, all zeros!"
            self.y = np.zeros(images.shape)
            
    def generateLabels(self, class_label):
            
        '''Generating class labels for the images'''
        
        print "Loading training set"
        y = np.genfromtxt(self.data_path + 'trainLabels.csv',delimiter=',',skip_header=1,converters={1: self.convert})
        intlabels = np.zeros((y.shape[0], self.n_classes), dtype='float32')
        for i in xrange(y.shape[0]):
            intlabels[i, y[i][1]] = 1.
        y = intlabels
        return np.array(y)
    
    def convert(self, pos):
        return self.label[pos]
