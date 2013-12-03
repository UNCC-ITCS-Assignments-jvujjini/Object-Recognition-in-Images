'''

feedForwardNetwork.py

simple FeedForwardNetwork for CIFAR 10 Object Recognition

Jagan Vujjini

'''

from kaggle import CIFAR10
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LinearLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError

print "Starting Process..."

print "\nLoading Training DataSet..."

train = CIFAR10('train', datapath='/home/jvujjini/Desktop/CIFAR/')

print "\nTraining DataSet Loaded...initializing Classification DataSet"

dataSet = ClassificationDataSet(train.images.shape[1], nb_classes=train.n_classes, class_labels=train.label_names)

print "\n Adding Training values to Dataset..."

for image, image_class in zip(train.images,train.y):
    dataSet.appendLinked(image, image_class.nonzero()[0])

dataSet._convertToOneOfMany( )

'''
print "\n Loading Testing DataSet..."

test = CIFAR10('test',datapath='/home/jvujjini/Desktop/CIFAR/')

testSet = ClassificationDataSet(test.X.shape[1], nb_classes=test.n_classes, class_labels=test.label_names)

print "\n Adding training values to Dataset..."

for timage, timage_class in zip(test.X,test.y):
    testSet.appendLinked(timage, timage_class.nonzero()[0])

testSet._convertToOneOfMany( )

print "\nBoth the DataSets Built!"

'''

print "\nTraining in progress..."

'''Building a fully connected Feed Forward Network'''
feedForwardNetwork = buildNetwork(dataSet.indim, 5, dataSet.outdim, outclass=LinearLayer)

'''using the Backprop Trainer'''
classifier = BackpropTrainer(feedForwardNetwork, dataset=dataSet, momentum=0.1, verbose=True, weightdecay=0.01)

for i in range(5):
    classifier.trainEpochs(1)
    result = percentError(classifier.testOnClassData(),dataSet['class'])
    print "epoch# %4d" % classifier.totalepochs, "  train error: %5.2f%%" % result