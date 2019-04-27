import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.io as sio
import FaceDataIO as fdio
import random
import EigenfaceMethod

acc_train=np.zeros([10])
acc_test=np.zeros([10])

for k in range(10):
    print('The %dth experiment' %(k))
    # experiment parameters
    # select the data to form the base
    subs=list(range(40)) #select the subjects
    imageNum=list(random.sample(range(10),7))
    np.save('trainSet_paras.npy',[subs,imageNum])
    np.save('EigenfaceParas.npy',['att','-1'])

    # select the test set
    test_subs=subs
    test_imageNum=list(set(range(10))-set(imageNum))
    np.save('testSet_paras.npy',[test_subs,test_imageNum])
    acc_test[k]=EigenfaceMethod.run() # run the experiment by Eigenface
    test_imageNum=imageNum
    np.save('testSet_paras.npy',[test_subs,test_imageNum])
    acc_train[k]=EigenfaceMethod.run()

# print result
print('the average training recognition accuraccy is')
print(np.mean(acc_train))

print('the average testing recognition accuraccy is')
print(np.mean(acc_test))