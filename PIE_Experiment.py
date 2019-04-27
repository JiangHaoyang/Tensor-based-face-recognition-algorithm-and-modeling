import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.io as sio
import FaceDataIO as fdio
import random
import TensorfaceMethod
import TensorfaceAndRandOneMethod
import EigenfaceMethod

repeatTimes=10
acc_tensorface=np.zeros([repeatTimes])
acc_rankOne=np.zeros([repeatTimes])
acc_eigenface_testing=np.zeros([repeatTimes])
acc_eigenface_training=np.zeros([repeatTimes])

for k in range(repeatTimes):
    # experiment parameters
    # select the data to form the base
    subs=list(range(68)) #select the subjects
    nums=list(random.sample(range(49),6))
    np.save('trainSet_paras.npy',[subs,nums])
    np.save('EigenfaceParas.npy',['PIE','-1'])
    np.save('TensorfaceParas.npy',['PIE'])

    # select the test set
    test_subs=subs
    test_nums=list(set(range(49))-set(nums))
    np.save('testSet_paras.npy',[test_subs,test_nums])

    acc_tensorface[k]=TensorfaceMethod.run() # run the experiment by Tensorface method
    acc_rankOne[k]=TensorfaceAndRandOneMethod.run() # run the experiment by Rank-one method
    acc_eigenface_testing[k]=EigenfaceMethod.run() # run the experiment by Eigenface method

    test_nums=nums
    np.save('testSet_paras.npy',[test_subs,test_nums])
    acc_eigenface_training[k]=EigenfaceMethod.run() # run the experiment by Eigenface method




# print result
print('the recognition accuracies of Eigenface method on testing set are')
print(acc_eigenface_testing.mean())
print('the recognition accuracies of Eigenface method on training set are')
print(acc_eigenface_training.mean())
print('the recognition accuracies of Tensorface method are')
print(acc_tensorface.mean())
print('the recognition accuracies of Rank-one approximate method are')
print(acc_rankOne.mean())
