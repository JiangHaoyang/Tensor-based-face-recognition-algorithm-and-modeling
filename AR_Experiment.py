import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.io as sio
import FaceDataIO as fdio
import random
import TensorfaceMethod
import TensorfaceAndRandOneMethod
import EigenfaceMethod

acc_tensorface=0
acc_rankOne=0
acc_eigenface_testing=0
acc_eigenface_training=0


# experiment parameters
# select the data to form the base
subs=list(range(100)) #select the subjects
nums=[6,13,14,15,17,18,19]
np.save('trainSet_paras.npy',[subs,nums])
np.save('EigenfaceParas.npy',['AR','-1'])
np.save('TensorfaceParas.npy',['AR'])

# select the test set
test_subs=subs
test_nums=[0,4,5]
np.save('testSet_paras.npy',[test_subs,test_nums])

acc_tensorface=TensorfaceMethod.run() # run the experiment by Tensorface method
acc_rankOne=TensorfaceAndRandOneMethod.run() # run the experiment by Rank-one method
acc_eigenface_testing=EigenfaceMethod.run() # run the experiment by Eigenface method

test_nums=nums
np.save('testSet_paras.npy',[test_subs,test_nums])
acc_eigenface_training=EigenfaceMethod.run() # run the experiment by Eigenface method




# print result
print('the recognition accuracies of Eigenface method on testing set are')
print(acc_eigenface_testing)
print('the recognition accuracies of Eigenface method on training set are')
print(acc_eigenface_training)
print('the recognition accuracies of Tensorface method are')
print(acc_tensorface)
print('the recognition accuracies of Rank-one approximate method are')
print(acc_rankOne)
