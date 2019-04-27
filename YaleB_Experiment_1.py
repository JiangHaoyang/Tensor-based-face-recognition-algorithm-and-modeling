import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.io as sio
import FaceDataIO as fdio
import random
import TensorfaceMethod
import TensorfaceAndRandOneMethod
import EigenfaceMethod

repeatTimes=2
acc_tensorface=np.zeros([repeatTimes])
acc_rankOne=np.zeros([repeatTimes])
acc_eigenface_testing=np.zeros([repeatTimes])
acc_eigenface_training=np.zeros([repeatTimes])

for k in range(repeatTimes):
    print('The %dth experiment' %(k))
    # experiment parameters
    # select the data to form the base
    subs=list(range(23)) #select the subjects
    poses=list(random.sample(range(9),5))
    illums=list(random.sample(range(58),5))
    np.save('trainSet_paras.npy',[subs,poses,illums])
    np.save('EigenfaceParas.npy',['yaleB','-1'])
    np.save('TensorfaceParas.npy',['yaleB'])

    # select the test set
    test_subs=subs
    test_poses=poses #same poses
    test_illums=list(set(range(58))-set(illums)) #different illums
    np.save('testSet_paras.npy',[test_subs,test_poses,test_illums])

    acc_tensorface[k]=TensorfaceMethod.run() # run the experiment by Tensorface method
    acc_rankOne[k]=TensorfaceAndRandOneMethod.run() # run the experiment by Rank-one method
    acc_eigenface_testing[k]=EigenfaceMethod.run() # run the experiment by Eigenface method

    test_illums=illums
    np.save('testSet_paras.npy',[test_subs,test_poses,test_illums])
    acc_eigenface_training[k]=EigenfaceMethod.run() # run the experiment by Eigenface method




# print result
print('the recognition accuracies of Eigenface method are')
print(acc_eigenface_testing)
print('the average recognition accuraccy is')
print(np.mean(acc_eigenface_testing))

print('the recognition accuracies of Tensorface method are')
print(acc_tensorface)
print('the average recognition accuraccy is')
print(np.mean(acc_tensorface))

print('the recognition accuracies of Rank-one approximate method are')
print(acc_rankOne)
print('the average recognition accuraccy is')
print(np.mean(acc_rankOne))