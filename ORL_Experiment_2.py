import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.io as sio
import FaceDataIO as fdio
import random
import EigenfaceMethod

repeatTimes=30
numOfEigenfaceRange=list(range(5,281,5))
acc_train=np.zeros([repeatTimes,len(numOfEigenfaceRange)])
acc_test=np.zeros([repeatTimes,len(numOfEigenfaceRange)])
tmp=0
for numOfEigenface in numOfEigenfaceRange:
    print('%d Eigenfaces' %(numOfEigenface))
    for k in range(repeatTimes):
        print('The %dth experiment' %(k))
        # experiment parameters
        # select the data to form the base
        subs=list(range(40)) #select the subjects
        imageNum=list(random.sample(range(10),7))
        np.save('trainSet_paras.npy',[subs,imageNum])
        np.save('EigenfaceParas.npy',['att',str(numOfEigenface)])

        # select the test set
        test_subs=subs
        test_imageNum=imageNum
        np.save('testSet_paras.npy',[test_subs,test_imageNum])
        acc_train[k,tmp]=EigenfaceMethod.run() # run the experiment by Eigenface

        # select the test set
        test_subs=subs
        test_imageNum=list(set(range(10))-set(imageNum))
        np.save('testSet_paras.npy',[test_subs,test_imageNum])
        acc_test[k,tmp]=EigenfaceMethod.run() # run the experiment by Eigenface
    tmp+=1


# print result
print('the training recognition accuraccy is')
print(acc_train)

print('the testing recognition accuraccy is')
print(acc_test)