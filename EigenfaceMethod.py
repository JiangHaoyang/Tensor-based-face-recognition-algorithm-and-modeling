import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.io as sio
import FaceDataIO as fdio
import tensortoolbox as ttl

################################# Eigenface Method #################################
def run():
    ################################################Trianning set
    dataSet=np.load('EigenfaceParas.npy')[0]
    numberOfEigenfaces=int(np.load('EigenfaceParas.npy')[1])

    if dataSet=='yaleB':
        [subs,poses,illums]=np.load('trainSet_paras.npy')
        # read the corresponding image
        [FaceTensor,_,_,_]=fdio.readFaceTensor('yaleB','select',[subs,poses,illums])
        pixelsNum=np.prod(FaceTensor.shape[3:5])
        faceNumOfEachSub=np.prod(FaceTensor.shape[1:3])
        # reshape into matrix
        Face=np.zeros([pixelsNum,np.prod(FaceTensor.shape[0:3])])
        tmp=0
        for i in range(len(subs)):
            for j in range(len(poses)):
                for k in range(len(illums)):
                    Face[:,tmp]=FaceTensor[i,j,k,:,:].reshape([pixelsNum])
                    tmp+=1
        referenceFace=np.zeros([pixelsNum,len(subs)])
        for i in range(len(subs)):
            referenceFace[:,i]=np.mean(Face[:,i*np.prod(FaceTensor.shape[1:3]):(i+1)*np.prod(FaceTensor.shape[1:3])],1)
    if dataSet=='att':
        [subs,imageNum]=np.load('trainSet_paras.npy')
        # read the corresponding image
        [FaceTensor,_,_]=fdio.readFaceTensor('att','select',[subs,imageNum])
        pixelsNum=np.prod(FaceTensor.shape[2:4])
        faceNumOfEachSub=FaceTensor.shape[1]
        # reshape into matrix
        Face=np.zeros([pixelsNum,np.prod(FaceTensor.shape[0:2])])
        tmp=0
        for i in range(len(subs)):
            for j in range(len(imageNum)):
                Face[:,tmp]=FaceTensor[i,j,:,:].reshape([pixelsNum])
                tmp+=1
        referenceFace=np.zeros([pixelsNum,len(subs)])
        for i in range(len(subs)):
            referenceFace[:,i]=np.mean(Face[:,i*faceNumOfEachSub:(i+1)*faceNumOfEachSub],1)
    if dataSet=='AR' or dataSet=='PIE':
        [subs,imageNum]=np.load('trainSet_paras.npy')
        # read the corresponding image
        [FaceTensor,_,_]=fdio.readFaceTensor(dataSet,'select',[subs,imageNum])
        pixelsNum=np.prod(FaceTensor.shape[2:4])
        faceNumOfEachSub=FaceTensor.shape[1]
        # reshape into matrix
        Face=np.zeros([pixelsNum,np.prod(FaceTensor.shape[0:2])])
        tmp=0
        for i in range(len(subs)):
            for j in range(len(imageNum)):
                Face[:,tmp]=FaceTensor[i,j,:,:].reshape([pixelsNum])
                tmp+=1
        referenceFace=np.zeros([pixelsNum,len(subs)])
        for i in range(len(subs)):
            referenceFace[:,i]=np.mean(Face[:,i*faceNumOfEachSub:(i+1)*faceNumOfEachSub],1)

    # calculate the meanface
    meanface=np.mean(Face,1).reshape([pixelsNum,1])
    # subtract the meanface from the Face
    trainSet=Face-meanface*np.ones([1,faceNumOfEachSub*len(subs)])

    # PCA
    eigVal,eigVector_prime=np.linalg.eig(np.dot(trainSet.T,trainSet))
    eigVector=np.dot(trainSet,eigVector_prime)
    ori_eigVector=eigVector
    # select the k largest eigenface
    if numberOfEigenfaces!=-1:
        descending_index=np.argsort(-np.abs(eigVal))
        eigVector=ori_eigVector[:,descending_index[0:numberOfEigenfaces]]
    # calculate the coefficient vector for each subject
    referenceFace=referenceFace-meanface*np.ones([1,len(subs)])
    Cp_ref=np.dot(referenceFace.T,eigVector)
    ################################################Testing set
    if dataSet=='yaleB':
        [test_subs,test_poses,test_illums]=np.load('testSet_paras.npy')
        # read the corresponding image
        [testFaceTensor,_,_,_]=fdio.readFaceTensor('yaleB','select',[test_subs,test_poses,test_illums])
        testfaceNumOfEachSub=np.prod(testFaceTensor.shape[1:3])
        # reshape into matrix
        testFace=np.zeros([pixelsNum,testfaceNumOfEachSub*len(test_subs)])
        tmp=0
        for i in range(len(test_subs)):
            for j in range(len(test_poses)):
                for k in range(len(test_illums)):
                    testFace[:,tmp]=testFaceTensor[i,j,k,:,:].reshape([pixelsNum])
                    tmp+=1
        correctMap=np.arange(len(test_subs)).reshape([len(test_subs),1])
        correctMap=correctMap*np.ones([1,testfaceNumOfEachSub])
        correctMap=correctMap.reshape(correctMap.size)
    if dataSet=='att':
        [test_subs,test_imageNum]=np.load('testSet_paras.npy')
        # read the corresponding image
        [testFaceTensor,_,_]=fdio.readFaceTensor('att','select',[test_subs,test_imageNum])
        testfaceNumOfEachSub=testFaceTensor.shape[1]
        # reshape into matrix
        testFace=np.zeros([pixelsNum,testfaceNumOfEachSub*len(test_subs)])
        tmp=0
        for i in range(len(test_subs)):
            for j in range(len(test_imageNum)):
                testFace[:,tmp]=testFaceTensor[i,j,:,:].reshape([pixelsNum])
                tmp+=1
        correctMap=np.arange(len(test_subs)).reshape([len(subs),1])
        correctMap=correctMap*np.ones([1,testfaceNumOfEachSub])
        correctMap=correctMap.reshape(correctMap.size)
    if dataSet=='AR' or dataSet=='PIE':
        [test_subs,test_imageNum]=np.load('testSet_paras.npy')
        # read the corresponding image
        [testFaceTensor,_,_]=fdio.readFaceTensor(dataSet,'select',[test_subs,test_imageNum])
        testfaceNumOfEachSub=testFaceTensor.shape[1]
        # reshape into matrix
        testFace=np.zeros([pixelsNum,testfaceNumOfEachSub*len(test_subs)])
        tmp=0
        for i in range(len(test_subs)):
            for j in range(len(test_imageNum)):
                testFace[:,tmp]=testFaceTensor[i,j,:,:].reshape([pixelsNum])
                tmp+=1
        correctMap=np.arange(len(test_subs)).reshape([len(subs),1])
        correctMap=correctMap*np.ones([1,testfaceNumOfEachSub])
        correctMap=correctMap.reshape(correctMap.size)

    # subtract the meanface from the Face
    testSet=testFace-meanface*np.ones([1,testfaceNumOfEachSub*len(test_subs)])
    # calculate the coefficient vector for each image
    Cp=np.dot(testSet.T,eigVector)

    # find the best match
    Cp=Cp.reshape(Cp.shape+(1,))*np.ones([1,1,Cp_ref.shape[0]])
    Cp_ref=Cp_ref.T.reshape((1,)+Cp_ref.T.shape)*np.ones([Cp.shape[0],1,1])

    err=np.square(Cp-Cp_ref).sum(axis=1)
    predict=np.argmin(err,1)

    # Trainging set accuracy
    acc=(predict==correctMap).sum()/correctMap.size
    return acc