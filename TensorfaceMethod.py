import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.io as sio
import FaceDataIO as fdio
import tensortoolbox as ttl


###################################### TensorFace Method ######################################
def run():
    database=np.load('TensorfaceParas.npy')[0]
    if database=='yaleB':
        ######################################################### Trainning set
        # load the trianning set parameters
        [subs,poses,illums]=np.load('trainSet_paras.npy')
        len_of_c_ref=len(subs)

        # read the corresponding image
        [FaceTensor,_,_,_]=fdio.readFaceTensor('yaleB','select',[subs,poses,illums])

        # calculate the base
        U=[]
        mat=ttl.tenmat(FaceTensor,0)
        U.append(np.linalg.eig(np.dot(mat,mat.T))[1])
        B=ttl.ttm(FaceTensor,[U[0].T],[0])

        # calculate the C_reference of the subjects
        # read one image for each subjects
        I=ttl.tenmat(FaceTensor[:,1,1,:,:],0)
        # select the corresponding base
        B_subset=ttl.tenmat(B[:,1,1,:,:],0)
        # calculate the C_reference
        Cp_ref=np.dot(np.linalg.pinv(B_subset.T),I.T)
        # calculate the transfer matrix
        inv_B_T=np.zeros([np.prod(B.shape[0:3]),np.prod(B.shape[3:5])])
        for i in range(len(poses)):
            for j in range(len(illums)):
                B_subset=ttl.tenmat(B[:,i,j,:,:],0)
                # print([(i*len(illums)+j)*len(subs),(i*len(illums)+j+1)*len(subs)])
                inv_B_T[(i*len(illums)+j)*len(subs):(i*len(illums)+j+1)*len(subs),:]=\
                    np.linalg.pinv(B_subset.T)


        ######################################################### Testing set
        # load the testing set parameters
        [subs,poses,illums]=np.load('testSet_paras.npy')
        # read the corresponding image
        [FaceTensor,_,_,_]=fdio.readFaceTensor('yaleB','select',[subs,poses,illums])

        # creat empty matrix to store Cp for the testing set
        Cp=np.zeros([len(subs),len(poses),len(illums),len_of_c_ref])
        predictMap=np.zeros([len(subs),len(poses),len(illums)])
        # process each image one by one
        for i in range(len(subs)):
            print(i)
            for j in range(len(poses)):
                # load images set of the same subject and same pose
                images_set=ttl.tenmat(FaceTensor[i,j,:,:,:],0)
                for k in range(len(illums)):
                    # process one image each time
                    one_image=images_set[k,:]
                    # calculate the Cp for this image
                    Cp_this_image=np.dot(inv_B_T,one_image.T)
                    Cp_this_image=Cp_this_image.reshape([Cp_this_image.size//len_of_c_ref,len_of_c_ref])
                    # find the best match
                    # set the first pair to be the best result
                    best_Cp=Cp_this_image[0,:].T
                    min_dist=np.sqrt((np.square(best_Cp-Cp_ref[:,0])).sum())
                    predict=0
                    # search the best result in N*M conditions
                    for n in range(Cp_this_image.shape[0]):
                        for m in range(len_of_c_ref):
                            dist=np.sqrt((np.square(Cp_this_image[n,:]-Cp_ref[:,m])).sum())
                            if (dist<min_dist):
                                # if find the best one, refresh the best data
                                best_Cp=Cp_this_image[n,:].T
                                min_dist=dist
                                predict=m
                    # store the best vector and the corresponding predict
                    Cp[i,j,k,:]=best_Cp
                    predictMap[i,j,k]=predict
                    #print(min_dist)

        # store the result
        np.save('tensorface_Cp.npy',Cp)
        np.save('tensorface_predictMap.npy',predictMap)

        Correct_map=np.arange(len(subs)).reshape([len(subs),1,1])\
            *np.ones([1,len(poses),1])*np.ones([1,1,len(illums)])

        acc=(Correct_map==predictMap).sum()/predictMap.size
        print(acc)
    if database=='AR' or database=='PIE':
        ######################################################### Trainning set
        # load the trianning set parameters
        [subs,nums]=np.load('trainSet_paras.npy')
        len_of_c_ref=len(subs)

        # read the corresponding image
        [FaceTensor,_,_]=fdio.readFaceTensor(database,'select',[subs,nums])

        # calculate the base
        U=[]
        mat=ttl.tenmat(FaceTensor,0)
        U.append(np.linalg.eig(np.dot(mat,mat.T))[1])
        B=ttl.ttm(FaceTensor,[U[0].T],[0])

        # calculate the C_reference of the subjects
        # read one image for each subjects
        I=ttl.tenmat(FaceTensor[:,1,:,:],0)
        # select the corresponding base
        B_subset=ttl.tenmat(B[:,1,:,:],0)
        # calculate the C_reference
        Cp_ref=np.dot(np.linalg.pinv(B_subset.T),I.T)
        # calculate the transfer matrix
        inv_B_T=np.zeros([np.prod(B.shape[0:2]),np.prod(B.shape[2:4])])
        for i in range(len(nums)):
            B_subset=ttl.tenmat(B[:,i,:,:],0)
            inv_B_T[i*len(subs):(i+1)*len(subs),:]=\
                np.linalg.pinv(B_subset.T)
        
        ######################################################### Testing set
        # load the testing set parameters
        [subs,nums]=np.load('testSet_paras.npy')
        # read the corresponding image
        [FaceTensor,_,_]=fdio.readFaceTensor(database,'select',[subs,nums])

        # creat empty matrix to store Cp for the testing set
        Cp=np.zeros([len(subs),len(nums),len_of_c_ref])
        predictMap=np.zeros([len(subs),len(nums)])
        # process each image one by one
        for i in range(len(subs)):
            print(i)
            images_set=ttl.tenmat(FaceTensor[i,:,:,:],0)
            for j in range(len(nums)):
                # process one image each time
                one_image=images_set[j,:]
                # calculate the Cp for this image
                Cp_this_image=np.dot(inv_B_T,one_image.T)
                Cp_this_image=Cp_this_image.reshape([Cp_this_image.size//len_of_c_ref,len_of_c_ref])
                # find the best match
                # set the first pair to be the best result
                best_Cp=Cp_this_image[0,:].T
                min_dist=np.sqrt((np.square(best_Cp-Cp_ref[:,0])).sum())
                predict=0
                # search the best result in N*M conditions
                for n in range(Cp_this_image.shape[0]):
                    for m in range(len_of_c_ref):
                        dist=np.sqrt((np.square(Cp_this_image[n,:]-Cp_ref[:,m])).sum())
                        if (dist<min_dist):
                            # if find the best one, refresh the best data
                            best_Cp=Cp_this_image[n,:].T
                            min_dist=dist
                            predict=m
                # store the best vector and the corresponding predict
                Cp[i,j,:]=best_Cp
                predictMap[i,j]=predict
                #print(min_dist)
        # store the result
        np.save('tensorface_Cp.npy',Cp)
        np.save('tensorface_predictMap.npy',predictMap)

        Correct_map=np.arange(len(subs)).reshape([len(subs),1])\
            *np.ones([1,len(nums)])

        acc=(Correct_map==predictMap).sum()/predictMap.size
        print(acc)
    return acc