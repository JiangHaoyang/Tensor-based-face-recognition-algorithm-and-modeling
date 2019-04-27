import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.io as sio
import FaceDataIO as fdio
import tensortoolbox as ttl
from tensorly.decomposition import parafac

################################### Improved Method ###################################
def run():
    database=np.load('TensorfaceParas.npy')[0]
    if database=='yaleB':
        ##################################################### Trianning set
        [subs,poses,illums]=np.load('trainSet_paras.npy')
        len_of_c_ref=len(subs)

        # read the corresponding image
        [FaceTensor,_,_,_]=fdio.readFaceTensor('yaleB','select',[subs,poses,illums])

        # calculate the base
        U=[]
        for i in range(5):
            mat=ttl.tenmat(FaceTensor,i)
            U.append(np.linalg.eig(np.dot(mat,mat.T))[1])
        #Z=ttl.ttm(FaceTensor,[U[0].T,U[1].T,U[2].T,U[3].T,U[4].T],list(range(5)))
        B=ttl.ttm(FaceTensor,[U[0].T,U[1].T,U[2].T],list(range(3)))

        if (os.path.exists('inv_B_T.npy')==False):
            # calculate the transfer matrix
            B_=np.zeros([np.prod(B.shape[0:3]),np.prod(B.shape[3:5])])
            for i in range(len(subs)):
                for j in range(len(poses)):
                    B_[(i*len(poses)+j)*len(illums):(i*len(poses)+j+1)*len(illums),:]=\
                        ttl.tenmat(B[i,j,:,:,:],0)
            # calculate the invert matrix of the base, it take some time
            print('calculating the invert matrix')
            inv_B_T=np.linalg.pinv(B_.T)
            print('finish calculation')
            # np.save('inv_B_T.npy',[inv_B_T])
            np.save('B_shape.npy',[B.shape])
        else:
            [inv_B_T]=np.load('inv_B_T.npy')
            print('loading the old invert matrix. Delete the file if want a new one')

        # calculate the Cp_ref
        Cp_ref=np.zeros([len(subs),len(subs)])
        I=ttl.tenmat(FaceTensor[:,1,1,:,:],0)
        Cp=np.dot(inv_B_T,I.T)
        for i in range(I.shape[0]):
            Cp_tmp=Cp[:,i]
            Cp_tmp=np.reshape(Cp_tmp,[len(subs),len(poses),len(illums)])

            Cp_ref[:,i]=parafac(Cp_tmp,rank=1)[0].reshape(Cp_ref[:,i].shape)
            Cp_ref[:,i]=Cp_ref[:,i]*np.sign(Cp_ref[0,i])
        #print(Cp_ref)



        ##################################################### Testing set

        # load the testing set parameters
        [test_subs,test_poses,test_illums]=np.load('testSet_paras.npy')
        # read the corresponding image
        [FaceTensor,_,_,_]=fdio.readFaceTensor('yaleB','select',[test_subs,test_poses,test_illums])

        # creat empty matrix to store Cp for the testing set
        Cp=np.zeros([len(test_subs),len(test_poses),len(test_illums),len_of_c_ref])

        # process each image one by one
        for i in range(len(test_subs)):
            print(i)
            for j in range(len(test_poses)):
                # load images set of the same subject and same pose
                images_set=ttl.tenmat(FaceTensor[i,j,:,:,:],0)
                for k in range(len(test_illums)):
                    # process one image each time
                    one_image=images_set[k,:]
                    # calculate the Cp for this image
                    Cp_this_image=np.dot(inv_B_T,one_image.T)
                    Cp_this_image=np.reshape(Cp_this_image,[len(subs),len(poses),len(illums)])
                    Cp[i,j,k,:]=parafac(Cp_this_image,rank=1)[0].reshape(Cp[i,j,k,:].shape)
                    
                    Cp[i,j,k,:]=Cp[i,j,k,:]*np.sign(Cp[i,j,k,0])
        # store the result
        np.save('improvedMethod_Cp.npy',Cp)

        predictMap=np.zeros([len(test_subs),len(test_poses),len(test_illums)])
        for i in range(len(test_subs)):
            for j in range(len(test_poses)):
                for k in range(len(test_illums)):
                    Cp_tmp=Cp[i,j,k,:].reshape([Cp.shape[3],1])*np.ones([1,Cp_ref.shape[1]])
                    tmp=np.sqrt(np.sum(np.square(Cp_tmp-Cp_ref),0))
                    #print(tmp)
                    predictMap[i,j,k]=np.argmin(tmp)

        Correct_map=np.arange(len(test_subs)).reshape([len(test_subs),1,1])\
            *np.ones([1,len(test_poses),1])*np.ones([1,1,len(test_illums)])

        acc=(Correct_map==predictMap).sum()/predictMap.size
        print(acc)

        '''
        Cp=Cp.reshape([np.prod(Cp.shape[0:3]),Cp.shape[3]])
        Correct_map=Correct_map.reshape([np.prod(Correct_map.shape[0:3]),1])
        np.save('SVMtest_Cp.npy',Cp)
        np.save('SVMtest_y.npy',Correct_map)
        '''
    if database=='AR' or database=='PIE':
        ##################################################### Trianning set
        [subs,nums]=np.load('trainSet_paras.npy')
        len_of_c_ref=len(subs)

        # read the corresponding image
        [FaceTensor,_,_]=fdio.readFaceTensor(database,'select',[subs,nums])

        # calculate the base
        U=[]
        for i in range(4):
            mat=ttl.tenmat(FaceTensor,i)
            U.append(np.linalg.eig(np.dot(mat,mat.T))[1])
        #Z=ttl.ttm(FaceTensor,[U[0].T,U[1].T,U[2].T,U[3].T,U[4].T],list(range(5)))
        B=ttl.ttm(FaceTensor,[U[0].T,U[1].T],list(range(2)))

        if (os.path.exists('inv_B_T.npy')==False):
            # calculate the transfer matrix
            B_=np.zeros([np.prod(B.shape[0:2]),np.prod(B.shape[2:4])])
            for i in range(len(subs)):
                B_[i*len(nums):(i+1)*len(nums),:]=ttl.tenmat(B[i,:,:,:],0)
            # calculate the invert matrix of the base, it take some time
            print('calculating the invert matrix')
            inv_B_T=np.linalg.pinv(B_.T)
            print('finish calculation')
            # np.save('inv_B_T.npy',[inv_B_T])
            np.save('B_shape.npy',[B.shape])
        else:
            [inv_B_T]=np.load('inv_B_T.npy')
            print('loading the old invert matrix. Delete the file if want a new one')

        # calculate the Cp_ref
        Cp_ref=np.zeros([len(subs),len(subs)])
        I=ttl.tenmat(FaceTensor[:,1,:,:],0)
        Cp=np.dot(inv_B_T,I.T)
        for i in range(I.shape[0]):
            Cp_tmp=Cp[:,i]
            Cp_tmp=np.reshape(Cp_tmp,[len(subs),len(nums)])

            Cp_ref[:,i]=parafac(Cp_tmp,rank=1)[0].reshape(Cp_ref[:,i].shape)
            Cp_ref[:,i]=Cp_ref[:,i]*np.sign(Cp_ref[0,i])
        #print(Cp_ref)


        ##################################################### Testing set

        # load the testing set parameters
        [test_subs,test_nums]=np.load('testSet_paras.npy')
        # read the corresponding image
        [FaceTensor,_,_]=fdio.readFaceTensor(database,'select',[test_subs,test_nums])

        # creat empty matrix to store Cp for the testing set
        Cp=np.zeros([len(test_subs),len(test_nums),len_of_c_ref])

        # process each image one by one
        for i in range(len(test_subs)):
            print(i)
            # load images set of the same subject
            images_set=ttl.tenmat(FaceTensor[i,:,:,:],0)
            for j in range(len(test_nums)):
                # process one image each time
                one_image=images_set[j,:]
                # calculate the Cp for this image
                Cp_this_image=np.dot(inv_B_T,one_image.T)
                Cp_this_image=np.reshape(Cp_this_image,[len(subs),len(nums)])
                Cp[i,j,:]=parafac(Cp_this_image,rank=1)[0].reshape(Cp[i,j,:].shape)
                
                Cp[i,j,:]=Cp[i,j,:]*np.sign(Cp[i,j,0])
        # store the result
        np.save('improvedMethod_Cp.npy',Cp)

        predictMap=np.zeros([len(test_subs),len(test_nums)])
        for i in range(len(test_subs)):
            for j in range(len(test_nums)):
                Cp_tmp=Cp[i,j,:].reshape([Cp.shape[2],1])*np.ones([1,Cp_ref.shape[1]])
                tmp=np.sqrt(np.sum(np.square(Cp_tmp-Cp_ref),0))
                predictMap[i,j]=np.argmin(tmp)

        Correct_map=np.arange(len(test_subs)).reshape([len(test_subs),1])\
            *np.ones([1,len(test_nums)])

        acc=(Correct_map==predictMap).sum()/predictMap.size
        print(acc)

    return acc

