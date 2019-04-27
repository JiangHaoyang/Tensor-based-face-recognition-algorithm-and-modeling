import os
import random
import numpy as np
import cv2 as cv

subs=['yaleB16','yaleB17','yaleB18','yaleB19',
	'yaleB20','yaleB21','yaleB22','yaleB23',
	'yaleB24','yaleB25','yaleB26','yaleB27',
	'yaleB28','yaleB29','yaleB30','yaleB31',
	'yaleB32','yaleB33','yaleB35','yaleB36',
	'yaleB37','yaleB38','yaleB39']
poses=['P00A','P01A','P02A',
    'P03A','P04A','P05A',
    'P06A','P07A','P08A']
illums=['+000E+00','+000E+20','+000E+45','+000E+90','+000E-20',  #0-4
    '+000E-35','+005E+10','+005E-10','+010E+00','+010E-20',  #5-9
    '+015E+20','+020E+10','+020E-10','+020E-40','+025E+00',  #10-14
	'+035E+15','+035E+40','+035E+65','+035E-20','+050E+00',  #15-19
	'-035E+40','+060E+20','+060E-20','+070E+00','+070E+45',  #20-24
	'+070E-35','+085E+20','+085E-20','-110E+40','+110E+15',  #25-29
	'+110E+40','+110E+65','+110E-20','+120E+00','+130E+20',  #30-34
	'-005E+10','-005E-10','-110E+65','-010E-20','-015E+20',  #35-39
	'-020E+10','-020E-10','-020E-40','-025E+00','-035E+15',  #40-44
	'-130E+20','-035E+65','-035E-20','-050E+00','-120E+00',  #45-49
	'-060E+20','-060E-20','-070E+00','-070E+45','-070E-35',  #50-54
	'-085E+20','-085E-20','-095E+00']  #55-57
'''
illums=['+000E+00','+000E+20','+000E+45','+000E+90','+000E-20',  #0-4
    '+000E-35','+005E+10','+005E-10','+010E+00','+010E-20',  #5-9
    '+015E+20','+020E+10','+020E-10','+020E-40','+025E+00',  #10-14
	'+035E+15','+035E+40','+035E+65','+035E-20','+050E+00',  #15-19
	'+050E-40','+060E+20','+060E-20','+070E+00','+070E+45',  #20-24
	'+070E-35','+085E+20','+085E-20','+095E+00','+110E+15',  #25-29
	'+110E+40','+110E+65','+110E-20','+120E+00','+130E+20',  #30-34
	'-005E+10','-005E-10','-010E+00','-010E-20','-015E+20',  #35-39
	'-020E+10','-020E-10','-020E-40','-025E+00','-035E+15',  #40-44
	'-130E+20','-035E+65','-035E-20','-050E+00','-050E-40',  #45-49
	'-060E+20','-060E-20','-070E+00','-070E+45','-070E-35',  #50-54
	'-085E+20','-085E-20','-095E+00','-110E+15','-110E+40',  #55-59
	'-110E+65','-110E-20','-120E+00','-035E+40']  #60-63
'''
# some image from the yaleB11 is corrupted, delect the images form yaleB11
# some image from the yaleB12 is corrupted, delect the images form yaleB12
# some image from the yaleB13 is corrupted, delect the images form yaleB13
# some image from the yaleB15 is corrupted, delect the images form yaleB15
# some image from the yaleB34 is corrupted, delect the images form yaleB34
# use 20 28 37 49 58 61 illums to train the best has great effect on the acc(all black image)
image_size=[120,100]
yaleB={'subs':subs,'poses':poses,'illums':illums,'image_size':image_size}



subs=['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10',
	's11','s12','s13','s14','s15','s16','s17','s18','s19','s20',
	's21','s22','s23','s24','s25','s26','s27','s28','s29','s30',
	's31','s32','s33','s34','s35','s36','s37','s38','s39','s40']
imageNum=['1','2','3','4','5','6','7','8','9','10']
att={'subs':subs,'imageNum':imageNum,'image_size':image_size}

subs=list(range(1,101))
imageNum=list(range(1,29))
image_size=[165,120]
AR={'subs':subs,'imageNum':imageNum,'image_size':image_size}

subs=list(range(68))
imageNum=list(range(0,49))
image_size=[64,64]
PIE={'subs':subs,'imageNum':imageNum,'image_size':image_size}

def readAR(SUB,NUM):
	cPath=os.getcwd()
	dataPath=cPath+"\\AR_aligned"
	filepath=dataPath
	fileName="m-"+str(SUB)+"-"+str(NUM)+".bmp"
	fullPath=filepath+"\\"+fileName
	I=cv.imread(fullPath)
	I=I[:,:,0]
	return I
	

def readYaleB(SUB,ANGLE,LIGHT):
	#print(SUB)
	#print(ANGLE)
	#print(LIGHT)
	cPath=os.getcwd()
	dataPath=cPath+"\\CroppedExtendedYaleB"
	filepath=dataPath+"\\"+SUB
	fileName=SUB+"_"+ANGLE+LIGHT+".jpg"
	fullPath=filepath+"\\"+fileName
	I=cv.imread(fullPath)
	I=I[:,:,0]
	return I

def readAtt(SUB,NUM):
	cPath=os.getcwd()
	dataPath=cPath+"\\CroppedAtt_faces"
	filepath=dataPath+"\\"+SUB
	fileName=NUM+".jpg"
	fullPath=filepath+"\\"+fileName
	I=cv.imread(fullPath)
	I=I[:,:,0]
	return I

def readPIE(SUB,NUM):
	cPath=os.getcwd()
	dataPath=cPath+"\\CMU_PIE_Face"
	filepath=dataPath+"\\"+"Pose05_64x64_files"
	fileName=str(SUB*49+NUM+1)+".jpg"
	fullPath=filepath+"\\"+fileName
	I=cv.imread(fullPath)
	I=I[:,:,0]
	return I
	

def readFaceTensor(database,mode,dims):
	if(database=='yaleB'):
		if(mode=='random'):
		# dims[0] is numOfSubs, dims[1] is numOfPoses, dims[2] isnumOfillums
			subs_selected=random.sample(range(len(yaleB['subs'])),dims[0])
			poses_selected=random.sample(range(len(yaleB['poses'])),dims[1])
			illums_selected=random.sample(range(len(yaleB['illums'])),dims[2])
		if(mode=='select'):
			subs_selected=dims[0]
			poses_selected=dims[1]
			illums_selected=dims[2]
			tensor=YaleBFaceTensor[subs_selected,:,:,:,:]
		tensor=YaleBFaceTensor[subs_selected,:,:,:,:]
		tensor=tensor[:,poses_selected,:,:,:]
		tensor=tensor[:,:,illums_selected,:,:]
		return [tensor,subs_selected,poses_selected,illums_selected]
	if(database=='att'):
		if(mode=='random'):
			subs_selected=random.sample(range(len(att['subs'])),dims[0])
			nums_selected=random.sample(range(len(att['imageNum'])),dims[1])
		if(mode=='select'):
			subs_selected=dims[0]
			nums_selected=dims[1]
		tensor=attFullTensor[subs_selected,:,:,:]
		tensor=tensor[:,nums_selected,:,:]
		return [tensor,subs_selected,nums_selected]
	if(database=='AR'):
		if(mode=='random'):
			x=0 #do nothing, not finish yet
		if(mode=='select'):
			subs_selected=dims[0]
			nums_selected=dims[1]
			tensor=ARFullTensor[subs_selected,:,:,:]
		tensor=tensor[:,nums_selected,:,:]
		return [tensor,subs_selected,nums_selected]
	if(database=='PIE'):
		if(mode=='random'):
			x=0 #do nothing, not finish yet
		if(mode=='select'):
			subs_selected=dims[0]
			nums_selected=dims[1]
			tensor=PIEFullTensor[subs_selected,:,:,:]
		tensor=tensor[:,nums_selected,:,:]
		return [tensor,subs_selected,nums_selected]

def constructTensorFile(dataBase):
	print('constructing tensor file')
	if dataBase=='yaleB':
		YaleBFaceTensor=np.zeros([len(yaleB['subs']),len(yaleB['poses']),len(yaleB['illums'])]+yaleB['image_size'])
		for sub in range(len(yaleB['subs'])):
			for pose in range(len(yaleB['poses'])):
				for illum in range(len(yaleB['illums'])):
						YaleBFaceTensor[sub,pose,illum,:,:]=readYaleB(yaleB['subs'][sub],yaleB['poses'][pose],yaleB['illums'][illum])
		np.save('YaleBFullTensor.npy',YaleBFaceTensor)
		print('file saved')
	if dataBase=='att':
		attFullTensor=np.zeros([len(att['subs']),len(att['imageNum'])]+att['image_size'])
		for sub in range(len(att['subs'])):
			for num in range(len(att['imageNum'])):
				attFullTensor[sub,num,:,:]=readAtt(att['subs'][sub],att['imageNum'][num])
		np.save('attFullTensor.npy',attFullTensor)
		print('file saved')
	if dataBase=='AR':
		ARFullTensor=np.zeros([len(AR['subs']),len(AR['imageNum'])]+AR['image_size'])
		for sub in range(len(AR['subs'])):
			for num in range(len(AR['imageNum'])):
				ARFullTensor[sub,num,:,:]=readAR(AR['subs'][sub],AR['imageNum'][num])
		np.save('ARFullTensor.npy',ARFullTensor)
		print('file saved')
	if dataBase=='PIE':
		PIEFullTensor=np.zeros([len(PIE['subs']),len(PIE['imageNum'])]+PIE['image_size'])
		for sub in range(len(PIE['subs'])):
			for num in range(len(PIE['imageNum'])):
				PIEFullTensor[sub,num,:,:]=readPIE(PIE['subs'][sub],PIE['imageNum'][num])
		np.save('PIEFullTensor.npy',PIEFullTensor)
		print('file saved')

if (os.path.exists('YaleBFullTensor.npy')==False):
    constructTensorFile('yaleB')
if (os.path.exists('attFullTensor.npy')==False):
	constructTensorFile('att')
if (os.path.exists('ARFullTensor.npy')==False):
	constructTensorFile('AR')
if (os.path.exists('PIEFullTensor.npy')==False):
	constructTensorFile('PIE')
YaleBFaceTensor=np.load('YaleBFullTensor.npy')
attFullTensor=np.load('attFullTensor.npy')
ARFullTensor=np.load('ARFullTensor.npy')
PIEFullTensor=np.load('PIEFullTensor.npy')


