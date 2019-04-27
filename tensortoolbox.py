import numpy as np
def tenmat(D,dim):
	N=D.ndim
	matDim=[D.size//D.shape[dim],D.shape[dim]]
	dimTransfer=list(range(N))
	dimTransfer.remove(dim)
	dimTransfer.reverse()
	dimTransfer+=[dim]
	return D.transpose(dimTransfer).reshape(matDim).T
def ttm(D,matList,dims):
	N=D.ndim
	for i in range(len(matList)):
		dimTransfer=list(range(N))
		dimTransfer.remove(dims[i])
		dimTransfer+=[dims[i]]
		# convert the tensor into matrix
		TMP=D.transpose(dimTransfer).reshape([D.size//D.shape[dims[i]],D.shape[dims[i]]])
		# calculate the product of the tensor and matrix
		TMP=np.dot(TMP,matList[i].T)
		shapeAfterProd=list(D.shape)
		shapeAfterProd.remove(D.shape[dims[i]])
		shapeAfterProd+=[matList[i].shape[0]]
		dimTransfer=list(np.arange(N-1))
		dimTransfer.insert(dims[i],N-1)
		# convert the matrix back to tensor
		D=TMP.reshape(shapeAfterProd).transpose(dimTransfer)
	return D