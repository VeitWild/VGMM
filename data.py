import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)

#Simualte Low Rank gaussian data
N=500
J =2
J_p =1

A= np.random.normal( size=(J,J_p), ) 
sigma_low = 0.0001
Z = np.random.normal(size= (N,J_p,1))
error = sigma_low*np.random.normal(size=(N,J,1))

Y = (np.matmul(A.reshape(1,J,J_p),Z) + error).reshape(N,J) #NxJ

#plt.scatter(Y[:,0],Y[:,1])
#plt.xlim(-3,3)
#plt.ylim(-3,3)
#plt.show()

#Calculate EVD of covariance matrix
#eigenvalues, eigenvectors= np.linalg.eig( A.transpose() @ A )
#print(eigenvalues)

#evectors are J'x J'
#V=A @ eigenvectors # first J' eigenvectors of A @ A.t()


# Create Dataloader implementation
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, Y):
        'Initialization'
        self.Y = Y

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.Y[:,0])

  def __getitem__(self, index):
        'Generates one sample of data'
        y = self.Y[index,:]

        return y
  

# CUDA for PyTorch
#use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")
#torch.backends.cudnn.benchmark = True


#Create Train-Test split
Y_train, Y_test = train_test_split(Y , test_size=0.2, random_state=1)



