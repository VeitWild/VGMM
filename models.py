import torch
import numpy as np
import matplotlib.pyplot as plt

from data import Y_test
from abc import ABC, abstractmethod

#Abstract base class that forces the FullRank and LowRank Gaussians to implement the methods below
class GaussianMeasure(ABC):

    @abstractmethod
    def get_evalues_cov_matrix(self, sale):
        pass

    @abstractmethod
    def nll(self,Y):
        pass

    @abstractmethod
    def sample(self, nr_samples):
        pass

    @abstractmethod
    def get_cov_matrix(self):
        pass


class GaussianFullRank(torch.nn.Module,GaussianMeasure):
    def __init__(self,dim_domain):
        super().__init__()

        #Initialise paramter of loss
        J = dim_domain
        nr_lower = int((J * (J - 1)) / 2)
        self.log_diag = torch.nn.Parameter(torch.randn(J)) # trainable parameters
        self.L_lower = torch.nn.Parameter(torch.randn(nr_lower)) # trainable parameters


    def assemble_cholesky(self):
        J = self.log_diag.size(0) # data-dimension

        #Build matrix with exp(log_diag) on diagonal, L_lower below and 0s above diagonal
        L = torch.ones(J, J)
        L = L - torch.diag(L) + torch.diag(torch.exp(self.log_diag) ) # I think this only works bc of some shady broadcasting
        indices = torch.tril_indices(J, J, -1)
        #print(L[indices[0,],indices[1,]]) # this gives a vector not a matrix for some wild reason
        L[indices[0,],indices[1,]] = self.L_lower

        return(L)
    
    def get_cov_matrix(self):
        L = self.assemble_cholesky()
        J = L.shape[0]
        evals, evectors = torch.linalg.eigh(L @ L.t()) 

        evals = (1/evals).reshape(1,J)

        cov_mat = evectors * evals # JxJ times 1xJ -> broadcasting gives JxJ 
        cov_mat = torch.matmul(cov_mat,evectors)

        return(cov_mat)

    def forward(self, Y_batch):

        #Obtain important indices
        size_batch = Y_batch.shape[0] #batch-size
        J = self.log_diag.size(0) # data-dimension

        L = self.assemble_cholesky()

        #L is 1xJxJ, Y_batch is size_batchxJx1 gives with broadcasting of L: size_batch x Jx1 -> can be reshaped to size_batchxJ

        #Calculate Loss term
        log_likelihood =   (- J/2 * np.log(2*3.14)) \
                        +   torch.sum(self.log_diag) \
                        -0.5 *1/size_batch* torch.matmul(L.t().reshape(1,J,J), Y_batch.reshape(size_batch,J,1)).pow(2).sum()

        #loss is the negative of average of log_likelihood for Y_1, ... Y_{N_batch}
        return(-log_likelihood)
    
    def nll(self, Y_batch):
        return self.forward(Y_batch)
    
    def get_evalues_cov_matrix(self):
        L = self.assemble_cholesky()
        cov_mat_inv = L @ L.t() #inverse of cov matrix
        evals = torch.linalg.eigvalsh(cov_mat_inv ) # evals of cov_mat^{-1} = 1/ (evals of cov_mat)
        return (1/evals)
    
    def sample(self,nr_samples):
        L = self.assemble_cholesky()
        J = L.shape[0]
        evals, evectors = torch.linalg.eigh(L @ L.t()) 

        evals = torch.square(1/evals).reshape(1,J)

        root_mat = evectors * evals # JxJ times 1xJ -> broadcasting gives JxJ 
        Z = torch.randn(nr_samples, J,1)

        samples = torch.matmul(root_mat.reshape(1,J,J),Z ).reshape(nr_samples,J)

        return(samples)

class GaussianLowRank(torch.nn.Module,GaussianMeasure):
    '''
    Implements a Gaussian measure with covariance matrix with lower rank structure
    We paramterise Cov_mat  = A A^T + jitter * Identity , where A is dim_domain x r with r << dim_domain
    Further A = [[A_upper], [A_lower,]] where A_upper is a lower triangular rxr matrix with strictly positive diagonal
    And A lower is arbitrary (dim_domain-r) x r 
    '''
    def __init__(self,dim_domain,rank,jitter=0.0001^2):
        super().__init__()

        #Initialise parameters
        self.dim_domain = dim_domain
        self.rank = rank
        self.jitter = jitter
        nr_lower = int((rank * (rank - 1)) / 2)
        self.A_upper_log_diag = torch.nn.Parameter(torch.randn(rank)) # log_diagonal entries of A_upper
        self.A_upper_lower = torch.nn.Parameter(torch.randn(nr_lower)) # below diagonal entries of A_upper
        self.A_lower = torch.nn.Parameter(torch.randn(dim_domain-rank,rank)) # A_lower matrix
    
    def assemble_cholesky(self):
        '''
        Assembles matrix A from cov_mat = A A^T + jitter I_{dim_domain}
        '''
        #Build A_upper ensuring positive entries on diagonal
        A_upper = torch.ones(self.rank, self.rank)
        A_upper = A_upper - torch.diag(A_upper) + torch.diag(torch.exp(self.A_upper_log_diag) ) # I think this only works bc of some shady broadcasting
        indices = torch.tril_indices(self.rank, self.rank, -1)
        #print(L[indices[0,],indices[1,]]) # this gives a vector not a matrix for some wild reason
        A_upper[indices[0,],indices[1,]] = self.A_upper_lower

        #Combin A_upper and A_lower to dim_data x rank - matrix
        A = torch.stack(A_upper,self.A_lower,dim=0)

        return(A)
    
    def forward(self, Y_batch):
        '''
        Computes average nll for Y_batch with Y \sim N(0, cov_mat) where cov_mat = A A^T + jitter I_{dim-domain}
        '''
        cov_matrix = self


    

    

    



    
gaussian = GaussianFullRank(dim_domain=2)
samples = gaussian.sample(nr_samples=100).detach().numpy()

# gaussian.eigenvalues()
print(gaussian.get_cov_matrix())
print(gaussian.get_evalues_cov_matrix())

#plt.scatter(samples[:,0],samples[:,1])
#plt.xlim(-3,3)
#plt.ylim(-3,3)
#plt.show()
