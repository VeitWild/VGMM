import torch
import matplotlib.pyplot as plt
from models import GaussianFullRank
from data import Y, Y_train

model = GaussianFullRank(dim_domain=2)
model.load_state_dict(torch.load('/home/vdwild/Code/VGMM/checkpoints/full_rank_epochs2000.pth'))

samples = model.sample(nr_samples=200).detach().numpy()

print(model.cov_matrix())
print()


plt.scatter(Y[:,0],Y[:,1])
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.scatter(samples[:,0], samples[:,1])
plt.show()


cov_mat_emp = 1/400 * Y_train.transpose() @ Y_train
print(cov_mat_emp)