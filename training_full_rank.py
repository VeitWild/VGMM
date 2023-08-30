import torch
import numpy as np
from data import Dataset, Y_train, Y_test
from models import GaussianFullRank

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

#Hyperparameters such as batch-size and learning rate
params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 2}

learning_rate = 1e-3

#Dataloaders for train and test set
training_set = Dataset(Y_train)
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(Y_test)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

gaussian_fr = GaussianFullRank(dim_domain=Y_train.shape[1]) #Thats the model



###Initialise optimiser
optimizer = torch.optim.Adam(gaussian_fr.parameters(), lr=learning_rate)

###Define train Loop
def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch_index,Y_batch in enumerate(dataloader):
        # Compute prediction and loss
        Y_batch = Y_batch.to(device).float()
        batch_loss= model(Y_batch)
        

        # Backpropagation
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_index % 100 == 0:
            batch_loss, current = batch_loss.item(), (batch_index + 1) * len(Y_batch)
            print(f"loss: {batch_loss:>7f}  [{current:>5d}/{size:>5d}]")
 

def test_loop(dataloader, model):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    nr_observations = len(dataloader.dataset)
    sum = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    # We calculate the average loss on hold-out data: 1/N_test \sum_{i=1}^{N_test} -\log p(y_i)
    with torch.no_grad():
        for Y_batch in dataloader:
            Y_batch = Y_batch.to(device).float()
            batch_size = Y_batch.shape[0]
            loss_batch= batch_size*model(Y_batch) #estimate sum of loss of all  atcb-points
            sum +=loss_batch
 

    sum /=  nr_observations # now sum becomes average over all test points
    print(f"Avg. test loss: {sum:>8f} \n")


epochs = 2000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(training_generator, gaussian_fr, optimizer)
    test_loop(validation_generator, gaussian_fr)
print("Done!")

path = '/home/vdwild/Code/VGMM/checkpoints/full_rank_epochs' + str(epochs) + '.pth' 
torch.save(gaussian_fr.state_dict(), path)







#loss.eigenvalues()

#for p in loss.parameters():
#    print(p)




#Some tests
#M = 5
#N = 10
#Y = torch.ones(8,M)
#M_tilde = int((M * (M - 1)) / 2)
#log_diag = torch.ones(M)
#L_lower = torch.arange(1,M_tilde+1).float()
#loss(Y_batch=Y, log_diag=log_diag, L_lower=L_lower,nr_observations=N)