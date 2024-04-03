#import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

device = torch.device('cpu') # cuda

import matplotlib.pyplot as plt

#add the path to DATA folder containing reactant and product profiles for the cellobiose systems



#function to voxelize the data that has been loaded to shape [simulation_time_steps x Npoints x Npoints x Npoints]
def voxelization(Npoints,box,data,atoms):
    delta_space=np.divide(box,Npoints) # the size of element along the length, breadth, height divided into Npoints
    print(Npoints)
    print(box)
    print(delta_space)
    voxel_data=[]
    traj_data=list(data) #make your data into a list so you can iterate over each list element that is a simulation time step
    for x in traj_data:
        count=np.zeros((Npoints, Npoints, Npoints)) #at each simulation time, count the number of atoms in the grid NpointsxNpointsxNpoints
        dat=x[1:] #take all atom positions, 1 onwards, becasue the first column records the simulation time step (how .xyz was converted to .npy as mentioned earlier)
        for i in range(0,dat.shape[0],3):#iterate every set of 3 coordinates that characterize an atomic coordinate at every simltn time
            a=dat[i:i+3] #take the x,y,z coordinates of every atom
            a[np.where(a<0)]=a[np.where(a<0)]+box[np.where(a<0)] #if any of it is negative add the box length, coz periodic boundary condition
            num=np.divide(a,delta_space) #figure out which of the grids the atom lies in
            num=np.ceil(num)-1

            num[np.where(num>Npoints-1)]=Npoints-1

            idx=tuple([int(e) for e in num]) #record the index i.e ith box in x, jth box in y, kth box in z is where the atom is
            count[idx]=count[idx]+1 #increment the atom count in that i,j,k th box

        #normalize atom counts by total num of atom to get a density at each sim time step    
        voxel_data.append(torch.tensor(np.divide(count,atoms).reshape((1,count.shape[0],count.shape[1],count.shape[2])),dtype=torch.float32) )

    return voxel_data


#now start with the 3D CNN code, it has the following components
#1. Dataloader 2. Defining the network structure 

#---------- define the dataloader here
class MD_Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        
        if self.transform:
            
            x = self.transform(x)
        
        return x
    
    def __len__(self):
        return len(self.data)
    

class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=1):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss[-1] - train_loss[-1]) > self.min_delta :
            self.counter +=1
            # if self.counter >1 and np.abs(validation_loss[-1]-validation_loss[-2])/validation_loss[-2]< 1e-3:
            #     self.counter+=1
            # if self.counter >1 and np.abs(train_loss[-1]-train_loss[-2])/train_loss[-2]< 1e-3:
            #     self.counter+=1
                
            if self.counter >= self.tolerance:  
                self.early_stop = True
        
                   
    
#---------- define the neural network architecture here

#USE BATCH NORMALIZATION, DROPOUT, XAVIER INITIALIZATION

#Building the 3D CNN Model

#USE BATCH NORMALIZATION,no DROPOUT, XAVIER INITIALIZATION

class Net(nn.Module):
    def __init__(self,time_slices):
        super(Net, self).__init__()
        self.time_slices=time_slices
        # activation map of size Bxtime_slicesx20X20x20
        self.conv1 = nn.Conv3d(in_channels = self.time_slices, out_channels =64, kernel_size = (3,3,3), stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1)) #------> CONV1
        self.bn1=nn.BatchNorm3d(num_features=64)
        # activation map of size Bx64X18X18X18    
        self.conv2=nn.Conv3d(in_channels = 64, out_channels =32, kernel_size = (3,3,3), stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1))#--------> CONV2
        self.bn2=nn.BatchNorm3d(num_features=32)
        # activation map of size Bx32x 16x 16x 16   
        
        self.pool1=nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0),dilation=(1,1,1),return_indices=True)#------> POOL 1
        # activation map of size Bx32x 8x 8x 8      

        self.conv3=nn.Conv3d(in_channels = 32, out_channels =16, kernel_size = (3,3,3), stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1))#-----> CONV3
        self.bn3=nn.BatchNorm3d(num_features=16)
        # activation map of size Bx16x 6x 6x 6    
        self.conv4=nn.Conv3d(in_channels = 16, out_channels =8, kernel_size = (3,3,3), stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1))#-----> CONV 4
        self.bn4=nn.BatchNorm3d(num_features=8)
        # activation map of size Bx8x 4x 4x 4    
        
        self.pool2=nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0),dilation=(1,1,1),return_indices=True)#----> POOL2
        # activation map of size Bx8X 2X 2X 2      
        # activation map of size Bx8X 2X 2X 2, then flatten it out to B x 64
       
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        
        
        self.fc3 = nn.Linear(in_features=16, out_features=32)
        
        self.fc4 = nn.Linear(in_features=32, out_features=64)
        
        
        #the decoder part starts from here
        self.unpool1=nn.MaxUnpool3d(kernel_size=2, stride=2, padding=0)
        self.convT1 = nn.ConvTranspose3d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.bn5=nn.BatchNorm3d(num_features=16)
        self.convT2 = nn.ConvTranspose3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.bn6=nn.BatchNorm3d(num_features=32)
        self.unpool2=nn.MaxUnpool3d(kernel_size=2, stride=2, padding=0)
        self.convT3 = nn.ConvTranspose3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.bn7=nn.BatchNorm3d(num_features=64)      
        self.convT4 = nn.ConvTranspose3d(in_channels=64, out_channels=self.time_slices, kernel_size=3, stride=1, padding=0)
        self.bn8=nn.BatchNorm3d(num_features=self.time_slices)      
        # activation map of size Bxtime_slicesx20X20x20


        
        
      

    def encoder(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x = F.relu(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=F.relu(x)
        
        x, indices1=self.pool1(x)
        
        x=self.conv3(x)
        x=self.bn3(x)
        x=F.relu(x)
        
        x=self.conv4(x)
        x=self.bn4(x)
        x=F.relu(x)
        
        x, indices2=self.pool2(x)
        
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
       
        x=self.fc2(x)
        x = F.relu(x)
        
        
        return x, indices1, indices2
    
    
    def decoder(self, x, indices1, indices2):
    
        
        x = self.fc3(x)
        x = F.relu(x)
        
        
        x = self.fc4(x)
        x = F.relu(x)
        
        
        x = x.view(x.size(0),8, 2,2,2)
        x=self.unpool1(x, indices2)
        
        x=self.convT1(x)
        x=self.bn5(x)
        x = F.relu(x)

        x=self.convT2(x)
        x=self.bn6(x)
        x = F.relu(x)

        x=self.unpool2(x, indices1)
    
        x=self.convT3(x)
        x=self.bn7(x)
        x=F.relu(x)
        
        x=self.convT4(x)
        x=self.bn8(x)
        x=F.relu(x)
        

        return x
    
    def forward(self, x):
        l, indices1, indices2 = self.encoder(x)
        output = self.decoder(l, indices1, indices2)
        return output
        
        

#initialize the weights of the neural network
def init_weights(m):
    if type(m) == nn.Conv3d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)



if __name__=='__main__':
    
    
    #load a trajectory file. Read a .xyz or .gro file of coordinates and convert it into a numpy array of shape [simulation_time_steps x number_of_atoms*3+1] the +1 is for the first column that records the simulation time step, *3 is for the x,y,z coordinates
    cb_100_reactant=np.load('CB_100_init.npy')
    cb_500_reactant=np.load('CB_500_init.npy')
    cb_900_reactant=np.load('CB_900_init.npy')
    cb_1200_reactant=np.load('CB_1200_init.npy')
    
    cb_100_pdt=np.load('CB_100_TG_pdt.npy')
    cb_500_pdt=np.load('CB_500_TG_pdt.npy')
    cb_900_pdt=np.load('CB_900_TG_pdt.npy')
    cb_1200_pdt=np.load('CB_1200_TG_pdt.npy')
    
    
    #load the simulaton box size as well. This is a 3x1 array of box length, breadth, height
    cb_100_box=np.load('Box_CB_100_init.npy')
    cb_500_box=np.load('Box_CB_500_init.npy')
    cb_900_box=np.load('Box_CB_900_init.npy')
    cb_1200_box=np.load('Box_CB_1200_init.npy')


    
    #number of boxes to divide each dimension into
    Npoints=20
    #take the total data and make a list of pytorch tensors, note that number of atoms is the dimension of a row minus the first column recording the sim time step, divided by 3 because each atom has x,y,z coordinate entries
    voxel_react_cb_100=voxelization(Npoints, cb_100_box, cb_100_reactant, int((cb_100_reactant.shape[1]-1)/3))
    voxel_react_cb_500=voxelization(Npoints, cb_500_box, cb_500_reactant, int((cb_500_reactant.shape[1]-1)/3))
    voxel_react_cb_900=voxelization(Npoints, cb_900_box, cb_900_reactant, int((cb_900_reactant.shape[1]-1)/3))
    voxel_react_cb_1200=voxelization(Npoints, cb_1200_box, cb_1200_reactant, int((cb_1200_reactant.shape[1]-1)/3))
    
    voxel_pdt_cb_100=voxelization(Npoints, cb_100_box, cb_100_pdt, int((cb_100_pdt.shape[1]-1)/3))
    voxel_pdt_cb_500=voxelization(Npoints, cb_500_box, cb_500_pdt, int((cb_500_pdt.shape[1]-1)/3))
    voxel_pdt_cb_900=voxelization(Npoints, cb_900_box, cb_900_pdt, int((cb_900_pdt.shape[1]-1)/3))
    voxel_pdt_cb_1200=voxelization(Npoints, cb_1200_box, cb_1200_pdt, int((cb_1200_pdt.shape[1]-1)/3))
    
    
    #so now voxel_react_cb_100 is of dimesnion num_sim_timesteps x Npoints x Npointsx Npoints
    #Suppose we would like to take 100 sim time slices and stack them together, so that each sample in new_data is now time_slices x NpointsxNpointsxNpoints. This is a donwsampling strategy and also ensures that when we convolve over the sample, spatio-temporal features are extracted across the voxel over 100 time steps of simulation
    
    time_slices=100
    
    # Stack all the rectant and product trajectory into time discretized voxels
    time_stacked_voxel=[]
    for i in range(0,len(voxel_react_cb_100),time_slices):
        if i+time_slices <len(voxel_react_cb_100):
            time_stacked_voxel.append(torch.cat((voxel_react_cb_100[i:i+time_slices]),dim=0))
        else:
            continue
    
    
    
    for i in range(0,len(voxel_react_cb_500),time_slices):
        if i+time_slices <len(voxel_react_cb_500):
            time_stacked_voxel.append(torch.cat((voxel_react_cb_500[i:i+time_slices]),dim=0))
        else:
            continue
        
    
    for i in range(0,len(voxel_react_cb_900),time_slices):
        if i+time_slices <len(voxel_react_cb_900):
            time_stacked_voxel.append(torch.cat((voxel_react_cb_900[i:i+time_slices]),dim=0))
        else:
            continue
    
    
    for i in range(0,len(voxel_react_cb_1200),time_slices):
        if i+time_slices <len(voxel_react_cb_1200):
            time_stacked_voxel.append(torch.cat((voxel_react_cb_1200[i:i+time_slices]),dim=0))
        else:
            continue
    
    
    for i in range(0,len(voxel_pdt_cb_100),time_slices):
        if i+time_slices <len(voxel_pdt_cb_100):
            time_stacked_voxel.append(torch.cat((voxel_pdt_cb_100[i:i+time_slices]),dim=0))
        else:
            continue
    
    
    for i in range(0,len(voxel_pdt_cb_500),time_slices):
        if i+time_slices <len(voxel_pdt_cb_500):
            time_stacked_voxel.append(torch.cat((voxel_pdt_cb_500[i:i+time_slices]),dim=0))
        else:
            continue
    
    
    for i in range(0,len(voxel_pdt_cb_900),time_slices):
        if i+time_slices <len(voxel_pdt_cb_900):
            time_stacked_voxel.append(torch.cat((voxel_pdt_cb_900[i:i+time_slices]),dim=0))
        else:
            continue
        
    
    for i in range(0,len(voxel_pdt_cb_1200),time_slices):
        if i+time_slices <len(voxel_pdt_cb_1200):
            time_stacked_voxel.append(torch.cat((voxel_pdt_cb_1200[i:i+time_slices]),dim=0))
        else:
            continue
        
    # Just check the dimesnions of what one time stacked voxel trajectory looks like    
    print('Each list element shape',time_stacked_voxel[0].shape)
    print('Length of total data is',len(time_stacked_voxel))
    
    ##---------------------- so far we have seen code to take reactant cellobiose trajectories and voxelizing them-----------
    ##----------------------- write a loop here or modify the above code to iteratively read all simulation trajectories and convert them to voxel data
    
    #then you could stack them all up into one big list, as data that goes into training the CNN
    #note that the data at 500, 900, 1200 are here just for illustartion and have not been read from file earlier, you have to read them depending on what your data is
    new_data=time_stacked_voxel

    
    #instantiate the model    
    model = Net(time_slices).to(device)
    early_stopping = EarlyStopping(tolerance=10, min_delta=1e-5)

    
    model.apply(init_weights)
    
    
    # Let's define an optimizer
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Let's define a Loss function
    
    lossfun = nn.MSELoss()
    
    
    #Finally perform epoch-wise training
    
    # Lets define a dataloader for train dataset
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if device==torch.device('cuda') else {}
    transform = transforms.Compose([transforms.ToTensor()])
    
    # train validation split
    train_size = int(0.8 * len(new_data))
    val_size = len(new_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(new_data, [train_size, val_size])
    
    #build the train and validation data loaders
    trainset=MD_Dataset(train_dataset, transform=None)
    train_loader=DataLoader(trainset, batch_size=50, shuffle=True, **kwargs)
    
    valset=MD_Dataset(val_dataset, transform=None)
    val_loader=DataLoader(valset, batch_size=50, shuffle=True, **kwargs)
    
    # Define train epochs
    
    epochs = 500
    avg_epoch_loss, avg_val_loss=[],[]
    
    training_start_time = time.time()
    
    for epoch in range(epochs):
        
        # iterate through train dataset
        epoch_loss = 0
    
        for batch_idx,data in enumerate(train_loader):
            
            data = data.to(device)
            
            # get output
            output = model(data)
            
            # compute loss function
            loss = lossfun(output, data)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # run optimizer 
            optimizer.step()
            
            # bookkeeping
            epoch_loss += loss.item() 
            
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                
        print('Train Epoch: {}\tAverage Loss: {:.6f}'.format(
            epoch+1, epoch_loss/len(train_loader.dataset)))
        
        
        avg_epoch_loss.append( epoch_loss/len(train_loader.dataset))
        print('-----------------Train Epoch: {}\tAverage Loss: {:.6f}'.format(
            epoch+1, epoch_loss/len(train_loader.dataset)))
            
        # start computing the validation loss for each epoch
        val_loss=0
        model.eval()
        for val_batch_idx, val_data in enumerate(val_loader):
            valdata= val_data.to(device)
            res=model(valdata)#*100
            
            loss=lossfun(res, valdata)
            val_loss += loss.item()
            
    
        avg_val_loss.append( val_loss/len(val_loader.dataset))
    
    
        #early stopping
        early_stopping(np.asarray(avg_epoch_loss), np.asarray(avg_val_loss))
        if early_stopping.early_stop:
          early_stop_epoch=epoch+1  
          print("We are at epoch:{0}".format(epoch+1))
          break
    
    print('Training finished, took {:.2f}s'.format(time.time() - training_start_time))
    #Training finished, took 8716.39s      
    # save network weights as follows, after training
    torch.save(model.state_dict(), "Some_model_name.pt")
    
    
    
    
    plt.figure(figsize=(7,7))
    plt.plot(avg_epoch_loss, label='Train')
    plt.plot(avg_val_loss, label='Validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
