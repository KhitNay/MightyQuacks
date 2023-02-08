import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from glob import glob
from os import path
import os
import re

class SteerDataSet(Dataset):
    
    def __init__(self,root_folder,img_ext = ".jpg" , transform=None):
        self.root_folder = root_folder
        self.transform = transform        
        self.img_ext = img_ext        
        self.filenames = os.listdir(self.root_folder)    
        self.totensor = transforms.ToTensor()
        
    def __len__(self):        
        return len(self.filenames)
    
    def __getitem__(self,idx):
        f = self.filenames[idx]        
        img = cv2.imread(self.root_folder + f)

        img_cut = img[100:,:]
       
        # print(img_cut.shape)

        img = cv2.resize(img_cut, [100,100])

        if self.transform == None:
            img = self.totensor(img)
        else:
            img = self.transform(img)   
        
        steering = re.findall("-\d.\d\d", f)

        if steering == []:
            steering = re.findall("\d\D\d\d", f)

        # print(steering)
        steering = np.float32(steering)  

        direction = None

        # Left
        if steering < 0:
            direction = 0

        # Straight
        elif steering == 0:
            direction = 1

        # Right
        else:
            direction = 2      
    
        sample = {"image":img , "steering":steering, "direction": direction}        
        
        return sample


def test():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds = SteerDataSet("/home/khit/MightyQuacks/penguinImages/data/",".jpg",transform)

    print("The dataset contains %d images " % len(ds))

    ds_dataloader = DataLoader(ds,batch_size=1,shuffle=True)
    for S in ds_dataloader:
        im = S["image"]    
        y  = S["direction"]
        
        
        print(im.shape)
        print(y)
        break



if __name__ == "__main__":
    test()
