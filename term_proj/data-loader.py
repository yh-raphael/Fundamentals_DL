import os
import pathlib
import numpy as np
from torch.utils.data import Dataset


class SkeletonTestDataset(Dataset):

    def __init__(self, strFolderPath : str):
        
        strDataFolderlist = os.listdir(strFolderPath)
        
        self.X = []
        
        for strLabelFolderPath in strDataFolderlist:
            strOneLabelDataPath = os.path.join(strFolderPath, strLabelFolderPath)            
            self.X.append(strOneLabelDataPath)                

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = np.load(self.X[idx], allow_pickle=True).flatten()     
        filename = pathlib.PurePath(self.X[idx])  
        filename = pathlib.Path(filename).stem
        return filename, x

class SkeletonDataset(Dataset):

    def __init__(self, strFolderPath : str):                                                        # sample_image_folder/skeleton_npy
        self.dictOfLabes = {'good' : 0, 'left' : 1, 'right' : 2, 'turtleneck' : 3}
        strDataFolderlist = os.listdir(strFolderPath)                                               # [good, left, right, turtleneck]
        
        self.X = []
        self.Y = []

        for strLabelFolderPath in strDataFolderlist:                                                # iterate: good -> left -> right -> turtleneck
            strOneLabelDataPath = os.path.join(strFolderPath, strLabelFolderPath)                   # sample_image_folder/skeleton_npy/good
            listOfOneLabelDataPath = os.listdir(strOneLabelDataPath)                                # [good_0.npy, good_1.npy . . .]
            for strNPDataPath in listOfOneLabelDataPath:                                            # iterate: good_0.npy -> good_1.npy -> . . .
                self.X.append(os.path.join(strOneLabelDataPath,strNPDataPath))                      # sample_image_folder/skeleton_npy/good/good_0.npy
                self.Y.append(self.dictOfLabes[strLabelFolderPath])                                 # 0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = np.load(self.X[idx], allow_pickle=True).flatten()
        y = self.Y[idx]
        return x, y