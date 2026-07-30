from torch.utils.data import Dataset,DataLoader
import numpy as np
import random
import cv2
import h5py

class TrainDataset(Dataset):
    def __init__(self, dir_rgb, dir_hsi, train_list_dir, bgr2rgb=True, train = True):

        #List to store hyper and rgb img paths 
        #Not storing images as it consumes too much space
        self.hyper_list = []
        self.bgr_list = []
        h,w = 482,512  # img shape
        if train:
            hyper_data_path = f'{dir_hsi}/Train_spectral/'
            bgr_data_path = f'{dir_rgb}/Train_RGB/'
        else:
            hyper_data_path = f'{dir_hsi}/Valid_spectral/'
            bgr_data_path = f'{dir_rgb}/Valid_RGB/'

        #Opens list file and saves the list as array
        with open(train_list_dir, 'r') as fin:
            self.hyper_list = [line.replace('\n','.mat') for line in fin]
            self.bgr_list = [line.replace('mat','jpg') for line in self.hyper_list]
        self.hyper_list.sort()
        self.bgr_list.sort()
        print(f'len(hyper) of ntire2022 dataset:{len(self.hyper_list)}')
        print(f'len(bgr) of ntire2022 dataset:{len(self.bgr_list)}')
        for i in range(len(self.hyper_list)):

            #check to see if the file names are same
            assert self.hyper_list[i].split('.')[0] ==self.bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'

            #Check to see if mat is present in the file name
            if 'mat' not in self.hyper_list[i]:
                continue

            #Creating array containing path of the files
            self.hyper_list[i] = hyper_data_path + self.hyper_list[i]
            self.bgr_list[i] = bgr_data_path + self.bgr_list[i]
            
            
        self.img_num = len(self.hyper_list)
        self.length = self.img_num

    
    def __getitem__(self, idx):

        #Reading hsi file
        with h5py.File(self.hyper_list, 'r') as mat:
            hyper =np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])

        #Reading rgb file
        bgr = cv2.imread(self.bgr_list)
        #Converting to rgb from bgr 
        if bgr2rgb:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            bgr = (bgr-bgr.min())/(bgr.max()-bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])  # [3,482,512]
            mat.close()

        
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)


    

    def __len__(self):
        return self.img_num


#Smoke test to check if code runs
if __name__ == "__main__":
    dir_rgb = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_RGB"
    dir_hsi = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_spectral"
    train_list_dir = "/kaggle/input/datasets/sriramhari14/mst-train-list/train_list.txt"
    dataset = TrainDataset(dir_rgb,dir_hsi,train_list_dir)
    print(TrainDataset.__len__(dataset))
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
    )
