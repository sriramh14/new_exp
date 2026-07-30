from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py

class TrainDataset(Dataset):
    def __init__(self, dir_rgb, dir_hsi, train_list_dir, bgr2rgb=True, train = True):

        #List to store hyper and rgb imgs
        self.hypers = []
        self.bgrs = []
        h,w = 482,512  # img shape
        if train:
            hyper_data_path = f'{dir_hsi}/Train_spectral/'
            bgr_data_path = f'{dir_rgb}/Train_RGB/'
        else:
            hyper_data_path = f'{dir_hsi}/Valid_spectral/'
            bgr_data_path = f'{dir_rgb}/Valid_RGB/'

        #Opens list file and saves the list as array
        with open(train_list_dir, 'r') as fin:
            hyper_list = [line.replace('\n','.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr) of ntire2022 dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue

            #Reading hsi file
            with h5py.File(hyper_path, 'r') as mat:
                hyper =np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])

            
            bgr_path = bgr_data_path + bgr_list[i]

            #check to see if rgb and hsi are from same scene
            assert hyper_list[i].split('.')[0] ==bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'

            #Reading rgb file and then converting from bgr to rgb
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)

            #Normalisation? Im not too sure 
            bgr = (bgr-bgr.min())/(bgr.max()-bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])  # [3,482,512]
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            mat.close()
            print(f'Ntire2022 scene {i} is loaded.')
        self.img_num = len(self.hypers)
        self.length = self.img_num

    
    def __getitem__(self, idx):
        bgr = self.bgrs[idx]
        hyper = self.hypers[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.img_num


#Smoke test to check if code runs
if __name__ == "__main__":
    dir_rgb = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_RGB"
    dir_hsi = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_spectral"
    dataset = TrainDataset(dir_rgb,dir_hsi)
    print(TrainDataset.__len__(dataset))
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
    )
