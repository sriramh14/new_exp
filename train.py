#Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#File imports
from dataset.dataset_loader import TrainDataset

#Model import
from models.MST_Plus_Plus import MST_Plus_Plus

#Loss import
from loss.mrae import mrae


#Config variables
train_dir_rgb = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_RGB"
train_dir_hsi = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_spectral"
val_dir_rgb = "/kaggle/input/datasets/sriramhari14/ntire-2022/Valid_RGB"
val_dir_hsi = "/kaggle/input/datasets/sriramhari14/ntire-2022/Valid_spectral"
train_list_dir = "dataset/train_list.txt"
valid_list_dir = "dataset/valid_list.txt"
SEED = 42
BATCH_SIZE = 4
SHUFFLE = True
NUM_WORKERS = 4
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
LEARNING_RATE_MIN = 1e-6
PIN_MEMORY = True
IN_CHANNELS = 3
OUT_CHANNELS = 31
N_FEAT = 31
STAGE = 1


def main() -> None:

  #CUDA device
  device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
  )

  #Creating train and valid datasets
  print("Creating train dataset\n")
  train = TrainDataset(
    dir_rgb = train_dir_rgb,
    dir_hsi = train_dir_hsi,
    train_list_dir = train_list_dir,
    train = True
  )
  print("Creating validation dataset\n")
  val = TrainDataset(
    dir_rgb = val_dir_rgb,
    dir_hsi = val_dir_hsi,
    train_list_dir = valid_list_dir,
    train = False
  )

  #Creating dataloader objects of val and train
  train_loader = DataLoader(
    train,batch_size = BATCH_SIZE,
    shuffle = SHUFFLE,
    num_workers = NUM_WORKERS,
    pin_memory=PIN_MEMORY
  )
  val_loader = DataLoader(
    val,
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers = NUM_WORKERS,
    pin_memory=PIN_MEMORY
  )

  #Model instantiation
  model = MST_Plus_Plus(
    in_channels = IN_CHANNELS,
    out_channels = OUT_CHANNELS,
    n_feat = N_FEAT,
    stage = STAGE
  )
  model = model.to(device)


  #Loss
  criterion = mrae()
  criterion.to(device)

  #Optimiser and LR scheduler
  optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
  )  
  scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max = NUM_EPOCHS,
    eta_min = LEARNING_RATE_MIN
  )
  best_loss = float('inf')
  #Training loop
  for epoch in range(NUM_EPOCHS):
    
    print(f"Starting epoch number:{epoch+1}")

    
    model.train()

    running_loss = 0
    correct = 0
    total = 0

    #Train
    for rgb, hsi in train_loader:

        rgb = rgb.to(device)
        hsi = hsi.to(device)

        #Resetting gradients to zero
        optimizer.zero_grad()

        outputs = model(rgb)

        loss = criterion(pred = hsi,target = outputs)

        loss.backward()

        optimizer.step()

        running_loss += loss.detach().float()

    train_loss = running_loss.item() / len(train_loader)

    scheduler.step()

    #Validation 
    model.eval()

    val_loss = 0
    correct = 0
    total = 0

    #Disabling grad
    with torch.no_grad():
    
        for rgb,hsi in val_loader:
    
            rgb = rgb.to(device)
            hsi = hsi.to(device)

    
            outputs = model(rgb)
    
            loss = criterion(pred = hsi,target = outputs)
    
            val_loss += loss.item()
    
    
    val_loss /= len(val_loader)

    #Save model if validation metrics are highest its ever been
    if val_loss < best_loss:
      best_loss = val_loss
      print("Saving model\n")
      torch.save(
          model.state_dict(),
          "best_model.pth"
      )

    #Logging metrics
    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
    )
  
  
if __name__ == "__main__":
  main()
  
