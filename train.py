#Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#File imports
from dataset.dataset_loader import TrainDataset

#Model import
from model.classifier import ASLClassifier


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
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
LEARNING_RATE_MIN = 1e-6
PIN_MEMORY = True



def main() -> None:

  #CUDA device
  device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
  )

  #Creating train and valid datasets
  train = TrainDataset(train_rgb_dir,train_hsi_dir,train_list_dir)
  val = TrainDataset(val_rgb_dir,val_hsi_dir,val_list_dir)

  #Creating dataloader objects of val and train
  train_loader = DataLoader(train,batch_size = BATCH_SIZE, shuffle = SHUFFLE, num_workers = NUM_WORKERS,pin_memory=PIN_MEMORY)
  val_loader = DataLoader(val,batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS,pin_memory=PIN_MEMORY)

  #Model instantiation
  model = ASLClassifier(num_classes = NUM_CLASSES)
  model = model.to(device)


  #Loss
  criterion = nn.CrossEntropyLoss()

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
  best_acc = 0.0
  #Training loop
  for epoch in range(NUM_EPOCHS):
    
    print(f"Starting epoch number:{epoch+1}")

    
    model.train()

    running_loss = 0
    correct = 0
    total = 0

    #Train
    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        #Resetting gradients to zero
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.detach().float()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    train_loss = running_loss.item() / len(train_loader)
    train_acc = correct / total

    scheduler.step()

    #Validation 
    model.eval()

    val_loss = 0
    correct = 0
    total = 0

    #Disabling grad
    with torch.no_grad():
    
        for images, labels in val_loader:
    
            images = images.to(device)
            labels = labels.to(device)
    
            outputs = model(images)
    
            loss = criterion(outputs, labels)
    
            val_loss += loss.item()
    
            predictions = outputs.argmax(dim=1)
    
            correct += (predictions == labels).sum().item()
    
            total += labels.size(0)
    
    val_loss /= len(val_loader)
    val_acc = correct / total

    #Save model if validation metrics are highest its ever been
    if val_acc > best_acc:
      best_acc = val_acc
  
      torch.save(
          model.state_dict(),
          "best_model.pth"
      )

    #Logging metrics
    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )
  
  
if __name__ == "__main__":
  main()
  
