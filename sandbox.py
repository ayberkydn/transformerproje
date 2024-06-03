#%%
# Import necessary libraries
import wandb
import torch
import numpy as np
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import cohen_kappa_score, confusion_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize wandb
wandb.init(project="transformerproje")  
config = wandb.config
config.batch_size = 64
config.image_size = (224, 224)
config.num_epochs = 10
config.learning_rate = 0.001
config.patience = 2
config.val_freq = 10



# Function to create dataloader
def create_dataloader(image_folder_path, batch_size=32, image_size=(224, 224)):
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(image_size),    
        transforms.ToTensor(),            
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset from folder
    dataset = datasets.ImageFolder(root=image_folder_path, transform=transform)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return dataloader

# load pretrained ResNet18
model = models.vit_b_32(weights=models.ViT_B_32_Weights)
model.heads = nn.Linear(768,4)



train_loader = create_dataloader('./data/dummy_set', batch_size=config.batch_size)
val_loader = create_dataloader('./data/dummy_set', batch_size=32)
test_loader = create_dataloader('./data/dummy_set', batch_size=32)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # You can experiment with different optimizers and learning rates



# %%
