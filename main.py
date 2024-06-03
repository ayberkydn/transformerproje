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
nnet = models.resnet18(pretrained=True)
nnet.fc = torch.nn.Identity()  
for param in nnet.parameters():
    param.requires_grad = False


classifier = nn.Linear(512, 4)  # ResNet18 embeddings are 512-dimensional
model = torch.nn.Sequential(
    nnet,
    classifier
)

train_loader = create_dataloader('./data/dummy_set', batch_size=config.batch_size)
val_loader = create_dataloader('./data/dummy_set', batch_size=32)
test_loader = create_dataloader('./data/dummy_set', batch_size=32)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # You can experiment with different optimizers and learning rates


def evaluate(model, dataloader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, predicted = torch.max(logits.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    qwk = cohen_kappa_score(all_labels, all_predictions, weights='quadratic')
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    return qwk, conf_matrix 

# Early stopping parameters
early_stopping_patience = config.patience
best_val_qwk = 0
patience_counter = 0

num_epochs = config.num_epochs
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()

    for images, labels in tqdm.tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Log the loss to wandb
        wandb.log({"batch_loss": loss.item()})

    # Log the average loss per epoch
    avg_train_loss = running_loss / len(train_loader)
    wandb.log({"epoch_loss": avg_train_loss})

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}")

    if (epoch + 1) % config.val_freq == 0:
        val_qwk, _ = evaluate(model, val_loader)
        wandb.log({"val_qwk": val_qwk})
        print(f"Validation QWK: {val_qwk}")

        # Check for early stopping
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

#test
test_qwk, test_cmatrix = evaluate(model, test_loader)
wandb.log({"test_qwk": test_qwk, "confusion_matrix": test_cmatrix})



wandb.finish()
