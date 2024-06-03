#%%
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#function definitions
def create_dataloader(image_folder_path, batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor(),      
    ])

    dataset = datasets.ImageFolder(root=image_folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return dataloader
def calculate_mean_and_variance(images):
    
    red_channel = images[:, 0, :, :]
    green_channel = images[:, 1, :, :]
    blue_channel = images[:, 2, :, :]
    
    # calculate mean and variance for each channel
    means = {
        'red_mean': np.mean(red_channel),
        'green_mean': np.mean(green_channel),
        'blue_mean': np.mean(blue_channel)
    }
    
    variances = {
        'red_variance': np.var(red_channel),
        'green_variance': np.var(green_channel),
        'blue_variance': np.var(blue_channel)
    }
    
    results = {**means, **variances}
    
    return results


def descriptive_stats(images, labels):
    images_by_labels = [[],[],[],[]]
    for img,label in zip(images, labels):
        images_by_labels[label].append(img)
    
    results = []
    for n in range(4):
        imgs = np.array(images_by_labels[n])
        res = calculate_mean_and_variance(imgs)
        results.append(res)
    
    return results
#%%

#Train+validation set descriptive statistics
loader = create_dataloader('./data/train_and_validation_sets')

images = []
labels = []

for image, label in loader:
    images.append(image.numpy()[0])
    labels.append(label.numpy()[0])

images = np.array(images)
labels = np.array(labels)

stats = descriptive_stats(images, labels) 
plt.hist(labels, bins=4, range=(0, 3), alpha=0.75, edgecolor='black', align='mid')
plt.title('Histogram of label distribution for training and validation sets')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.xticks([0, 1, 2, 3])

plt.show()
for n in range(4):
    results = stats[n]
    print(f'Mean and variance for each color channel for training+validation set images with label {n}')
    print(f'Red channel mean: {results["red_mean"]}')
    print(f'Blue channel mean: {results["blue_mean"]}')
    print(f'Green channel mean: {results["green_mean"]}')
    
    print(f'Red channel variance: {results["red_variance"]}')
    print(f'Blue channel variance: {results["blue_variance"]}')
    print(f'Green channel variance: {results["green_variance"]}')
    print('--------------')
#%%
#Test set descriptive statistics
print('----------------------')
print('----------------------')
print('----------------------')
loader = create_dataloader('./data/test_set')

images = []
labels = []

for image, label in loader:
    images.append(image.numpy()[0])
    labels.append(label.numpy()[0])

images = np.array(images)
labels = np.array(labels)

stats = descriptive_stats(images, labels) 

plt.hist(labels, bins=4, range=(0, 3), alpha=0.75, edgecolor='black', align='mid')
plt.title('Histogram of label distribution for test set')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.xticks([0, 1, 2, 3])
plt.show()
for n in range(4):
    results = stats[n]
    print(f'Mean and variance for each color channel for test set images with label {n}')
    print(f'Red channel mean: {results["red_mean"]}')
    print(f'Blue channel mean: {results["blue_mean"]}')
    print(f'Green channel mean: {results["green_mean"]}')
    
    print(f'Red channel variance: {results["red_variance"]}')
    print(f'Blue channel variance: {results["blue_variance"]}')
    print(f'Green channel variance: {results["green_variance"]}')
    print('--------------')
# %%
