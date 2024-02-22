import pandas as pd
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import math
import os

class CustomDataset(Dataset):
    def __init__(self, dataset_tuple: tuple, image_size: int, classes: list) -> None:
        super().__init__()
        self.file_paths, self.labels = dataset_tuple
        self.classes = classes
        self.image_size = image_size
        self.transformation = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path)
        return self.transformation(image), label

class CustomDatasetCreator:
    def __init__(self, dataset_path: str, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
        self.dataset_path = Path(dataset_path)
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.labels = [label for label in os.listdir(self.dataset_path) if os.path.isdir(self.dataset_path.joinpath(label))]

    def create_dataset(self, random_state: int = None) -> tuple:
        # Load file paths and labels
        all_files = []
        all_labels = []
        for i, label in enumerate(self.labels):
            image_files = list(self.dataset_path.joinpath(label).glob('*'))
            all_files.extend(image_files)
            all_labels.extend([i] * len(image_files))
        
        # Split dataset into train, test, and validation sets
        train_files, test_val_files, train_labels, test_val_labels = train_test_split(all_files, all_labels, test_size=self.test_ratio+self.val_ratio, random_state=random_state)
        val_files, test_files, val_labels, test_labels = train_test_split(test_val_files, test_val_labels, test_size=self.test_ratio/(self.test_ratio+self.val_ratio), random_state=random_state)
        
        return (train_files, train_labels), (test_files, test_labels), (val_files, val_labels), self.labels


#test classes
# train = CustomDataset('./custom_dataset/train', 72, True)
# print(train.get_labels())
# train_dataset = DataLoader(train, batch_size=2, shuffle=True)
# for img, label in iter(train_dataset):
#     print(img.shape, label)
#     plt.figure(1, figsize=(5,5))
#     # Assuming your image size is 500x500
#     plt.imshow((img[1].permute(1, 2, 0).cpu().numpy() + 1) / 2)  # De-normalize and permute dimensions
#     plt.show()
#     break

# dataset = MNISTDataset('./MNIST_dataset', 2, 72)
# MNIST_train_dataset, MNIST_test_dataset = dataset.load_dataset()
# print(MNIST_test_dataset)
# for img1, label in iter(MNIST_train_dataset):
#     print(img1.shape, label)
#     plt.figure(1, figsize=(10,10)) 
#     plt.imshow((img1[1].squeeze()).cpu().numpy(), cmap='gray')  # Display the image using imshow and specify the colormap
#     plt.show()
#     break