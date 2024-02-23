from dataset import CustomDatasetCreator, CustomDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTModel
import torch.optim as optim
import torch.nn as nn
from vit_model.vit import Vit
import torch
import pandas as pd
from pathlib import Path
import logging
import os
import math

class Trainer:
    def __init__(self, config):
        self.training_loss = []
        self.training_accuracy = []
        self.validation_loss = []
        self.validation_accuracy = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.criterion = nn.CrossEntropyLoss()
        if config.custom_dataset:
            self.model = Vit(classes=config.classes, blocks=config.encoder_blocks,
                             channels=config.channels, height=config.image_size,
                             width=config.image_size, patch_size=config.patch_size,
                             H=config.H, inner_dim=config.inner_dim, dropout=config.dropout)
            if not os.path.exists(Path(config.dataset_path)):
                os.mkdir(Path(config.dataset_path))
            dataset_obj = CustomDatasetCreator(config.dataset_path, 0.7, 0.2, 0.1)
            self.train_dataset, self.test_dataset, self.val_dataset, self.classes = dataset_obj.create_dataset(random_state=42)
            self.train_set = CustomDataset(self.train_dataset, config.image_size, self.classes)
            self.test_set = CustomDataset(self.test_dataset, config.image_size, self.classes)
            self.val_set = CustomDataset(self.val_dataset, config.image_size, self.classes)
            self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True)
            self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False)
            self.val_loader = DataLoader(self.val_set, batch_size=config.batch_size, shuffle=True)
        else:
            mnist_dataset_obj = datasets.MNIST(root=config.mnist_path, train=True, download=True,
                                               transform=transforms.Compose([
                                                   transforms.Resize((config.image_size, config.image_size)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5,), (0.5,))
                                               ]))
            self.train_loader = DataLoader(mnist_dataset_obj, batch_size=config.batch_size, shuffle=True)
            mnist_dataset_obj = datasets.MNIST(root=config.mnist_path, train=False, download=True,
                                               transform=transforms.Compose([
                                                   transforms.Resize((config.image_size, config.image_size)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5,), (0.5,))
                                               ]))
            self.val_loader = DataLoader(mnist_dataset_obj, batch_size=config.batch_size, shuffle=False)

            self.model = Vit(classes=config.mnist_classes, blocks=config.encoder_blocks,
                             channels=config.mnist_channels, height=config.image_size,
                             width=config.image_size, patch_size=config.patch_size,
                             H=config.H_mnist, inner_dim=config.inner_dim, dropout=config.dropout)
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate, momentum=config.momentum)

    def trainer(self, config):
        logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        if config.load_weights:
            checkpoint_path = Path(config.model_weights_path).joinpath("best.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Best weights and optimizer parameters are loaded")

            else:
                if not os.path.exists(Path(config.pre_trained_model_path)):
                    os.mkdir(Path(config.pre_trained_model_path))
                pretrained_model = ViTModel.from_pretrained('google/vit-base-patch16-224', cache_dir=Path(config.pre_trained_model_path))
                state_dict = pretrained_model.state_dict()
                state_dict = self.rename_state_dict_keys(state_dict)
                self.model.load_state_dict(state_dict, strict=False)  # Set strict=False to ignore any mismatch in keys
                print("Pre-trained weights are loaded")
        else:
            print("Training from scratch")
        print("Device: ", self.device)
        print("Training data: ", self.train_set.__len__())
        print("validation data: ", self.val_set.__len__())
        print("Test data: ", self.test_set.__len__())
        print("----Training started----")
        # total_batches = math.floor(self.train_set.__len__() / config.batch_size)
        best_loss = float("inf")
        for epoch in range(config.epochs):
            print(f"{epoch}: Epoch")
            self.model.train()
            correct_predictions = 0
            total_samples = 0
            losses = []
            # i = 0
            for img, label in iter(self.train_loader):
                # i += 1
                self.optimizer.zero_grad()
                # print(img.shape)
                img, label = torch.tensor(img), torch.tensor(label)
                # print(img.shape)
                img = img.to(self.device)
                label = label.to(self.device)
                self.model.to(self.device)
                pred = self.model(img)
                loss = self.criterion(pred, label)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                # Calculate accuracy
                _, predicted_labels = torch.max(pred, 1)
                correct_predictions += (predicted_labels == label).sum().item()
                total_samples += label.size(0)
                # if total_batches == i:
                #   break
            # print(i)
            self.training_loss.append(sum(losses) / len(losses))
            self.training_accuracy.append(correct_predictions / total_samples)
            
            # Validation
            losses = []
            correct_predictions = 0
            total_samples = 0
            self.model.eval()
            with torch.no_grad():
                # i = 0
                # total_batches = math.floor(self.val_set.__len__() / config.batch_size)
                for img, label in iter(self.val_loader):
                    # i += 1
                    img, label = torch.tensor(img), torch.tensor(label)
                    img = img.to(self.device)
                    label = label.to(self.device)
                    self.model.to(self.device)
                    pred = self.model(img)
                    loss = self.criterion(pred, label)
                    losses.append(loss.item())
                    # Calculate accuracy
                    _, predicted_labels = torch.max(pred, 1)
                    correct_predictions += (predicted_labels == label).sum().item()
                    total_samples += label.size(0)
                    # if i == total_batches:
                    #   break
            self.validation_loss.append(sum(losses) / len(losses))
            self.validation_accuracy.append(correct_predictions / total_samples)

            logging.info(f'Epoch: {epoch}, Training Loss: {self.training_loss[-1]}, Training Accuracy: {self.training_accuracy[-1]}, Validation Loss: {self.validation_loss[-1]}, Validation Accuracy: {self.validation_accuracy[-1]}')
            print(f'Epoch: {epoch}, Training Loss: {self.training_loss[-1]}, Training Accuracy: {self.training_accuracy[-1]}, Validation Loss: {self.validation_loss[-1]}, Validation Accuracy: {self.validation_accuracy[-1]}')

            # Save the model if the validation loss improves
            if (self.training_loss[-1] < best_loss) and (self.validation_loss[-1] < best_loss):
                if not os.path.exists(Path(config.model_weights_path)):
                    os.mkdir(Path(config.model_weights_path))
                print("Saving model...")
                best_loss = self.training_loss[-1]
                torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                        }, Path(config.model_weights_path).joinpath("best.pth"))
                print("---Best weights and optimizer parameters are saved---")
    def test(self, config):
        if config.custom_dataset:
            losses = []
            self.model.eval()
            correct_predictions = 0
            total_samples = 0
            with torch.no_grad():
                for img, label in iter(self.test_loader):
                        img, label = torch.tensor(img), torch.tensor(label)
                        img = img.to(self.device)
                        label = label.to(self.device)
                        self.model.to(self.device)
                        pred = self.model(img)
                        loss = self.criterion(pred, label)
                        losses.append(loss)
                        _, predicted_labels = torch.max(pred, 1)
                        correct_predictions += (predicted_labels == label).sum().item()
                        total_samples += label.size(0)
            print("Test loss: ", sum(losses) / len(losses))
        else:
            print("No Test Data!")
    
    def convert_model_to_onnx(self, config):
        if not os.path.exists(Path(config.onnx_model_path)):
            os.mkdir(Path(config.onnx_model_path))
        input_sample = torch.randn(1, 3, 224, 224)
        torch.onnx.export(self.model,
                        input_sample, 
                        Path(config.onnx_model_path).joinpath("model.onnx"),
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output'],
                        dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})
        print("Model is converted to onnx and saved")
    
    def rename_state_dict_keys(self, state_dict:dict) -> dict:
        new_state_dict = {}
        out_count = 0
        msa_out_count = 0
        for name, param in state_dict.items():
            lis = name.split('.')
            if 'patch_embeddings' in lis:
                lis.remove('patch_embeddings')
            if 'layer' in lis:
                lis.remove('layer')
            if 'attention' in lis:
                lis.remove('attention')
            if 'attention' in lis:
                lis[lis.index('attention')] = 'msa'
            if 'dense' in lis:
                lis.remove('dense')
            if 'output' in lis:
                if msa_out_count < 2:
                    lis[lis.index('output')] = 'msa.output'
                    msa_out_count += 1
                else:
                    out_count += 1
                    if out_count == 2:
                        out_count = 0
                        msa_out_count = 0
            name = '.'.join(lis)
            new_state_dict[name] = param
            
        return new_state_dict

    def save_to_dataframe(self,config):
        data = {
            'training_loss': self.training_loss,
            'training_accuracy': self.training_accuracy,
            'validation_loss': self.validation_loss,
            'validation_accuracy': self.validation_accuracy
        }
        df = pd.DataFrame(data)
        if not os.path.exists(Path(config.result_folder_path)):
            os.mkdir(Path(config.result_folder_path))
        df.to_csv(Path(config.result_folder_path).joinpath("results.csv"), index=False)
        df_classes = pd.DataFrame(self.classes)
        df_classes.to_csv(Path(config.result_folder_path).joinpath("classes.csv"), index=False)





