import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt

class CustomImageFolderDataset(Dataset):
    """
    Custom Dataset that combines ImageFolder with custom targets
    """
    def __init__(self, image_folder, targets_dict, transform=None):
        self.image_folder = image_folder
        self.targets_dict = targets_dict
        self.transform = transform
        
        # Create mapping between folder names and targets
        self.targets = []
        self.image_paths = []
        
        for class_name in os.listdir(image_folder.root):
            class_path = os.path.join(image_folder.root, class_name)
            if os.path.isdir(class_path):
                # Ensure the class name exists in targets_dict
                if class_name in targets_dict:
                    class_target = targets_dict[class_name]
                    
                    # Get all image paths for this class
                    for img_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, img_name)
                        self.image_paths.append(img_path)
                        self.targets.append(class_target)
                else:
                    print(f"Warning: No target found for class {class_name}")
        
        # Process targets to ensure consistent length
        self.processed_targets = self._process_targets()

    def _flatten_target(self, target):
        """Recursively flatten nested lists into a single list of floats"""
        flattened = []
        for item in target:
            if isinstance(item, list):
                flattened.extend(self._flatten_target(item))
            else:
                flattened.append(float(item))
        return flattened

    def _process_targets(self):
        """
        Process targets into a consistent numpy array, 
        padding shorter targets with zeros
        """
        # Find max length of flattened targets
        max_length = max(len(self._flatten_target(target)) for target in self.targets)
        
        # Process and pad targets
        processed_targets = []
        for target in self.targets:
            flat_target = self._flatten_target(target)
            # Pad or truncate to max length
            padded_target = (
                flat_target + [0.0] * (max_length - len(flat_target))
            )[:max_length]
            processed_targets.append(padded_target)
        
        return torch.FloatTensor(processed_targets)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        target = self.processed_targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target
    
class FlexibleRegressor(nn.Module):
    """
    Neural network that can handle variable-length output regression
    """
    def __init__(self, input_features=2048, max_output_length=8):
        super(FlexibleRegressor, self).__init__()
        
        # Pretrained ResNet50 as feature extractor
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze early layers
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Flexible regression head
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, max_output_length)
        )

    def forward(self, x):
        features = self.features(x)
        return self.regression_head(features)

class RegressionTrainer:
    def __init__(self, targets_dict, data_dir, max_output_length=8, device=None):
        # Set device
        self.device = device or (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Targets dictionary
        self.targets_dict = targets_dict
        
        # Transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Image folder dataset
        self.image_folder = datasets.ImageFolder(data_dir)
        
        # Create custom dataset
        self.dataset = CustomImageFolderDataset(
            self.image_folder, 
            targets_dict, 
            transform=self.train_transform
        )
        
        # Split dataset
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        
        # Initialize model
        self.model = FlexibleRegressor(
            max_output_length=max_output_length
        ).to(self.device)

    def train(self, epochs=50, batch_size=16, learning_rate=0.001):
        # Data loaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5
        )
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for images, targets in train_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}')
            print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        
        return self.model

    def predict(self, image_dir):
        """
        Predict for all images in a nested folder structure.
        
        Args:
            image_dir (str): Path to the directory containing subdirectories of images.
            
        Returns:
            dict: A dictionary mapping relative image paths to predictions.
        """
        self.model.eval()
        
        predictions = {}
        # Traverse the directory and its subdirectories
        for root, _, files in os.walk(image_dir):
            for file_name in files:
                image_path = os.path.join(root, file_name)
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Ensure it's an image
                    try:
                        # Load and preprocess the image
                        image = Image.open(image_path).convert('RGB')
                        image = self.val_transform(image)
                        image = image.unsqueeze(0).to(self.device)

                        # Make prediction
                        with torch.no_grad():
                            prediction = self.model(image)

                        # Use a relative path for output
                        relative_path = os.path.relpath(image_path, image_dir)
                        predictions[relative_path] = [round(float(p), 3) for p in prediction.cpu().numpy()[0]]
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
        
        return predictions

    def save_model(self, path='regression_model.pth'):
        """Save the trained model"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

def main():
    # Your targets dictionary
    targets = {
        'Slab 1': [4.0, 7.5, 21.25, 6.0],
        'Slab 2': [4.0, [7, 7], 21.25, 6.0],
        'Slab 3': [4.0, [5.5, 4.25, 3.5], 21.25, 6.0],
        'Slab 4': [3.0, [8.0, 8.0, 8.0, 8.0], 34.5, 6.0],
        'Slab 5': [[2.0, 2.0], 8.25, 34.5, 12.0],
        'Slab 6': [6.2272, 24, 9.0, 6.0, 0.70, 3.2015, 2.0],
    }

    # Add zero-padding to make it symmetric
    targets = {
        'Slab 1': [4.0, 7.5, 21.25, 6.0, 0.0, 0.0, 0.0],
        'Slab 2': [4.0, [7, 7], 21.25, 6.0, 0.0, 0.0],
        'Slab 3': [4.0, [5.5, 4.25, 3.5], 21.25, 6.0, 0.0],
        'Slab 4': [3.0, [8.0, 8.0, 8.0, 8.0], 34.5, 6.0],
        'Slab 5': [[2.0, 2.0], 8.25, 34.5, 12.0, 0.0, 0.0],
        'Slab 6': [6.2272, 24, 9.0, 6.0, 0.70, 3.2015, 2.0],
    }

    # Path to your image folder
    # Folder structure should be:
    # data_dir/
    #   1/
    #     image1.jpg
    #     image2.jpg
    #   2/
    #     image1.jpg
    #     ...
    data_dir = '/mnt/d/OneDrive - Rowan University/RA/Fall 24/Civil/dataset/Train/'

    # Save targets to a JSON file for reference
    with open('targets.json', 'w') as f:
        json.dump(targets, f, indent=2)
    
    # Initialize and train
    trainer = RegressionTrainer(
        targets_dict=targets, 
        data_dir=data_dir,
        max_output_length=7  # Adjust based on max target length
    )
    
    # Train the model
    model = trainer.train(
        epochs=50,
        batch_size=16,
        learning_rate=0.001
    )
    
    # Save the model
    trainer.save_model('regression_model.pth')
    
    # Example prediction (replace with an actual image path)
    predictions = trainer.predict('/mnt/d/OneDrive - Rowan University/RA/Fall 24/Civil/dataset/Test/')
    print("Prediction:", predictions)

    # Save predictions to JSON
    with open('predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {'predictions.json'}")

if __name__ == '__main__':
    main()

