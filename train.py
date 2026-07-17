import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from src.model import RobustAttentionGuidedEdgeViT
from src.dataset import RobustLeafDataset, download_cotton_dataset

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=5):
    best_val_acc = 0.0
    best_epoch = 0
    patience = 5
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        scheduler.step()

        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}')

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_hybrid_vit_model.pth')
            print(f"New best model saved as 'best_hybrid_vit_model.pth' with val_acc: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    return model

def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    dataset_path = download_cotton_dataset()
    
    data_dir = None
    for root, dirs, files in os.walk(dataset_path):
        if "Augmented Dataset" in dirs:
            data_dir = os.path.join(root, "Augmented Dataset")
            break
            
    if data_dir is None:
        print(f"Contents of downloaded path: {os.listdir(dataset_path)}")
        raise FileNotFoundError("Could not find 'Augmented Dataset' folder. Check the printed contents above.")
    
    print(f"✅ Found dataset directory at: {data_dir}")

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = RobustLeafDataset(data_dir, transform=None)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_dataset.dataset.transform = transform_train
    val_dataset.dataset.transform = transform_test
    test_dataset.dataset.transform = transform_test

    num_workers = 0 if os.name == 'nt' else 4
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers)

    model = RobustAttentionGuidedEdgeViT(num_classes=7, edge_mode='sobel')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    print("Starting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10)

if __name__ == "__main__":
    main()
