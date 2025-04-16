import os
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

def extract_and_save_features(data_loader, model, dataset, save_path, device):
    os.makedirs(save_path, exist_ok=True)
    
    all_features = []
    all_labels = []
    all_filenames = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='提取特征'):
            images = images.to(device)
            features = model(images)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            
            batch_filenames = [dataset.samples[i][0] for i in range(len(labels))]
            all_filenames.extend(batch_filenames)
    
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    np.save(os.path.join(save_path, 'features.npy'), all_features)
    np.save(os.path.join(save_path, 'labels.npy'), all_labels)
    
    with open(os.path.join(save_path, 'filenames.txt'), 'w') as f:
        for filename in all_filenames:
            f.write(f"{filename}\n")
    
    print(f"特征已保存到 {save_path}")
    print(f"特征形状: {all_features.shape}, 标签形状: {all_labels.shape}")

def load_features(feature_dir):
    features = np.load(os.path.join(feature_dir, 'features.npy'))
    labels = np.load(os.path.join(feature_dir, 'labels.npy'))
    return features, labels

def create_data_loaders(train_features, train_labels, val_features, val_labels, test_features, test_labels, batch_size=64):
    train_dataset = TensorDataset(
        torch.FloatTensor(train_features), 
        torch.LongTensor(train_labels)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_features), 
        torch.LongTensor(val_labels)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_features), 
        torch.LongTensor(test_labels)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader