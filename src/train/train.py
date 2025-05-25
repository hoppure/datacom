import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


def train(model, train_loader, optimizer, criterion, device, backbone_name, epoch):
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"[{backbone_name} Epoch {epoch+1}/Train]"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)
