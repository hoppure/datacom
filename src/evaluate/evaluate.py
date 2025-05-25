import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import log_loss


def evaluate(model, val_loader, criterion, device, backbone_name, epoch, class_names):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"[{backbone_name} Epoch {epoch+1}/Val]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    val_logloss = log_loss(all_labels, all_probs, labels=list(range(len(class_names))))
    return avg_val_loss, val_accuracy, val_logloss
