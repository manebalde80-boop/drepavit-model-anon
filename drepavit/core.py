# code extracted from notebook

from google.colab import drive
drive.mount('/content/drive')

# ----

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import timm
import torch.cuda.amp as amp
import random
from torchvision.utils import make_grid
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ----

class Config:
    TRAIN_DIR = '/content/drive/MyDrive/Drepa_Vite/Dataset_drépano/train'
    VAL_DIR   = '/content/drive/MyDrive/Drepa_Vite/Dataset_drépano/val'
    TEST_DIR  = '/content/drive/MyDrive/Drepa_Vite/Dataset_drépano/test'

    IMG_SIZE = 224
    NUM_CLASSES = 3
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=self.alpha)

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma) * -logpt
        return loss.mean()

# ----

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(Config.TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(Config.VAL_DIR, transform=val_test_transforms)
test_dataset = datasets.ImageFolder(Config.TEST_DIR, transform=val_test_transforms)

class_counts = torch.bincount(torch.tensor([label for _, label in train_dataset]))
class_weights = (1. / class_counts.float()).to(Config.DEVICE)
class_weights = class_weights / class_weights.sum()

sample_weights = [class_weights[label] for _, label in train_dataset]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, sampler=sampler, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

# ----

def count_images_in_folder(folder):
    total = 0
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            total += len(os.listdir(label_folder))
    return total

train_count = count_images_in_folder(Config.TRAIN_DIR)
val_count = count_images_in_folder(Config.VAL_DIR)
test_count = count_images_in_folder(Config.TEST_DIR)

print(f"Number of images:")
print(f" - Training: {train_count}")
print(f" - Validation: {val_count}")
print(f" - Test: {test_count}")

def visualize_batch(dataloader, title, class_names):
    images, labels = next(iter(dataloader))
    images = images[:8]
    labels = labels[:8]

    inv_norm = transforms.Normalize(
        mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1/s for s in [0.229, 0.224, 0.225]]
    )
    images = torch.stack([inv_norm(img) for img in images])

    grid = make_grid(images, nrow=4)
    plt.figure(figsize=(12, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title + " | " + ", ".join([class_names[label] for label in labels]))
    plt.axis('off')
    plt.show()

class_names = train_dataset.classes

visualize_batch(train_loader, "Training Images", class_names)
visualize_batch(val_loader, "Validation Images", class_names)
visualize_batch(test_loader, "Test Images", class_names)

# ----

idx = random.randint(0, len(test_dataset) - 1)
image_tensor, label = test_dataset[idx]

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
image_tensor_denorm = image_tensor * std + mean
image_np = image_tensor_denorm.permute(1, 2, 0).numpy()

patch_size = 16
patches = []

for i in range(0, image_np.shape[0], patch_size):
    for j in range(0, image_np.shape[1], patch_size):
        patch = image_np[i:i+patch_size, j:j+patch_size, :]
        patches.append(patch)

fig = plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title("Selected Image")
plt.axis("off")

num_patches_per_row = image_np.shape[0] // patch_size
fig_patches, axs = plt.subplots(num_patches_per_row, num_patches_per_row, figsize=(6, 6))

for idx, patch in enumerate(patches):
    row = idx // num_patches_per_row
    col = idx % num_patches_per_row
    axs[row, col].imshow(patch)
    axs[row, col].axis("off")

fig_patches.suptitle("Extracted Patches (Vision Transformer", fontsize=14)
plt.tight_layout()
plt.show()

# ----

# MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.NUM_HEADS
        self.head_dim = config.HIDDEN_DIM // config.NUM_HEADS

        self.qkv = nn.Linear(config.HIDDEN_DIM, 3 * config.HIDDEN_DIM)
        self.proj = nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.head_dim**-0.5
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

# ----

# Bloc MLP
class MLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.HIDDEN_DIM, config.MLP_DIM)
        self.fc2 = nn.Linear(config.MLP_DIM, config.HIDDEN_DIM)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# ----

# Bloc Transformer
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.HIDDEN_DIM)
        self.attn = MultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(config.HIDDEN_DIM)
        self.mlp = MLPBlock(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ----

class HybridModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Branche CNN
        self.cnn = timm.create_model(
            config.CNN_BACKBONE,
            pretrained=True,
            features_only=True,
            out_indices=[4]
        )

        # Projection CNN
        with torch.no_grad():
            dummy = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
            cnn_feat = self.cnn(dummy)[0]
            self.cnn_feat_dim = cnn_feat.shape[1]

        self.cnn_proj = nn.Sequential(
            nn.Conv2d(self.cnn_feat_dim, config.HIDDEN_DIM, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # Branche Transformer
        self.patch_embed = nn.Conv2d(
            config.NUM_CHANNELS,
            config.HIDDEN_DIM,
            kernel_size=config.PATCH_SIZE,
            stride=config.PATCH_SIZE
        )

        num_patches = (config.IMG_SIZE // config.PATCH_SIZE) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, config.HIDDEN_DIM))
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.HIDDEN_DIM))
        self.dropout = nn.Dropout(config.DROPOUT)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.NUM_LAYERS)
        ])

        self.norm = nn.LayerNorm(config.HIDDEN_DIM)
        self.head = nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES)

    def forward(self, x):
        # Extraction des caracteristique des CNN
        cnn_features = self.cnn_proj(self.cnn(x)[0])

        # Passage au Transformer
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = x[:, 0]

        # Fusion Tranformer + CNN
        x = self.norm(x + cnn_features)
        return self.head(x)

# ----

Config.CNN_BACKBONE = 'resnet50'
Config.HIDDEN_DIM = 256
Config.MLP_DIM = 512
Config.NUM_HEADS = 4
Config.NUM_LAYERS = 4
Config.DROPOUT = 0.1
Config.NUM_CHANNELS = 3
Config.PATCH_SIZE = 16

model = HybridModel(Config).to(Config.DEVICE)

criterion = FocalLoss(alpha=class_weights, gamma=2.0).to(Config.DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
scaler = amp.GradScaler()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

# ----

from torchinfo import summary
from thop import profile

# ----

# --- Résumé & FLOPs (⚠️ remplace Cfg par Config) ---
from torchinfo import summary
from thop import profile

device_str = "cuda" if torch.cuda.is_available() else "cpu"
summary(model, input_size=(1, 3, Config.IMG_SIZE, Config.IMG_SIZE), device=device_str)

model.eval()
dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE, device=Config.DEVICE)
with torch.no_grad():
    flops, params = profile(model, inputs=(dummy,), verbose=False)

print(f"\nParamètres: {params/1e6:.2f} M")
print(f"FLOPs (forward, bs=1): {flops/1e9:.2f} GFLOPs")
print(f"FLOPs par batch (bs={Config.BATCH_SIZE}): {(flops*Config.BATCH_SIZE)/1e9:.2f} GFLOPs")
print("Note: l'entraînement (forward+backward) ≈ 2–3× ces FLOPs.")

# ----

def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
        optimizer.zero_grad()
        with amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), 100. * correct / total

# ----

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), 100. * correct / total, all_preds, all_labels

# ----

train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

for epoch in range(Config.NUM_EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler)
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

epochs = range(1, Config.NUM_EPOCHS + 1)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ----

# 2

# ----

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import numpy as np

def full_evaluation(model, loader, criterion, name=""):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    sensitivity = np.mean(np.diag(cm) / np.maximum(1, np.sum(cm, axis=1)))  # Rappel moyen
    specificity = np.mean([
        (np.sum(cm) - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])) /
        max(1, (np.sum(cm) - cm[:, i].sum()))
        for i in range(len(cm))
    ])

    try:
        one_hot_labels = label_binarize(all_labels, classes=np.unique(all_labels))
        auc = roc_auc_score(one_hot_labels, all_probs, average='macro', multi_class='ovo')
    except Exception as e:
        print(f" Erreur AUC ({name}) :", e)
        auc = float('nan')

    print(f"\n Évaluation sur {name} :")
    print(f"Loss        : {total_loss / len(loader):.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1-Score    : {f1:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"MCC         : {mcc:.4f}")
    print(f"AUC         : {auc:.4f}")

# ----

# Bonne
full_evaluation(model, train_loader, criterion, name="Entraînement")

# ----

full_evaluation(model, train_loader, criterion, name="Entraînement")

# ----

# Bonne
full_evaluation(model, val_loader, criterion, name="Validation")

# ----

# 2
full_evaluation(model, val_loader, criterion, name="Validation")

# ----

full_evaluation(model, val_loader, criterion, name="Validation")

# ----

# bonne
full_evaluation(model, test_loader, criterion, name="Test")

# ----

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, roc_auc_score, classification_report
)
from sklearn.preprocessing import label_binarize
import numpy as np

def full_evaluation(model, loader, criterion, name=""):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    sensitivity = np.mean(np.diag(cm) / np.maximum(1, np.sum(cm, axis=1)))
    specificity = np.mean([
        (np.sum(cm) - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])) /
        max(1, (np.sum(cm) - cm[:, i].sum()))
        for i in range(len(cm))
    ])

    try:
        one_hot_labels = label_binarize(all_labels, classes=np.unique(all_labels))
        auc = roc_auc_score(one_hot_labels, all_probs, average='macro', multi_class='ovo')
    except Exception as e:
        print(f" Erreur AUC ({name}) :", e)
        auc = float('nan')

    print(f"\n Évaluation sur {name} :")
    print(f"Loss        : {total_loss / len(loader):.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1-Score    : {f1:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"MCC         : {mcc:.4f}")
    print(f"AUC         : {auc:.4f}")

    target_names = ["Autres", "Falciformes", "Normales"]
    print("\n Rapport de classification :\n")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=2))

# ----

#2
full_evaluation(model, train_loader, criterion, name="Entraînement")
full_evaluation(model, val_loader, criterion, name="Validation")
full_evaluation(model, test_loader, criterion, name="Test")

# ----

full_evaluation(model, train_loader, criterion, name="Entraînement")
full_evaluation(model, val_loader, criterion, name="Validation")
full_evaluation(model, test_loader, criterion, name="Test")

# ----

import matplotlib.pyplot as plt
import seaborn as sns

def full_evaluation(model, loader, criterion, name=""):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # --- Metrics ---
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    sensitivity = np.mean(np.diag(cm) / np.maximum(1, np.sum(cm, axis=1)))
    specificity = np.mean([
        (np.sum(cm) - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])) /
        max(1, (np.sum(cm) - cm[:, i].sum()))
        for i in range(len(cm))
    ])

    try:
        one_hot_labels = label_binarize(all_labels, classes=np.unique(all_labels))
        auc = roc_auc_score(one_hot_labels, all_probs, average='macro', multi_class='ovo')
    except Exception as e:
        print(f" Erreur AUC ({name}) :", e)
        auc = float('nan')

    # --- Print results ---
    print(f"\n Évaluation sur {name} :")
    print(f"Loss        : {total_loss / len(loader):.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1-Score    : {f1:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"MCC         : {mcc:.4f}")
    print(f"AUC         : {auc:.4f}")

    target_names = ["Autres", "Falciformes", "Normales"]
    print("\n Rapport de classification :\n")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=2))

    # --- Matrice de confusion ---
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités")
    plt.title(f"Matrice de confusion - {name}")
    plt.show()

# ----

import matplotlib.pyplot as plt
import seaborn as sns

def full_evaluation(model, loader, criterion, name=""):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # --- Metrics ---
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    sensitivity = np.mean(np.diag(cm) / np.maximum(1, np.sum(cm, axis=1)))
    specificity = np.mean([
        (np.sum(cm) - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])) /
        max(1, (np.sum(cm) - cm[:, i].sum()))
        for i in range(len(cm))
    ])

    try:
        one_hot_labels = label_binarize(all_labels, classes=np.unique(all_labels))
        auc = roc_auc_score(one_hot_labels, all_probs, average='macro', multi_class='ovo')
    except Exception as e:
        print(f" AUC Error ({name}) :", e)
        auc = float('nan')

    # --- Print results ---
    print(f"\n Evaluation on {name} :")
    print(f"Loss        : {total_loss / len(loader):.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1-Score    : {f1:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"MCC         : {mcc:.4f}")
    print(f"AUC         : {auc:.4f}")

    target_names = ["Others", "Sickle Cells", "Normals"]
    print("\n Classification report :\n")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=2))

    # --- Confusion matrix ---
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truth")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# ----

#Bonne
full_evaluation(model, train_loader, criterion, name="Train")
full_evaluation(model, val_loader, criterion, name="Validation")
full_evaluation(model, test_loader, criterion, name="Test")

# ----

#Bonne
full_evaluation(model, train_loader, criterion, name="Train")
full_evaluation(model, val_loader, criterion, name="Validation")
full_evaluation(model, test_loader, criterion, name="Test")

# ----

full_evaluation(model, train_loader, criterion, name="Train")
full_evaluation(model, val_loader, criterion, name="Validation")
full_evaluation(model, test_loader, criterion, name="Test")

# ----

# Évaluation sur le jeu de test
full_evaluation(model, test_loader, criterion, name="Test")

# ----

# --- Sauvegarde (après entraînement)
torch.save(model.state_dict(), "hybrid_model_weights.pth")

# ----

#torch.save(model, "model_complet.pth")
#torch.save(model, "00model_complet.pth")
torch.save(model, "0model_complet.pth")

# ----

# --- Rechargement (plus tard / dans un autre script)
model = HybridModel(Config)
model.load_state_dict(torch.load("hybrid_model_weights.pth", map_location=Config.DEVICE))
model.to(Config.DEVICE)
model.eval()

# ----

# ===== 1) Charger le meilleur modèle =====
ckpt_path = "hybrid_model_weights.pth"   # ajuste si ton meilleur fichier a un autre nom
model = HybridModel(Config)
state = torch.load(ckpt_path, map_location=Config.DEVICE)
model.load_state_dict(state)
model.to(Config.DEVICE)
model.eval()
print(f"✅ Modèle chargé depuis: {ckpt_path}")

# ----

# ===== 2) Évaluer sur le jeu de TEST =====
# Si tu utilises la version qui affiche la matrice (avec seaborn), appelle:
full_evaluation(model, test_loader, criterion, name="Test")

# OU, si tu utilises la version qui renvoie les métriques + cm :
# metrics_test, cm_test = full_evaluation_return(model, test_loader, criterion, name="Test")

# ===== 3) (Optionnel) Sauvegarder métriques + matrice de confusion =====
# Décommente si tu as les helpers save_confusion_matrix / save_metrics
# import os
# os.makedirs("eval_test_outputs", exist_ok=True)
# save_confusion_matrix(cm_test, ["Autres","Falciformes","Normales"],
#                       out_path="eval_test_outputs/confusion_matrix_test.png",
#                       title="Matrice de confusion - Test")
# save_metrics(metrics_test, out_json="eval_test_outputs/metrics_test.json",
#              out_csv="eval_test_outputs/metrics_test.csv")

# ----

import torch.nn.functional as F

y_true = []
y_pred_proba = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred_proba.extend(probs.cpu().numpy())

# ----

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

y_true = np.array(y_true)
y_pred_proba = np.array(y_pred_proba)
n_classes = y_pred_proba.shape[1]

# One-hot encode labels
y_test_bin = label_binarize(y_true, classes=np.arange(n_classes))

fpr, tpr, roc_auc = {}, {}, {}

# ROC per class
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# --- Micro-average ---
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# --- Plot ---
plt.figure(figsize=(8, 6))
colors = cycle(['#1f77b4', '#d62728', '#2ca02c'])
target_names = ['Normal', 'Sickle', 'Other']

# Plot per class
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{target_names[i]} (AUC = {roc_auc[i]:.2f})')

# Plot micro-average (gray, dashed)
plt.plot(fpr["micro"], tpr["micro"],
         color='gray', linestyle='--', linewidth=2.5,
         label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})')

# Diagonal line (random chance)
plt.plot([0, 1], [0, 1], 'k--', lw=1)

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curves per Class with Micro-Average', fontsize=14)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# ----

# 2
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

y_true = np.array(y_true)
y_pred_proba = np.array(y_pred_proba)
n_classes = y_pred_proba.shape[1]

# One-hot encode labels
y_test_bin = label_binarize(y_true, classes=np.arange(n_classes))

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
colors = cycle(['#1f77b4', '#d62728', '#2ca02c'])
target_names = ['Normal', 'Sickle', 'Other']

# Plot ROC curve for each class
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{target_names[i]} (AUC = {roc_auc[i]:.2f})')

# Diagonal line (random chance)
plt.plot([0, 1], [0, 1], 'k--', lw=1)

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curves per Class', fontsize=14)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# ----

# 2
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

y_true = np.array(y_true)
y_pred_proba = np.array(y_pred_proba)
n_classes = y_pred_proba.shape[1]

y_test_bin = label_binarize(y_true, classes=np.arange(n_classes))

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
colors = cycle(['#1f77b4', '#d62728', '#2ca02c'])
target_names = ['Normal', 'Sickle', 'Other']

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{target_names[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate (1-Specificity)', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
#plt.title('Courbes ROC par classe', fontsize=14)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# ----

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# === ROC Curve (multi-classes) ===
def plot_roc_curve(model, loader, name="Test"):
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # binarisation des labels
    classes = np.unique(all_labels)
    y_test = label_binarize(all_labels, classes=classes)
    n_classes = y_test.shape[1]

    plt.figure(figsize=(7,6))

    # courbe ROC pour chaque classe
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2,
                 label=f"{['Autres','Falciformes','Normales'][i]} (AUC = {roc_auc:.3f})")

    # micro-avg
    fpr, tpr, _ = roc_curve(y_test.ravel(), all_probs.ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="black", linestyle="--",
             label=f"Micro-average (AUC = {roc_auc:.3f})")

    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("Taux de faux positifs (FPR)")
    plt.ylabel("Taux de vrais positifs (TPR)")
    plt.title(f"Courbe ROC - {name}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# === Utilisation ===
plot_roc_curve(model, test_loader, name="Test")

# ----

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# === ROC Curve (multi-classes) ===
def plot_roc_curve(model, loader, name="Test"):
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # binarisation des labels
    classes = np.unique(all_labels)
    y_test = label_binarize(all_labels, classes=classes)
    n_classes = y_test.shape[1]

    plt.figure(figsize=(7,6))

    # courbe ROC pour chaque classe
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2,
                 label=f"{['Autres','Falciformes','Normales'][i]} (AUC = {roc_auc:.3f})")

    # micro-avg
    fpr, tpr, _ = roc_curve(y_test.ravel(), all_probs.ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="black", linestyle="--",
             label=f"Micro-average (AUC = {roc_auc:.3f})")

    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("Taux de faux positifs (FPR)")
    plt.ylabel("Taux de vrais positifs (TPR)")
    plt.title(f"Courbe ROC - {name}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# === Utilisation ===
plot_roc_curve(model, test_loader, name="Test")

# ----

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# Adapte si besoin à l’ordre de tes labels
TARGET_NAMES = ["Autres", "Falciformes", "Normales"]

def plot_pr_curve(model, loader, name="Test"):
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Binarisation des labels
    classes = np.unique(all_labels)
    y_true = label_binarize(all_labels, classes=classes)
    n_classes = y_true.shape[1]

    # Noms des classes (si dispo)
    if len(TARGET_NAMES) == n_classes:
        class_names = TARGET_NAMES
    else:
        class_names = [f"Classe {c}" for c in classes]

    # AP par classe
    ap_per_class = []
    plt.figure(figsize=(7,6))

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], all_probs[:, i])
        ap = average_precision_score(y_true[:, i], all_probs[:, i])
        ap_per_class.append(ap)
        plt.plot(recall, precision, lw=2, label=f"{class_names[i]} (AP = {ap:.3f})")

    # Micro-average (global, tous labels confondus)
    precision_micro, recall_micro, _ = precision_recall_curve(y_true.ravel(), all_probs.ravel())
    ap_micro = average_precision_score(y_true, all_probs, average="micro")
    plt.plot(recall_micro, precision_micro, linestyle="--", lw=2,
             label=f"Micro-average (AP = {ap_micro:.3f})")

    # Macro-average (moyenne simple des AP par classe)
    ap_macro = float(np.mean(ap_per_class))

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve - {name} (macro AP = {ap_macro:.3f})")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left")
    plt.show()

# === Utilisation sur le test ===
plot_pr_curve(model, test_loader, name="Test")

# ----

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class GradCAM:
    """
    Grad-CAM classique pour un module convolutionnel cible.
    target_layer: nn.Module (ex: model.cnn.layer4[-1])
    """
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Hooks
        self.fwd_hook = target_layer.register_forward_hook(self._save_activations)
        # full_backward_hook (>=1.8). Si erreur, remplace par register_backward_hook
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        # output: (B, C, H, W)
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        # grad_output[0]: (B, C, H, W)
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    @torch.no_grad()
    def _normalize(self, cam):
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def __call__(self, images, targets=None, device=None):
        """
        images: tensor (B,3,H,W) ou (B,1,H,W)
        targets: liste/tensor de classes cibles (B,) ou None (=classe prédite)
        return: heatmaps (B,H,W) dans [0,1]
        """
        device = device or next(self.model.parameters()).device
        images = images.to(device)

        # FORWARD
        outputs = self.model(images)  # (B, num_classes)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        if targets is None:
            targets = preds
        else:
            targets = torch.as_tensor(targets, device=device)

        # BACKWARD sur le logit de la classe cible (pas la prob)
        self.model.zero_grad(set_to_none=True)
        selected = outputs.gather(1, targets.view(-1,1)).sum()
        selected.backward()

        # activations: (B,C,H,W), gradients: (B,C,H,W)
        A = self.activations           # (B,C,H,W)
        dA = self.gradients            # (B,C,H,W)
        B, C, H, W = A.shape

        # poids: moyenne spatiale des gradients
        weights = dA.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)  # (B,C,1,1)
        cam = (weights * A).sum(dim=1)                            # (B,H,W)
        cam = F.relu(cam)

        # normalisation par image
        heatmaps = torch.stack([self._normalize(cam[i]) for i in range(B)], dim=0)  # (B,H,W)
        return heatmaps.cpu(), preds.cpu(), probs.detach().cpu()

# ----

def show_gradcam(images, heatmaps, true_labels=None, pred_labels=None, class_names=None, cols=4):
    """
    images: tensor (B,3,H,W) ou (B,1,H,W) en 0-1 (si ce n'est pas le cas, clamp/normalize avant)
    heatmaps: tensor (B,H,W) en [0,1]
    """
    imgs = images.detach().cpu()
    if imgs.shape[1] == 1:  # grayscale -> répéter pour affichage couleur
        imgs = imgs.repeat(1,3,1,1)

    B = imgs.shape[0]
    rows = int(np.ceil(B / cols))
    plt.figure(figsize=(4*cols, 4*rows))

    for i in range(B):
        plt.subplot(rows, cols, i+1)
        img = np.transpose(imgs[i].numpy(), (1,2,0))
        hm = heatmaps[i].numpy()

        # overlay: image + heatmap (rouge) via alpha
        plt.imshow(img, interpolation='bilinear')
        plt.imshow(hm, cmap='jet', alpha=0.4, interpolation='bilinear')  # alpha fixe
        title = ""
        if pred_labels is not None and class_names is not None:
            title += f"Pred: {class_names[int(pred_labels[i])]}"
        if true_labels is not None and class_names is not None:
            title += f"\nTrue: {class_names[int(true_labels[i])]}"
        plt.title(title.strip(), fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# ----

def show_gradcam_with_original(images, heatmaps, true_labels=None, pred_labels=None,
                               class_names=None, scores=None, score_name="p", cols=4):
    """
    Affiche l'image originale + la superposition Grad-CAM côte à côte.
    images: tensor (B,3,H,W) ou (B,1,H,W)
    heatmaps: tensor (B,H,W)
    """
    imgs = images.detach().cpu()
    if imgs.shape[1] == 1:  # grayscale → RGB
        imgs = imgs.repeat(1, 3, 1, 1)

    B = imgs.shape[0]
    rows = int(np.ceil(B / cols))
    plt.figure(figsize=(8*cols, 4*rows))  # 2x plus large car 2 colonnes par image

    for i in range(B):
        img = np.transpose(imgs[i].numpy(), (1, 2, 0))
        hm = heatmaps[i].numpy()

        # --- Image originale ---
        plt.subplot(rows, cols*2, 2*i+1)
        plt.imshow(img, interpolation='bilinear')
        title_parts = []
        if true_labels is not None and class_names is not None:
            title_parts.append(f"True: {class_names[int(true_labels[i])]}")
        plt.title(" | ".join(title_parts), fontsize=9)
        plt.axis("off")

        # --- Image avec heatmap ---
        plt.subplot(rows, cols*2, 2*i+2)
        plt.imshow(img, interpolation='bilinear')
        plt.imshow(hm, cmap='jet', alpha=0.4, interpolation='bilinear')

        title_parts = []
        if pred_labels is not None and class_names is not None:
            title_parts.append(f"Pred: {class_names[int(pred_labels[i])]}")
        if scores is not None:
            title_parts.append(f"{score_name}={float(scores[i]):.3f}")
        plt.title(" | ".join(title_parts), fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# ----

# --- Paramètres ---
CLASS_NAMES = ["Autres", "Falciformes", "Normales"]  # adapte à ton mapping
device = Config.DEVICE

# --- Choix de la couche cible ---
# Exemple si ton HybridModel a un backbone ResNet sous model.cnn
target_layer = model.cnn.layer4[-1]   # <-- adapte selon ta classe

# --- Instancier Grad-CAM ---
cam = GradCAM(model.to(device).eval(), target_layer=target_layer)

# --- Récupérer un mini-batch du test pour visualisation ---
batch_images, batch_labels = next(iter(test_loader))
batch_images = batch_images.to(device)
batch_labels = batch_labels.to(device)

# (Option) choisir les classes cibles = vraies étiquettes pour voir l'explication "correcte"
# sinon, mets targets=None pour expliquer la classe prédite
heatmaps, preds, probs = cam(batch_images, targets=batch_labels, device=device)

# --- Afficher ---
show_gradcam(batch_images, heatmaps, true_labels=batch_labels.cpu(),
             pred_labels=preds, class_names=CLASS_NAMES, cols=4)

# --- Nettoyer les hooks si tu n'en as plus besoin ---
cam.remove_hooks()

# ----

# --- Paramètres ---
CLASS_NAMES = ["Autres", "Falciformes", "Normales"]  # adapte à ton mapping
device = Config.DEVICE

# --- Choix de la couche cible ---
# Exemple si ton HybridModel a un backbone ResNet sous model.cnn
target_layer = model.cnn.layer4[-1]   # <-- adapte selon ta classe

# --- Instancier Grad-CAM ---
cam = GradCAM(model.to(device).eval(), target_layer=target_layer)

# --- Récupérer un mini-batch du test pour visualisation ---
batch_images, batch_labels = next(iter(test_loader))
batch_images = batch_images.to(device)
batch_labels = batch_labels.to(device)

# (Option) choisir les classes cibles = vraies étiquettes pour voir l'explication "correcte"
# sinon, mets targets=None pour expliquer la classe prédite
heatmaps, preds, probs = cam(batch_images, targets=batch_labels, device=device)

# --- Afficher ---
show_gradcam(batch_images, heatmaps, true_labels=batch_labels.cpu(),
             pred_labels=preds, class_names=CLASS_NAMES, cols=4)

# --- Nettoyer les hooks si tu n'en as plus besoin ---
cam.remove_hooks()

# ----

def show_gradcam(images, heatmaps, true_labels=None, pred_labels=None,
                 class_names=None, cols=4, scores=None, score_name="p"):
    """
    images: tensor (B,3,H,W) ou (B,1,H,W)
    heatmaps: tensor (B,H,W)
    scores: array-like (B,) -> affiché comme score_name=0.987 (ex. proba)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    imgs = images.detach().cpu()
    if imgs.shape[1] == 1:  # si grayscale, répéter pour affichage couleur
        imgs = imgs.repeat(1,3,1,1)

    B = imgs.shape[0]
    rows = int(np.ceil(B / cols))
    plt.figure(figsize=(4*cols, 4*rows))

    for i in range(B):
        plt.subplot(rows, cols, i+1)
        img = np.transpose(imgs[i].numpy(), (1,2,0))
        hm = heatmaps[i].numpy()

        plt.imshow(img, interpolation='bilinear')
        plt.imshow(hm, cmap='jet', alpha=0.4, interpolation='bilinear')

        title_parts = []
        if pred_labels is not None and class_names is not None:
            title_parts.append(f"Pred: {class_names[int(pred_labels[i])]}")
        if true_labels is not None and class_names is not None:
            title_parts.append(f"True: {class_names[int(true_labels[i])]}")
        if scores is not None:
            title_parts.append(f"{score_name}={float(scores[i]):.3f}")

        plt.title(" | ".join(title_parts), fontsize=9)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# ----

def show_gradcam_with_original(images, heatmaps, true_labels=None, pred_labels=None,
                               class_names=None, scores=None, score_name="p", cols=4):
    """
    Affiche l'image originale + la superposition Grad-CAM côte à côte.
    images: tensor (B,3,H,W) ou (B,1,H,W)
    heatmaps: tensor (B,H,W)
    """
    imgs = images.detach().cpu()
    if imgs.shape[1] == 1:  # grayscale → RGB
        imgs = imgs.repeat(1, 3, 1, 1)

    B = imgs.shape[0]
    rows = int(np.ceil(B / cols))
    plt.figure(figsize=(8*cols, 4*rows))  # 2x plus large car 2 colonnes par image

    for i in range(B):
        img = np.transpose(imgs[i].numpy(), (1, 2, 0))
        hm = heatmaps[i].numpy()

        # --- Image originale ---
        plt.subplot(rows, cols*2, 2*i+1)
        plt.imshow(img, interpolation='bilinear')
        title_parts = []
        if true_labels is not None and class_names is not None:
            title_parts.append(f"True: {class_names[int(true_labels[i])]}")
        plt.title(" | ".join(title_parts), fontsize=9)
        plt.axis("off")

        # --- Image avec heatmap ---
        plt.subplot(rows, cols*2, 2*i+2)
        plt.imshow(img, interpolation='bilinear')
        plt.imshow(hm, cmap='jet', alpha=0.4, interpolation='bilinear')

        title_parts = []
        if pred_labels is not None and class_names is not None:
            title_parts.append(f"Pred: {class_names[int(pred_labels[i])]}")
        if scores is not None:
            title_parts.append(f"{score_name}={float(scores[i]):.3f}")
        plt.title(" | ".join(title_parts), fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# ----

show_gradcam_with_original(batch_images, heatmaps,
                           true_labels=batch_labels.cpu(),
                           pred_labels=preds,
                           class_names=CLASS_NAMES,
                           scores=conf_pred,
                           score_name="p_pred")

# ----

conf_pred = get_confidence_vector(probs, preds=preds)

show_gradcam(batch_images, heatmaps,
             true_labels=batch_labels, pred_labels=preds,
             class_names=CLASS_NAMES, cols=4,
             scores=conf_pred, score_name="p_pred")

# ----

import numpy as np
import torch
import matplotlib.pyplot as plt

# Si tes images ont été normalisées comme ImageNet :
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def denormalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    img_tensor: (C,H,W) torch.Tensor
    Retourne un numpy (H,W,C) clampé dans [0,1] pour affichage.
    """
    x = img_tensor.detach().cpu().float()
    if x.shape[0] == 1:  # grayscale -> afficher en pseudo-RGB
        x = x.repeat(3,1,1)
    m = torch.tensor(mean).view(-1,1,1)
    s = torch.tensor(std).view(-1,1,1)
    x = (x * s) + m
    x = x.clamp(0,1)
    return np.transpose(x.numpy(), (1,2,0))

def pick_indices_per_class(labels, class_ids, k_per_class=3):
    """
    labels: tensor/ndarray (B,)
    class_ids: liste des ids de classes à couvrir (ex: [0,1,2])
    Retourne jusqu'à k_per_class indices par classe.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    idxs = []
    for c in class_ids:
        pool = np.where(labels == c)[0]
        if len(pool) > 0:
            k = min(k_per_class, len(pool))
            chosen = np.random.choice(pool, size=k, replace=False)
            idxs.extend(chosen.tolist())
    return idxs

# ----

def show_gradcam_side_by_side(images, heatmaps,
                              true_labels=None, pred_labels=None,
                              class_names=None, indices=None, k_per_class=3,
                              choose_by="pred"):
    """
    images    : (B,C,H,W) torch.Tensor
    heatmaps  : (B,H,W)   torch.Tensor (normalisés 0..1 par ta classe GradCAM)
    choose_by : "pred" -> sélection par classe prédite, "true" -> classe vraie
    indices   : liste d'indices pour forcer l'affichage (sinon on choisit par classe)
    """
    imgs = images.detach().cpu()
    hmaps = heatmaps.detach().cpu().numpy() if isinstance(heatmaps, torch.Tensor) else heatmaps

    # Déterminer quelles classes couvrir
    if class_names is None:
        raise ValueError("Fournis class_names dans le même ordre que tes labels.")
    class_ids = np.arange(len(class_names))

    # Choisir les indices à afficher
    if indices is None:
        base = pred_labels if choose_by == "pred" else true_labels
        if base is None:
            raise ValueError("choose_by='pred' nécessite pred_labels, 'true' nécessite true_labels.")
        indices = pick_indices_per_class(base, class_ids, k_per_class=k_per_class)

    n = len(indices)
    cols = 2  # Original | Grad-CAM
    rows = int(np.ceil(n))
    plt.figure(figsize=(6*cols, 3*rows))

    for j, idx in enumerate(indices):
        img_np = denormalize(imgs[idx])  # on n'altère PAS le heatmap
        hm = hmaps[idx]

        # Titres
        title_lines = []
        if pred_labels is not None:
            title_lines.append(f"Pred: {class_names[int(pred_labels[idx])]}")
        if true_labels is not None:
            title_lines.append(f"True: {class_names[int(true_labels[idx])]}")
        title = " | ".join(title_lines)

        # Colonne 1 : Original
        plt.subplot(rows, cols, 2*j + 1)
        plt.imshow(img_np, interpolation='bilinear')
        plt.title(f"Original\n{title}", fontsize=9)
        plt.axis('off')

        # Colonne 2 : Overlay Grad-CAM (heatmap inchangé)
        plt.subplot(rows, cols, 2*j + 2)
        plt.imshow(img_np, interpolation='bilinear')
        plt.imshow(hm, cmap='jet', alpha=0.40, interpolation='bilinear')
        plt.title(f"Grad-CAM\n{title}", fontsize=9)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# ----

# On récupère un batch du test
batch_images, batch_labels = next(iter(test_loader))
batch_images = batch_images.to(device)
batch_labels = batch_labels.to(device)

# Grad-CAM (sur la classe prédite pour chaque image)
heatmaps, preds, probs = cam(batch_images, targets=None, device=device)

# Afficher k images par classe prédite (ex: 3) : toutes les classes seront représentées si présentes dans le batch
show_gradcam_side_by_side(batch_images, heatmaps,
                          true_labels=batch_labels, pred_labels=preds,
                          class_names=CLASS_NAMES,
                          k_per_class=3, choose_by="pred")

# (Option) Sélection par classe VRAIE au lieu de prédite :
# show_gradcam_side_by_side(batch_images, heatmaps,
#                           true_labels=batch_labels, pred_labels=preds,
#                           class_names=CLASS_NAMES,
#                           k_per_class=3, choose_by="true")

# ----

def get_confidence_vector(probs, preds=None, targets=None):
    """
    probs : torch.Tensor (B, num_classes)
    preds : Tensor (B,) classes prédites
    targets : Tensor (B,) classes vraies
    Retour : np.array (B,) des probabilités correspondantes
    """
    probs_np = probs.detach().cpu().numpy()
    if preds is not None:
        idx = preds.detach().cpu().numpy()
    elif targets is not None:
        idx = targets.detach().cpu().numpy()
    else:
        raise ValueError("Fournir preds OU targets.")
    return probs_np[np.arange(len(probs_np)), idx]

# ----

def show_gradcam_side_by_side(images, heatmaps,
                              true_labels=None, pred_labels=None,
                              class_names=None, indices=None, k_per_class=3,
                              choose_by="pred",
                              scores=None, score_name="p_pred",
                              extra_scores=None, extra_score_name="p_true"):
    """
    images       : (B,C,H,W)
    heatmaps     : (B,H,W)
    scores       : array-like (B,) -> affiché comme score_name=0.987 (ex: proba prédite)
    extra_scores : array-like (B,) -> second score optionnel (ex: proba vraie)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    imgs = images.detach().cpu()
    hmaps = heatmaps.detach().cpu().numpy() if isinstance(heatmaps, torch.Tensor) else heatmaps

    if class_names is None:
        raise ValueError("Fournis class_names dans le même ordre que tes labels.")
    class_ids = np.arange(len(class_names))

    # Choix indices
    if indices is None:
        base = pred_labels if choose_by == "pred" else true_labels
        if base is None:
            raise ValueError("choose_by='pred' nécessite pred_labels, 'true' nécessite true_labels.")
        indices = pick_indices_per_class(base, class_ids, k_per_class=k_per_class)

    n = len(indices)
    cols = 2
    rows = int(np.ceil(n))
    plt.figure(figsize=(6*cols, 3*rows))

    for j, idx in enumerate(indices):
        img_np = denormalize(imgs[idx])
        hm = hmaps[idx]

        # Titre
        title_parts = []
        if pred_labels is not None:
            title_parts.append(f"Pred: {class_names[int(pred_labels[idx])]}")
        if true_labels is not None:
            title_parts.append(f"True: {class_names[int(true_labels[idx])]}")
        if scores is not None:
            title_parts.append(f"{score_name}={float(scores[idx]):.3f}")
        if extra_scores is not None:
            title_parts.append(f"{extra_score_name}={float(extra_scores[idx]):.3f}")
        title = " | ".join(title_parts)

        # Original
        plt.subplot(rows, cols, 2*j + 1)
        plt.imshow(img_np, interpolation='bilinear')
        plt.title(f"Original\n{title}", fontsize=9)
        plt.axis('off')

        # Grad-CAM overlay (heatmap inchangé)
        plt.subplot(rows, cols, 2*j + 2)
        plt.imshow(img_np, interpolation='bilinear')
        plt.imshow(hm, cmap='jet', alpha=0.40, interpolation='bilinear')
        plt.title(f"Grad-CAM\n{title}", fontsize=9)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# ----

# Batch du test
batch_images, batch_labels = next(iter(test_loader))
batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

# Grad-CAM (cible = classe prédite)
heatmaps, preds, probs = cam(batch_images, targets=None, device=device)

# Scores
conf_pred = get_confidence_vector(probs, preds=preds)           # p_pred
conf_true = get_confidence_vector(probs, targets=batch_labels)  # p_true

# Afficher k images par classe prédite (ex: 3) avec les deux scores
show_gradcam_side_by_side(batch_images, heatmaps,
                          true_labels=batch_labels, pred_labels=preds,
                          class_names=CLASS_NAMES,
                          k_per_class=3, choose_by="pred",
                          scores=conf_pred, score_name="p_pred",
                          extra_scores=conf_true, extra_score_name="p_true")

# ----

import torch
import numpy as np
import matplotlib.pyplot as plt

def resolve_class_id_by_name(dataset, desired_names, fallback_idx):
    """
    Essaie de retrouver l'ID d'une classe via dataset.class_to_idx
    en testant plusieurs variantes de nom. Sinon renvoie fallback_idx.
    """
    if hasattr(dataset, "class_to_idx"):
        c2i = dataset.class_to_idx
        for name in desired_names:
            if name in c2i:
                return c2i[name]
    return fallback_idx

def gather_k_examples_per_class(loader, class_id, k=10, device=None, ensure_k=True):
    """
    Retourne EXACTEMENT k images/labels pour la classe `class_id`.
    - Si < k dispo, répète des exemples jusqu'à k (ensure_k=True).
    - Si > k dispo, tronque à k.
    """
    imgs_list, labs_list = [], []
    for images, labels in loader:
        mask = (labels == class_id)
        if mask.any():
            imgs_list.append(images[mask])
            labs_list.append(labels[mask])
            # stop si on a déjà >= k en stock
            if sum(t.size(0) for t in labs_list) >= k:
                break

    if not imgs_list:
        raise RuntimeError(f"Aucune image trouvée pour la classe id={class_id}.")

    imgs = torch.cat(imgs_list, dim=0)
    labs = torch.cat(labs_list, dim=0)

    n = imgs.size(0)
    if n >= k:
        imgs = imgs[:k]
        labs = labs[:k]
    else:
        if ensure_k:
            # Répéter pour atteindre k
            reps = (k + n - 1) // n  # plafond(k/n)
            imgs = imgs.repeat((reps, 1, 1, 1))[:k]
            labs = labs.repeat(reps)[:k]
            print(f"[Info] Classe {class_id}: {n} trouvées < {k}, répétition pour atteindre {k}.")
        else:
            # On garde n < k, l'affichage gérera les colonnes vides
            print(f"[Info] Classe {class_id}: seulement {n}/{k} images disponibles.")

    if device is not None:
        imgs = imgs.to(device)
        labs = labs.to(device)
    return imgs, labs

def plot_gradcam_10_per_class(
    cam,
    loader,
    class_names=("Autres","Falciformes","Normales"),
    k_per_class=10,
    save_path="gradcam_10_per_class.png",
    overlay_alpha=0.45,
    denorm_mean=None,
    denorm_std=None,
    show_pred_scores=True,
    dpi=300
):
    """
    Figure: 6 rangées x k_per_class colonnes (par classe: Original + Grad-CAM).
    Garantit exactement k_per_class par classe (répétition si nécessaire).
    """
    ds = loader.dataset if hasattr(loader, "dataset") else None

    autres_id = resolve_class_id_by_name(ds, ["Autres","Other","Others"], fallback_idx=class_names.index("Autres"))
    falci_id  = resolve_class_id_by_name(ds, ["Falciformes","Sickle","Sickled","Sickle Cells"], fallback_idx=class_names.index("Falciformes"))
    normal_id = resolve_class_id_by_name(ds, ["Normales","Normal","Normals"], fallback_idx=class_names.index("Normales"))

    # Récupération + Grad-CAM pour chaque classe (EXACTEMENT k_per_class)
    per_class = []
    for cid in (autres_id, falci_id, normal_id):
        imgs, labs = gather_k_examples_per_class(
            loader, class_id=cid, k=k_per_class,
            device=next(cam.model.parameters()).device, ensure_k=True
        )
        heatmaps, preds, probs = cam(imgs, targets=labs)  # explique la VÉRITÉ
        # Sécuriser la taille à k_per_class (au cas où)
        heatmaps = heatmaps[:k_per_class]
        preds    = preds[:k_per_class]
        probs    = probs[:k_per_class]

        confs = get_confidence_vector(probs, preds=preds)

        disp = imgs
        if disp.shape[1] == 1:
            disp = disp.repeat(1, 3, 1, 1)
        disp = denormalize(disp, mean=denorm_mean, std=denorm_std)
        disp = disp[:k_per_class]

        per_class.append((disp.cpu().numpy(), heatmaps.cpu().numpy(), preds.cpu().numpy(), confs))

    # Construction de la figure
    rows, cols = 6, k_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(2.1*cols, 2.1*rows))
    if rows == 1:
        axes = np.array([axes])

    # Noms propres (affichage) dans l'ordre Autres, Falciformes, Normales
    display_names = [
        class_names[class_names.index("Autres")],
        class_names[class_names.index("Falciformes")],
        class_names[class_names.index("Normales")],
    ]

    for class_idx, (disp, heat, preds, confs) in enumerate(per_class):
        top_row = 2*class_idx       # originales
        bot_row = 2*class_idx + 1   # overlays

        # Boucle sur EXACTEMENT k_per_class colonnes
        for j in range(k_per_class):
            # sécurité : si jamais un décalage survient, on coupe
            if j >= disp.shape[0] or j >= heat.shape[0]:
                # masquer cellule
                axes[top_row, j].axis("off")
                axes[bot_row, j].axis("off")
                continue

            img = np.transpose(disp[j], (1, 2, 0))
            hm  = heat[j]

            # Original
            ax = axes[top_row, j]
            ax.imshow(img, interpolation="bilinear")
            if j == 0:
                ax.set_ylabel(f"{display_names[class_idx]}\nOriginal", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

            # Grad-CAM
            ax = axes[bot_row, j]
            ax.imshow(img, interpolation="bilinear")
            ax.imshow(hm, cmap="jet", alpha=overlay_alpha, interpolation="bilinear")
            if j == 0:
                ax.set_ylabel("Grad-CAM", fontsize=10)
            if show_pred_scores:
                ax.set_title(f"Pred: {class_names[int(preds[j])]} (p={confs[j]:.2f})", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Grad-CAM — 10 images par classe (Original & Overlay)", fontsize=12)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    print(f"[OK] Figure enregistrée : {save_path}")

# Optionnel : si tes images sont normalisées ImageNet
# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD  = (0.229, 0.224, 0.225)

plot_gradcam_10_per_class(
    cam, test_loader, class_names=CLASS_NAMES, k_per_class=10,
    save_path="gradcam_10_per_class.png",
    overlay_alpha=0.45,
    denorm_mean=None, denorm_std=None,   # mets IMAGENET_MEAN/STD si besoin
    show_pred_scores=True, dpi=300
)

# ----

import torch
import numpy as np
import matplotlib.pyplot as plt

def resolve_class_id_by_name(dataset, desired_names, fallback_idx):
    """
    Essaie de retrouver l'ID d'une classe via dataset.class_to_idx
    en testant plusieurs variantes de nom. Sinon renvoie fallback_idx.
    """
    if hasattr(dataset, "class_to_idx"):
        c2i = dataset.class_to_idx
        for name in desired_names:
            if name in c2i:
                return c2i[name]
    return fallback_idx

def gather_k_examples_per_class(loader, class_id, k=10, device=None, ensure_k=True):
    """
    Retourne EXACTEMENT k images/labels pour la classe `class_id`.
    - Si < k dispo, répète des exemples jusqu'à k (ensure_k=True).
    - Si > k dispo, tronque à k.
    """
    imgs_list, labs_list = [], []
    for images, labels in loader:
        mask = (labels == class_id)
        if mask.any():
            imgs_list.append(images[mask])
            labs_list.append(labels[mask])
            # stop si on a déjà >= k en stock
            if sum(t.size(0) for t in labs_list) >= k:
                break

    if not imgs_list:
        raise RuntimeError(f"Aucune image trouvée pour la classe id={class_id}.")

    imgs = torch.cat(imgs_list, dim=0)
    labs = torch.cat(labs_list, dim=0)

    n = imgs.size(0)
    if n >= k:
        imgs = imgs[:k]
        labs = labs[:k]
    else:
        if ensure_k:
            # Répéter pour atteindre k
            reps = (k + n - 1) // n  # plafond(k/n)
            imgs = imgs.repeat((reps, 1, 1, 1))[:k]
            labs = labs.repeat(reps)[:k]
            print(f"[Info] Classe {class_id}: {n} trouvées < {k}, répétition pour atteindre {k}.")
        else:
            # On garde n < k, l'affichage gérera les colonnes vides
            print(f"[Info] Classe {class_id}: seulement {n}/{k} images disponibles.")

    if device is not None:
        imgs = imgs.to(device)
        labs = labs.to(device)
    return imgs, labs

def plot_gradcam_10_per_class(
    cam,
    loader,
    class_names=("Others", "Sickle Cells", "Normals"),
    k_per_class=10,
    save_path="gradcam_10_per_class.png",
    overlay_alpha=0.45,
    denorm_mean=None,
    denorm_std=None,
    show_pred_scores=True,
    dpi=300,
    label_inside=True  # ← True = text inside the image, False = as axis title
):
    """
    Build a 6-row x k-column figure showing 10 images per class:
      - For each class: top row = original images, bottom row = Grad-CAM overlays.
    Ensures exactly k_per_class per class (repeats samples if needed).
    Saves a high-resolution PNG for papers.

    Args:
        cam: GradCAM instance (callable) -> returns (heatmaps, preds, probs).
        loader: DataLoader (ideally your test loader).
        class_names: tuple/list of class display names (len = num_classes).
        k_per_class: number of examples to display per class.
        save_path: output PNG path.
        overlay_alpha: transparency for the heatmap overlay.
        denorm_mean, denorm_std: use if images were normalized (e.g., ImageNet stats).
        show_pred_scores: whether to show predicted label + prob.
        dpi: output figure DPI.
        label_inside: True to draw text inside the image (never cut), False to use axis title.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Utilities -----------------------------------------------------------
    def fallback_from_candidates(names, candidates, default_idx=0):
        """Pick the first candidate present in `names`; otherwise return default_idx."""
        for nm in candidates:
            if nm in names:
                return names.index(nm)
        return default_idx

    def resolve_class_id_by_name(dataset, desired_names, fallback_candidates, default_idx=0):
        """
        Try to resolve a class id via dataset.class_to_idx using any of `desired_names`.
        If not found, fall back to the first match within class_names from `fallback_candidates`.
        """
        if hasattr(dataset, "class_to_idx"):
            c2i = dataset.class_to_idx
            for nm in desired_names:
                if nm in c2i:
                    return c2i[nm]
        # Fallback using provided display class_names
        return fallback_from_candidates(class_names, fallback_candidates, default_idx=default_idx)

    def gather_k_examples_per_class(loader, class_id, k=10, device=None, ensure_k=True):
        """
        Return EXACTLY k images/labels for a given class_id.
        - If fewer than k are available and ensure_k=True, repeat samples to reach k.
        - If more are available, truncate to k.
        """
        imgs_list, labs_list = [], []
        for images, labels in loader:
            mask = (labels == class_id)
            if mask.any():
                imgs_list.append(images[mask])
                labs_list.append(labels[mask])
                if sum(t.size(0) for t in labs_list) >= k:
                    break
        if not imgs_list:
            raise RuntimeError(f"No image found for class id={class_id}.")
        imgs = torch.cat(imgs_list, dim=0)
        labs = torch.cat(labs_list, dim=0)

        n = imgs.size(0)
        if n >= k:
            imgs = imgs[:k]; labs = labs[:k]
        else:
            if ensure_k:
                reps = (k + n - 1) // n  # ceil(k/n)
                imgs = imgs.repeat((reps, 1, 1, 1))[:k]
                labs = labs.repeat(reps)[:k]
                print(f"[Info] Class {class_id}: found {n} < {k}, repeating to reach {k}.")
            # else: keep <k and leave blanks (not used here)

        if device is not None:
            imgs = imgs.to(device); labs = labs.to(device)
        return imgs, labs
    # ------------------------------------------------------------------------

    ds = loader.dataset if hasattr(loader, "dataset") else None

    # Build a reliable index->name mapping (prefer dataset ordering if present)
    if hasattr(ds, "class_to_idx"):
        idx_to_name = {v: k for k, v in ds.class_to_idx.items()}
    else:
        idx_to_name = {i: n for i, n in enumerate(class_names)}

    # Resolve class IDs robustly (accept English & French synonyms)
    others_id = resolve_class_id_by_name(
        ds,
        desired_names=["Others", "Other", "Autres"],
        fallback_candidates=["Others", "Other", "Autres"],
        default_idx=0
    )
    sickle_id = resolve_class_id_by_name(
        ds,
        desired_names=["Sickle Cells", "Sickled", "Sickle", "Falciformes"],
        fallback_candidates=["Sickle Cells", "Sickled", "Sickle", "Falciformes"],
        default_idx=1
    )
    normal_id = resolve_class_id_by_name(
        ds,
        desired_names=["Normals", "Normal", "Normales"],
        fallback_candidates=["Normals", "Normal", "Normales"],
        default_idx=2
    )

    # Collect and compute Grad-CAM for each class (exactly k_per_class)
    per_class = []
    for cid in (others_id, sickle_id, normal_id):
        imgs, labs = gather_k_examples_per_class(
            loader, class_id=cid, k=k_per_class,
            device=next(cam.model.parameters()).device, ensure_k=True
        )
        heatmaps, preds, probs = cam(imgs, targets=labs)  # explain the GROUND TRUTH (targets=labs)
        heatmaps = heatmaps[:k_per_class]
        preds    = preds[:k_per_class]
        probs    = probs[:k_per_class]
        confs = get_confidence_vector(probs, preds=preds)

        disp = imgs
        if disp.shape[1] == 1:
            disp = disp.repeat(1, 3, 1, 1)
        disp = denormalize(disp, mean=denorm_mean, std=denorm_std)[:k_per_class]

        per_class.append((disp.cpu().numpy(), heatmaps.cpu().numpy(), preds.cpu().numpy(), confs))

    # Build figure: 6 rows (Original/Grad-CAM per class) x k columns
    rows, cols = 6, k_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(2.2*cols, 2.4*rows), constrained_layout=True)
    axes = np.atleast_2d(axes)

    # Display names (in the order: Others, Sickle Cells, Normals)
    display_names = [
        fallback_from_candidates(class_names, ["Others", "Other", "Autres"], default_idx=0)  or 0,
        fallback_from_candidates(class_names, ["Sickle Cells", "Sickled", "Sickle", "Falciformes"], default_idx=1) or 1,
        fallback_from_candidates(class_names, ["Normals", "Normal", "Normales"], default_idx=2)  or 2,
    ]
    # convert indices → names
    display_names = [class_names[i] if 0 <= i < len(class_names) else f"class_{i}" for i in display_names]

    for class_idx, (disp, heat, preds, confs) in enumerate(per_class):
        top_row = 2 * class_idx
        bot_row = 2 * class_idx + 1

        for j in range(k_per_class):
            if j >= disp.shape[0] or j >= heat.shape[0]:
                axes[top_row, j].axis("off")
                axes[bot_row, j].axis("off")
                continue

            img = np.transpose(disp[j], (1, 2, 0))
            hm  = heat[j]

            # Original
            ax = axes[top_row, j]
            ax.imshow(img, interpolation="bilinear")
            if j == 0:
                ax.set_ylabel(f"{display_names[class_idx]}\nOriginal", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

            # Grad-CAM
            ax = axes[bot_row, j]
            ax.imshow(img, interpolation="bilinear")
            ax.imshow(hm, cmap="jet", alpha=overlay_alpha, interpolation="bilinear")
            if j == 0:
                ax.set_ylabel("Grad-CAM", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

            if show_pred_scores:
                pred_idx = int(preds[j])
                # robust predicted class name
                pred_name = idx_to_name.get(
                    pred_idx,
                    class_names[pred_idx] if 0 <= pred_idx < len(class_names) else f"class_{pred_idx}"
                )
                conf_val = float(confs[j]) if np.isfinite(confs[j]) else float("nan")

                if label_inside:
                    # text inside the image (never cut off)
                    ax.text(
                        0.02, 0.98,
                        f"Pred: {pred_name}\n(p={conf_val:.2f})",
                        transform=ax.transAxes,
                        va="top", ha="left",
                        fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none")
                    )
                else:
                    # axis title (can be compressed in tight layouts)
                    ax.set_title(f"Pred: {pred_name} (p={conf_val:.2f})", fontsize=8, pad=2)

    fig.suptitle("Grad-CAM — 10 images per class (Original & Overlay)", fontsize=12)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    print(f"[OK] Saved figure: {save_path}")

# Optionnel : si tes images sont normalisées ImageNet
# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD  = (0.229, 0.224, 0.225)

plot_gradcam_10_per_class(
    cam, test_loader, class_names=CLASS_NAMES, k_per_class=10,
    save_path="gradcam_10_per_class.png",
    overlay_alpha=0.45,
    denorm_mean=None, denorm_std=None,   # mets IMAGENET_MEAN/STD si besoin
    show_pred_scores=True, dpi=300
)

# ----

# ===============================
# Panel unique 3 classes (1 figure)
# ===============================

import torch
import numpy as np
import matplotlib.pyplot as plt

# Optionnel : si tes images sont normalisées ImageNet, décommente :
# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD  = (0.229, 0.224, 0.225)

def resolve_class_id_by_name(dataset, desired_names, fallback_idx):
    """
    desired_names: liste de noms candidats pour une même classe.
    Renvoie l'ID trouvé via dataset.class_to_idx, sinon fallback_idx.
    """
    if hasattr(dataset, "class_to_idx"):
        c2i = dataset.class_to_idx
        for name in desired_names:
            if name in c2i:
                return c2i[name]
    return fallback_idx

def gather_one_example_per_class(loader, class_ids, device=None):
    """
    Retourne (imgs, labels) avec exactement 1 image par classe demandée (dans l'ordre).
    Lève une erreur si aucune image trouvée pour une classe.
    """
    found = {cid: None for cid in class_ids}
    for images, labels in loader:
        for cid in class_ids:
            if found[cid] is None:
                mask = (labels == cid)
                if mask.any():
                    idx = torch.nonzero(mask, as_tuple=False)[0].item()
                    img = images[idx:idx+1]      # (1,C,H,W)
                    lab = labels[idx:idx+1]      # (1,)
                    found[cid] = (img, lab)
        if all(v is not None for v in found.values()):
            break

    if not all(v is not None for v in found.values()):
        missing = [cid for cid, v in found.items() if v is None]
        raise RuntimeError(f"Aucune image trouvée pour les classes: {missing}")

    imgs = torch.cat([found[cid][0] for cid in class_ids], dim=0)
    labs = torch.cat([found[cid][1] for cid in class_ids], dim=0)
    if device is not None:
        imgs = imgs.to(device)
        labs = labs.to(device)
    return imgs, labs

def plot_gradcam_panel_three_classes(
    cam,
    loader,
    class_names=("Autres","Falciformes","Normales"),
    save_path="gradcam_3classes_panel.png",
    overlay_alpha=0.45,
    denorm_mean=None,
    denorm_std=None,
    show_pred_scores=True,
    dpi=300
):
    """
    Crée une figure unique : 2 rangées x 3 colonnes
      - Rangée 1 : images originales (Autres, Falciformes, Normales)
      - Rangée 2 : mêmes images avec Grad-CAM superposé
    Sauvegarde en PNG haute résolution (dpi).
    """
    ds = loader.dataset if hasattr(loader, "dataset") else None

    # Résolution robuste des IDs (synonymes possibles)
    autres_id = resolve_class_id_by_name(ds, ["Autres","Other","Others"], fallback_idx=class_names.index("Autres"))
    falci_id  = resolve_class_id_by_name(ds, ["Falciformes","Sickle","Sickled","Sickle Cells"], fallback_idx=class_names.index("Falciformes"))
    normal_id = resolve_class_id_by_name(ds, ["Normales","Normal","Normals"], fallback_idx=class_names.index("Normales"))
    class_ids = [autres_id, falci_id, normal_id]

    # Récupérer 1 exemple par classe (dans l'ordre)
    imgs, labs = gather_one_example_per_class(loader, class_ids, device=next(cam.model.parameters()).device)

    # Calcul Grad-CAM (targets = vraies étiquettes pour illustrer la classe correcte)
    heatmaps, preds, probs = cam(imgs, targets=labs)
    # Confiances de la classe prédite
    conf_pred = get_confidence_vector(probs, preds=preds)

    # Préparer images pour affichage
    disp = imgs
    if disp.shape[1] == 1:
        disp = disp.repeat(1,3,1,1)
    disp = denormalize(disp, mean=denorm_mean, std=denorm_std)  # ou laisse None si déjà dans [0,1]

    # Figure : 2x3
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for j in range(3):
        img = np.transpose(disp[j].cpu().numpy(), (1,2,0))
        hm  = heatmaps[j].numpy()

        # Rangée 1 : Original
        ax = axes[0, j]
        ax.imshow(img, interpolation="bilinear")
        ax.set_title(f"{class_names[class_ids[j]]}", fontsize=11)
        ax.axis("off")

        # Rangée 2 : Overlay Grad-CAM
        ax = axes[1, j]
        ax.imshow(img, interpolation="bilinear")
        ax.imshow(hm, cmap="jet", alpha=overlay_alpha, interpolation="bilinear")
        if show_pred_scores:
            ax.set_title(f"Pred: {class_names[int(preds[j])]}  (p={conf_pred[j]:.2f})", fontsize=10)
        else:
            ax.set_title("Grad-CAM", fontsize=10)
        ax.axis("off")

    # Titres globaux discrets et mise en page
    fig.suptitle("Illustration Grad-CAM (3 classes)", fontsize=12)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    # Sauvegarde haute résolution pour insertion article
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    print(f"[OK] Figure enregistrée : {save_path}")

# ======================
# ===== Utilisation =====
# ======================
# Si tu normalises en ImageNet, passe denorm_mean/std :
# plot_gradcam_panel_three_classes(
#     cam, test_loader, class_names=CLASS_NAMES,
#     save_path="gradcam_3classes_panel.png",
#     overlay_alpha=0.45,
#     denorm_mean=IMAGENET_MEAN, denorm_std=IMAGENET_STD,
#     show_pred_scores=True, dpi=300
# )

plot_gradcam_panel_three_classes(
    cam, test_loader, class_names=CLASS_NAMES,
    save_path="gradcam_3classes_panel.png",
    overlay_alpha=0.45,
    denorm_mean=None, denorm_std=None,  # ← mets tes moyennes/écarts si besoin
    show_pred_scores=True, dpi=300
)

# ----

# ============================================================
# Grad-CAM + affichage image originale & superposition heatmap
# + extraction ciblée pour la classe "Normales"
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# (Optionnel) valeurs ImageNet si tes DataLoaders normalisent en ImageNet
# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD  = (0.229, 0.224, 0.225)

# -------------------------
# Utils d'affichage/entrée
# -------------------------
def denormalize(imgs, mean=None, std=None):
    """
    imgs: Tensor (B,3,H,W)
    mean, std: tuples len=3. Si None -> pas de denorm.
    """
    imgs = imgs.detach().cpu().float()
    if (mean is None) or (std is None):
        return imgs.clamp(0, 1)
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std  = torch.tensor(std).view(1, 3, 1, 1)
    imgs = imgs * std + mean
    return imgs.clamp(0, 1)

def get_confidence_vector(probs, preds=None):
    """
    probs: (B,C) probabilités softmax.
    preds: (B,) classes. Si None -> argmax(probs).
    Retour: (B,) confiances associées.
    """
    probs_np = probs.detach().cpu().numpy() if isinstance(probs, torch.Tensor) else np.asarray(probs)
    if preds is None:
        preds_idx = probs_np.argmax(axis=1)
    else:
        preds_idx = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.asarray(preds)
    return probs_np[np.arange(len(probs_np)), preds_idx]

def show_gradcam_with_original(
    images,
    heatmaps,
    true_labels=None,
    pred_labels=None,
    class_names=None,
    scores=None,
    score_name="p",
    cols=4,
    denorm_mean=None,
    denorm_std=None,
    overlay_alpha=0.4
):
    """
    Affiche l'image originale + la superposition Grad-CAM côte à côte.
    """
    imgs = images
    if imgs.shape[1] == 1:  # grayscale → RGB
        imgs = imgs.repeat(1, 3, 1, 1)

    imgs = denormalize(imgs, mean=denorm_mean, std=denorm_std)

    B = imgs.shape[0]
    rows = int(np.ceil(B / cols))
    plt.figure(figsize=(8 * cols, 4 * rows))  # 2 colonnes (original + overlay)

    for i in range(B):
        img = np.transpose(imgs[i].numpy(), (1, 2, 0))
        hm = heatmaps[i].numpy()

        # Original
        ax1 = plt.subplot(rows, cols * 2, 2 * i + 1)
        ax1.imshow(img, interpolation='bilinear')
        title_left = []
        if true_labels is not None and class_names is not None:
            title_left.append(f"True: {class_names[int(true_labels[i])]}")
        ax1.set_title(" | ".join(title_left), fontsize=9)
        ax1.axis("off")

        # Overlay Grad-CAM
        ax2 = plt.subplot(rows, cols * 2, 2 * i + 2)
        ax2.imshow(img, interpolation='bilinear')
        ax2.imshow(hm, cmap='jet', alpha=overlay_alpha, interpolation='bilinear')
        title_right = []
        if pred_labels is not None and class_names is not None:
            title_right.append(f"Pred: {class_names[int(pred_labels[i])]}")
        if scores is not None:
            title_right.append(f"{score_name}={float(scores[i]):.3f}")
        ax2.set_title(" | ".join(title_right), fontsize=9)
        ax2.axis("off")

    plt.tight_layout()
    plt.show()

# -------------------------
# Trouver une couche cible
# -------------------------
def find_last_conv_module(model: nn.Module):
    """
    Retourne le dernier nn.Conv2d trouvé en parcourant le modèle.
    Utile si tu ne connais pas exactement le chemin (ex: model.cnn.layer4[-1]).
    """
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("Aucun nn.Conv2d trouvé dans le modèle pour Grad-CAM.")
    return last_conv

# -------------
# Grad-CAM core
# -------------
class GradCAM:
    """
    Grad-CAM pour un module convolutionnel cible (target_layer).
    """
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_hook = target_layer.register_forward_hook(self._save_activations)
        try:
            self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)
        except Exception:
            self.bwd_hook = target_layer.register_backward_hook(self._save_gradients_fallback)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()  # (B,C,H,W)

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()  # (B,C,H,W)

    def _save_gradients_fallback(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        try: self.fwd_hook.remove()
        except Exception: pass
        try: self.bwd_hook.remove()
        except Exception: pass

    @torch.no_grad()
    def _normalize(self, cam):
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def __call__(self, images, targets=None, device=None):
        """
        images: (B,3,H,W) ou (B,1,H,W)
        targets: (B,) indices de classes. None => classe prédite.
        return: heatmaps (B,H,W), preds (B,), probs (B,num_classes)
        """
        device = device or next(self.model.parameters()).device
        images = images.to(device)

        outputs = self.model(images)  # (B,C)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        if targets is None:
            targets = preds
        else:
            targets = torch.as_tensor(targets, device=device)

        self.model.zero_grad(set_to_none=True)
        selected = outputs.gather(1, targets.view(-1, 1)).sum()
        selected.backward()

        A = self.activations      # (B,C,H,W)
        dA = self.gradients       # (B,C,H,W)
        weights = dA.flatten(2).mean(dim=2).view(A.size(0), A.size(1), 1, 1)  # (B,C,1,1)
        cam = (weights * A).sum(dim=1)                                        # (B,H,W)
        cam = F.relu(cam)

        heatmaps = torch.stack([self._normalize(cam[i]) for i in range(cam.size(0))], dim=0)
        return heatmaps.cpu(), preds.detach().cpu(), probs.detach().cpu()

# -----------------------------------------
# Récupérer des exemples d'une classe donnée
# -----------------------------------------
def gather_class_examples(loader, class_id, max_images=8, device=None):
    """
    Parcourt le loader et renvoie jusqu'à max_images images de la classe `class_id`.
    """
    imgs_chunks, labels_chunks = [], []
    total = 0
    for images, labels in loader:
        mask = (labels == class_id)
        if mask.any():
            imgs_chunks.append(images[mask])
            labels_chunks.append(labels[mask])
            total += int(mask.sum().item())
            if total >= max_images:
                break
    if not imgs_chunks:
        raise RuntimeError(f"Aucune image trouvée pour la classe id={class_id}.")
    imgs = torch.cat(imgs_chunks, dim=0)[:max_images]
    labs = torch.cat(labels_chunks, dim=0)[:max_images]
    if device is not None:
        imgs = imgs.to(device)
        labs = labs.to(device)
    return imgs, labs

# ============================
# ======== EXEMPLE ========== #
# ============================
# Hypothèses: objets existants : model, test_loader, Config (avec DEVICE)
CLASS_NAMES = ["Autres", "Falciformes", "Normales"]  # adapte si besoin

device = getattr(Config, "DEVICE", next(model.parameters()).device)

# 1) Choix de la couche cible: essai direct, sinon auto-détection
try:
    target_layer = model.cnn.layer4[-1]  # ← adapte au chemin réel de ton modèle
except Exception:
    target_layer = find_last_conv_module(model)
    print("[Info] target_layer auto-détecté (dernier Conv2d):", target_layer.__class__.__name__)

# 2) Instancier Grad-CAM
cam = GradCAM(model.to(device).eval(), target_layer=target_layer)

# 3) Trouver l'ID de la classe "Normales" de manière robuste
if hasattr(test_loader, "dataset") and hasattr(test_loader.dataset, "class_to_idx"):
    class_to_idx = test_loader.dataset.class_to_idx
    normal_name_candidates = ["Normales", "Normal", "Normals"]  # ajoute d'autres variantes si besoin
    NORMAL_ID = None
    for name in normal_name_candidates:
        if name in class_to_idx:
            NORMAL_ID = class_to_idx[name]
            break
    if NORMAL_ID is None:
        NORMAL_ID = CLASS_NAMES.index("Normales")  # fallback via CLASS_NAMES
else:
    NORMAL_ID = CLASS_NAMES.index("Normales")
print(f"[Info] ID 'Normales' = {NORMAL_ID}")

# 4) Extraire des exemples de la classe "Normales"
try:
    normals_images, normals_labels = gather_class_examples(
        test_loader, class_id=NORMAL_ID, max_images=8, device=device
    )
except RuntimeError as e:
    print("[Alerte]", e)
    normals_images = None

# 5) Calcul Grad-CAM et affichage
if normals_images is not None:
    # Explication par rapport à la vérité (cible = 'Normales')
    heatmaps_n, preds_n, probs_n = cam(normals_images, targets=normals_labels, device=device)

    # (Option) Pour forcer l'explication sur "Normales" même si la prédiction diffère :
    # forced_targets = torch.full((normals_images.size(0),), NORMAL_ID, device=device, dtype=torch.long)
    # heatmaps_n, preds_n, probs_n = cam(normals_images, targets=forced_targets, device=device)

    conf_n = get_confidence_vector(probs_n, preds=preds_n)

    show_gradcam_with_original(
        images=normals_images,
        heatmaps=heatmaps_n,
        true_labels=normals_labels.detach().cpu(),
        pred_labels=preds_n,
        class_names=CLASS_NAMES,
        scores=conf_n,
        score_name="p_pred",
        cols=4,
        # denorm_mean=IMAGENET_MEAN,  # décommente si tes images sont normalisées
        # denorm_std=IMAGENET_STD,
        overlay_alpha=0.4
    )

# 6) Nettoyer les hooks
cam.remove_hooks()

# ----

# ============================================================
# Grad-CAM + affichage image originale & superposition heatmap
# ============================================================
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# (Optionnel) si tes DataLoaders normalisent en ImageNet :
# mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)

def denormalize(imgs, mean=None, std=None):
    """
    imgs: Tensor (B,3,H,W) sur CPU ou GPU dans [0,1] normalisé éventuellement.
    mean, std: tuples/list len=3. Si None -> pas de denorm.
    Retourne un Tensor (B,3,H,W) CPU en [0,1] clampé.
    """
    imgs = imgs.detach().cpu().float()
    if (mean is None) or (std is None):
        return imgs.clamp(0, 1)

    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std  = torch.tensor(std).view(1, 3, 1, 1)
    imgs = imgs * std + mean
    return imgs.clamp(0, 1)

class GradCAM:
    """
    Grad-CAM classique pour un module convolutionnel cible.
    target_layer: nn.Module (ex: model.cnn.layer4[-1] pour ResNet)
    """
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Hook forward
        self.fwd_hook = target_layer.register_forward_hook(self._save_activations)

        # Hook backward (full_backward_hook si dispo, sinon fallback)
        try:
            self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)
        except Exception:
            self.bwd_hook = target_layer.register_backward_hook(self._save_gradients_fallback)

    def _save_activations(self, module, input, output):
        # output: (B, C, H, W)
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        # grad_output[0]: (B, C, H, W)
        self.gradients = grad_output[0].detach()

    # Pour compatibilité versions plus anciennes de PyTorch
    def _save_gradients_fallback(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        try:
            self.fwd_hook.remove()
        except Exception:
            pass
        try:
            self.bwd_hook.remove()
        except Exception:
            pass

    @torch.no_grad()
    def _normalize(self, cam):
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def __call__(self, images, targets=None, device=None):
        """
        images: tensor (B,3,H,W) ou (B,1,H,W)
        targets: liste/tensor de classes cibles (B,) ou None (=classe prédite)
        return: heatmaps (B,H,W) dans [0,1], preds (B,), probs (B,num_classes)
        """
        device = device or next(self.model.parameters()).device
        images = images.to(device)

        # Forward
        outputs = self.model(images)  # (B, num_classes)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        # Cibles de backprop
        if targets is None:
            targets = preds
        else:
            targets = torch.as_tensor(targets, device=device)

        # Backward sur le logit de la classe cible
        self.model.zero_grad(set_to_none=True)
        selected = outputs.gather(1, targets.view(-1, 1)).sum()
        selected.backward()

        # Activations & gradients -> carte CAM
        A = self.activations           # (B,C,H,W)
        dA = self.gradients            # (B,C,H,W)

        # Poids: moyenne spatiale des gradients
        weights = dA.flatten(2).mean(dim=2).view(A.size(0), A.size(1), 1, 1)  # (B,C,1,1)
        cam = (weights * A).sum(dim=1)                                        # (B,H,W)
        cam = F.relu(cam)

        # Normalisation par image
        heatmaps = torch.stack([self._normalize(cam[i]) for i in range(cam.size(0))], dim=0)
        return heatmaps.cpu(), preds.detach().cpu(), probs.detach().cpu()

def get_confidence_vector(probs, preds=None):
    """
    probs: Tensor/ndarray (B, num_classes) de probabilités (softmax).
    preds: Tensor/ndarray (B,) indices de classes. Si None -> argmax(probs).
    Retourne: ndarray (B,) des confiances correspondantes.
    """
    if isinstance(probs, torch.Tensor):
        probs_np = probs.detach().cpu().numpy()
    else:
        probs_np = np.asarray(probs)

    if preds is None:
        preds_idx = probs_np.argmax(axis=1)
    else:
        preds_idx = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.asarray(preds)

    return probs_np[np.arange(len(probs_np)), preds_idx]

def show_gradcam_with_original(
    images,
    heatmaps,
    true_labels=None,
    pred_labels=None,
    class_names=None,
    scores=None,
    score_name="p",
    cols=4,
    denorm_mean=None,
    denorm_std=None,
    overlay_alpha=0.4
):
    """
    Affiche l'image originale + la superposition Grad-CAM côte à côte.

    images: tensor (B,3,H,W) ou (B,1,H,W)
    heatmaps: tensor (B,H,W)
    true_labels, pred_labels: (B,)
    class_names: liste de noms de classes (len = num_classes)
    scores: array-like (B,) -> affiché comme score_name=0.987 (ex. proba)
    denorm_mean/std: tuples ImageNet si tes images sont normalisées
    overlay_alpha: transparence de la heatmap
    """
    imgs = images
    if imgs.shape[1] == 1:  # grayscale → RGB
        imgs = imgs.repeat(1, 3, 1, 1)

    # Dé-normalise pour affichage si nécessaire
    imgs = denormalize(imgs, mean=denorm_mean, std=denorm_std)

    B = imgs.shape[0]
    rows = int(np.ceil(B / cols))
    plt.figure(figsize=(8 * cols, 4 * rows))  # 2 colonnes par image (original + overlay)

    for i in range(B):
        img = np.transpose(imgs[i].numpy(), (1, 2, 0))  # (H,W,3), [0,1]
        hm = heatmaps[i].numpy()

        # --- Colonne 1: Image originale ---
        ax1 = plt.subplot(rows, cols * 2, 2 * i + 1)
        ax1.imshow(img, interpolation='bilinear')
        title_left = []
        if true_labels is not None and class_names is not None:
            title_left.append(f"True: {class_names[int(true_labels[i])]}")
        ax1.set_title(" | ".join(title_left), fontsize=9)
        ax1.axis("off")

        # --- Colonne 2: Overlay Grad-CAM ---
        ax2 = plt.subplot(rows, cols * 2, 2 * i + 2)
        ax2.imshow(img, interpolation='bilinear')
        ax2.imshow(hm, cmap='jet', alpha=overlay_alpha, interpolation='bilinear')

        title_right = []
        if pred_labels is not None and class_names is not None:
            title_right.append(f"Pred: {class_names[int(pred_labels[i])]}")
        if scores is not None:
            title_right.append(f"{score_name}={float(scores[i]):.3f}")
        ax2.set_title(" | ".join(title_right), fontsize=9)
        ax2.axis("off")

    plt.tight_layout()
    plt.show()


# ============================
# ======== EXEMPLE ========== #
# ============================

# --- Paramètres ---
CLASS_NAMES = ["Autres", "Falciformes", "Normales"]  # adapte à ton mapping
device = Config.DEVICE  # suppose que tu as déjà Config.DEVICE défini

# --- Choix de la couche cible ---
# Ex: ResNet -> le dernier bloc de layer4
# Adapte ce chemin selon ton modèle (ex: model.backbone.layer4[-1], etc.)
target_layer = model.cnn.layer4[-1]

# --- Instancier Grad-CAM ---
cam = GradCAM(model.to(device).eval(), target_layer=target_layer)

# --- Mini-batch du test pour visualisation ---
batch_images, batch_labels = next(iter(test_loader))
batch_images = batch_images.to(device)
batch_labels = batch_labels.to(device)

# (Option) choisir les classes cibles = vraies étiquettes pour voir l'explication "correcte"
# sinon, mets targets=None pour expliquer la classe prédite
heatmaps, preds, probs = cam(batch_images, targets=batch_labels, device=device)

# Confiance sur la classe prédite (max softmax)
conf_pred = get_confidence_vector(probs, preds=preds)

# --- Afficher: image originale + Grad-CAM ---
# Si tes images sont normalisées ImageNet, dé-commente les deux lignes suivantes
# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD  = (0.229, 0.224, 0.225)

show_gradcam_with_original(
    images=batch_images,
    heatmaps=heatmaps,
    true_labels=batch_labels.cpu(),
    pred_labels=preds,
    class_names=CLASS_NAMES,
    scores=conf_pred,
    score_name="p_pred",
    cols=4,
    # denorm_mean=IMAGENET_MEAN,   # ← décommente si besoin
    # denorm_std=IMAGENET_STD,     # ← décommente si besoin
    overlay_alpha=0.4
)

# --- Nettoyer les hooks si tu n'en as plus besoin ---
cam.remove_hooks()

# ----

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

def set_publication_style(base_fontsize=10):
    # Police sans-serif lisible, tailles cohérentes, fond blanc
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize,
        "axes.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize-1,
        "xtick.labelsize": base_fontsize-1,
        "ytick.labelsize": base_fontsize-1,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.transparent": False,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def set_seed_all(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed);
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_publication_style(10)
set_seed_all(42)

# ----

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def denormalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    x = img_tensor.detach().cpu().float()
    if x.shape[0] == 1:  # grayscale -> pseudo-RGB
        x = x.repeat(3,1,1)
    m = torch.tensor(mean).view(-1,1,1)
    s = torch.tensor(std).view(-1,1,1)
    x = (x * s) + m
    x = x.clamp(0,1)
    return np.transpose(x.numpy(), (1,2,0))

def pick_indices_per_class(labels, class_ids, k_per_class=3):
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    idxs = []
    for c in class_ids:
        pool = np.where(labels == c)[0]
        if len(pool) > 0:
            k = min(k_per_class, len(pool))
            chosen = np.random.choice(pool, size=k, replace=False)
            idxs.extend(chosen.tolist())
    return idxs

def get_confidence_vector(probs, preds=None, targets=None):
    probs_np = probs.detach().cpu().numpy()
    if preds is not None:
        idx = preds.detach().cpu().numpy()
    elif targets is not None:
        idx = targets.detach().cpu().numpy()
    else:
        raise ValueError("Fournir preds OU targets.")
    return probs_np[np.arange(len(probs_np)), idx]

# ----

def save_gradcam_panel(images, heatmaps,
                       true_labels=None, pred_labels=None,
                       class_names=None, indices=None,
                       k_per_class=3, choose_by="pred",
                       scores_pred=None,               # proba de la classe prédite (0..1)
                       out_path="fig_gradcam_test.png",
                       cols=2,
                       tile_height_in=2.0, tile_width_in=2.2,  # tuiles + petites
                       cmap="magma", alpha=0.45, dpi=300,
                       add_colorbar=False):
    """
    Titre court: 'True: X | Pred: Y | conf: 98%'
    Images rapprochées (wspace/hspace faibles)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    assert class_names is not None, "class_names requis."
    imgs = images.detach().cpu()
    hmaps = heatmaps.detach().cpu().numpy() if isinstance(heatmaps, torch.Tensor) else heatmaps

    class_ids = np.arange(len(class_names))
    # Sélection équilibrée par classe (prédite par défaut)
    if indices is None:
        base = pred_labels if choose_by == "pred" else true_labels
        if base is None:
            raise ValueError("choose_by='pred' nécessite pred_labels, 'true' nécessite true_labels.")
        indices = pick_indices_per_class(base, class_ids, k_per_class=k_per_class)

    n = len(indices)
    rows = int(np.ceil(n))
    fig_w = cols * tile_width_in
    fig_h = rows * tile_height_in

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h),
                             gridspec_kw={"wspace": 0.02, "hspace": 0.04})  # images rapprochées
    if rows == 1:
        axes = np.array([axes])

    # Boucle
    for j, idx in enumerate(indices):
        r = j  # une rangée par échantillon (2 colonnes)
        img_np = denormalize(imgs[idx])          # image originale
        hm     = hmaps[idx]                      # heatmap déjà normalisé

        # Noms courts & minuscules
        true_txt = class_names[int(true_labels[idx])].lower() if true_labels is not None else "?"
        pred_txt = class_names[int(pred_labels[idx])].lower() if pred_labels is not None else "?"
        # Confiance en %
        if scores_pred is not None:
            conf_pct = int(round(float(scores_pred[idx]) * 100))
            conf_txt = f" | conf: {conf_pct}%"
        else:
            conf_txt = ""

        title_short = f"True: {true_txt} | Pred: {pred_txt}{conf_txt}"

        # Col 1 : Original
        ax1 = axes[r, 0]
        ax1.imshow(img_np, interpolation="nearest")
        ax1.set_title(title_short, fontsize=8)
        ax1.axis("off")

        # Col 2 : Grad-CAM
        ax2 = axes[r, 1]
        ax2.imshow(img_np, interpolation="nearest")
        ax2.imshow(hm, cmap=cmap, alpha=alpha, interpolation="bilinear")
        ax2.set_title("Grad-CAM", fontsize=8)
        ax2.axis("off")

    # Sauvegarde serrée
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"🖼️  Figure enregistrée : {out_path} (dpi={dpi})")

# ----

# ============================
# 0) Style "publication"
# ============================
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

def set_publication_style(base_fontsize=10):
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize,
        "axes.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize-1,
        "xtick.labelsize": base_fontsize-1,
        "ytick.labelsize": base_fontsize-1,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.transparent": False,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def set_seed_all(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_publication_style(10)
set_seed_all(42)

# ----

# ============================
# 1) Helpers (denorm, pick, scores)
# ============================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def denormalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    img_tensor: (C,H,W) torch.Tensor — retourne un numpy (H,W,C) dans [0,1].
    """
    x = img_tensor.detach().cpu().float()
    if x.shape[0] == 1:  # grayscale -> pseudo-RGB
        x = x.repeat(3,1,1)
    m = torch.tensor(mean).view(-1,1,1)
    s = torch.tensor(std).view(-1,1,1)
    x = (x * s) + m
    x = x.clamp(0,1)
    return np.transpose(x.numpy(), (1,2,0))

def pick_indices_per_class(labels, class_ids, k_per_class=3):
    """
    Retourne jusqu'à k_per_class indices par classe pour une couverture équilibrée.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    idxs = []
    for c in class_ids:
        pool = np.where(labels == c)[0]
        if len(pool) > 0:
            k = min(k_per_class, len(pool))
            chosen = np.random.choice(pool, size=k, replace=False)
            idxs.extend(chosen.tolist())
    return idxs

def get_confidence_vector(probs, preds=None, targets=None):
    """
    probs : torch.Tensor (B, num_classes)
    preds : Tensor (B,) classes prédites
    targets : Tensor (B,) classes vraies
    Retour : np.array (B,) des probabilités correspondantes
    """
    probs_np = probs.detach().cpu().numpy()
    if preds is not None:
        idx = preds.detach().cpu().numpy()
    elif targets is not None:
        idx = targets.detach().cpu().numpy()
    else:
        raise ValueError("Fournir preds OU targets.")
    return probs_np[np.arange(len(probs_np)), idx]

# ----

def save_gradcam_panel(images, heatmaps,
                       true_labels=None, pred_labels=None,
                       class_names=None, indices=None,
                       k_per_class=3, choose_by="pred",
                       scores_pred=None,
                       out_path="fig_gradcam_test.png",
                       cols=2,
                       tile_height_in=1.6, tile_width_in=1.8,  # encore plus compact
                       cmap="magma", alpha=0.45, dpi=300):
    """
    Affiche Original | Grad-CAM côte-à-côte.
    Texte 'True | Pred | conf' AU-DESSUS de l'image Grad-CAM (taille réduite).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    assert class_names is not None, "class_names requis."
    imgs = images.detach().cpu()
    hmaps = heatmaps.detach().cpu().numpy() if isinstance(heatmaps, torch.Tensor) else heatmaps

    class_ids = np.arange(len(class_names))
    if indices is None:
        base = pred_labels if choose_by == "pred" else true_labels
        indices = pick_indices_per_class(base, class_ids, k_per_class=k_per_class)

    n = len(indices)
    rows = int(np.ceil(n))
    fig_w = cols * tile_width_in
    fig_h = rows * tile_height_in

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h),
                             gridspec_kw={"wspace": 0.01, "hspace": 0.20})
    if rows == 1:
        axes = np.array([axes])

    for j, idx in enumerate(indices):
        r = j
        img_np = denormalize(imgs[idx])
        hm     = hmaps[idx]

        # Texte court
        true_txt = class_names[int(true_labels[idx])].lower() if true_labels is not None else "?"
        pred_txt = class_names[int(pred_labels[idx])].lower() if pred_labels is not None else "?"
        conf_txt = ""
        if scores_pred is not None:
            conf_pct = int(round(float(scores_pred[idx]) * 100))
            conf_txt = f" | conf: {conf_pct}%"
        title_short = f"True: {true_txt} | Pred: {pred_txt}{conf_txt}"

        # Col 1 : Original
        ax1 = axes[r, 0]
        ax1.imshow(img_np, interpolation="nearest")
        ax1.set_title("Original", fontsize=6)  # plus petit
        ax1.axis("off")

        # Col 2 : Grad-CAM
        ax2 = axes[r, 1]
        ax2.imshow(img_np, interpolation="nearest")
        ax2.imshow(hm, cmap=cmap, alpha=alpha, interpolation="bilinear")
        ax2.set_title(title_short, fontsize=6)  # texte compact
        ax2.axis("off")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"🖼️  Figure enregistrée : {out_path} (dpi={dpi})")

# ----

# ============================
# 3) Utilisation (PNG 300 dpi + TIFF 600 dpi)
# ============================
CLASS_NAMES = ["Autres", "Falciformes", "Normales"]
device = Config.DEVICE

# 1) Obtenir un batch de test
batch_images, batch_labels = next(iter(test_loader))
batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

# 2) Grad-CAM (cible = classe prédite pour chaque image)
#    cam = GradCAM(model.to(device).eval(), target_layer=model.cnn.layer4[-1])  # déjà défini plus haut
heatmaps, preds, probs = cam(batch_images, targets=None, device=device)

# 3) Score de confiance (classe prédite)
conf_pred = get_confidence_vector(probs, preds=preds)

# 4) Sauvegardes prêtes pour l'article
# PNG 300 dpi
save_gradcam_panel(batch_images, heatmaps,
                   true_labels=batch_labels, pred_labels=preds,
                   class_names=CLASS_NAMES,
                   k_per_class=3, choose_by="pred",
                   scores_pred=conf_pred,
                   out_path="fig_gradcam_test_compact_smalltext.png",
                   cmap="magma", alpha=0.45, dpi=300)


# TIFF 600 dpi (si requis par le journal)
save_gradcam_panel(batch_images, heatmaps,
                   true_labels=batch_labels, pred_labels=preds,
                   class_names=CLASS_NAMES,
                   k_per_class=3, choose_by="pred",
                   scores_pred=conf_pred,  # <-- pas de scores_true ici
                   out_path="fig_gradcam_test_600dpi.tif",
                   cmap="magma", alpha=0.45, dpi=600)

# ----

def collect_gradcam_samples(model, cam, loader, class_names, k_per_class=3, device="cuda"):
    """
    Parcourt le loader jusqu'à avoir au moins k_per_class exemples par classe.
    Retourne les tenseurs et sélections équilibrées.
    """
    all_images, all_labels, all_heatmaps, all_preds, all_probs = [], [], [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Grad-CAM
        heatmaps, preds, probs = cam(images, targets=None, device=device)

        all_images.append(images.cpu())
        all_labels.append(labels.cpu())
        all_heatmaps.append(heatmaps)
        all_preds.append(preds)
        all_probs.append(probs)

        # Vérifier si toutes les classes sont couvertes
        if len(set(torch.cat(all_labels).numpy())) == len(class_names):
            # suffisant pour au moins un exemple par classe
            break

    # Concaténer
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_heatmaps = torch.cat(all_heatmaps, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_probs = torch.cat(all_probs, dim=0)

    # Sélection équilibrée (k par classe)
    indices = pick_indices_per_class(all_labels, np.arange(len(class_names)), k_per_class=k_per_class)

    return all_images, all_labels, all_heatmaps, all_preds, all_probs, indices

# ----

CLASS_NAMES = ["Autres", "Falciformes", "Normales"]

# On collecte les images + heatmaps pour toutes les classes
imgs, labels, hmaps, preds, probs, idxs = collect_gradcam_samples(
    model, cam, test_loader, CLASS_NAMES, k_per_class=3, device=device
)

# Scores
conf_pred = get_confidence_vector(probs, preds=preds)

# Affichage équilibré
save_gradcam_panel(imgs, hmaps,
                   true_labels=labels, pred_labels=preds,
                   class_names=CLASS_NAMES,
                   indices=idxs,   # <- on force la sélection équilibrée
                   scores_pred=conf_pred,
                   out_path="fig_gradcam_all_classes.png",
                   cmap="magma", alpha=0.45, dpi=300)

# ----

from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

def extract_features(model, loader, device="cuda"):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            lbls = lbls.to(device)

            # Forward jusqu’aux features (avant la classification finale)
            # ⚠️ Adapter selon ton HybridModel (par ex. model.forward_features)
            x = model.cnn(images)          # exemple : CNN backbone
            x = torch.flatten(x, 1)        # aplatissement
            feats.append(x.cpu())
            labels.append(lbls.cpu())

    feats = torch.cat(feats, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return feats, labels

# ----

import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def _to_2d_features(x: torch.Tensor) -> torch.Tensor:
    """
    Reçoit un Tensor de features et le convertit en (B, D).
    - Si 4D (B,C,H,W) -> GAP + flatten
    - Si 2D (B,D)     -> retourne tel quel
    - Autre           -> essaie flatten(B, -1)
    """
    if x.dim() == 4:                 # feature map conv
        x = F.adaptive_avg_pool2d(x, 1)  # (B,C,1,1)
        x = x.flatten(1)                 # (B,C)
    elif x.dim() == 2:
        pass
    else:
        x = x.flatten(1)
    return x

def _pick_tensor_from_output(out):
    """
    Prend une sortie potentiellement complexe (Tensor / list / tuple / dict)
    et retourne un Tensor de features raisonnable.
    - list/tuple: prend le 1er élément Tensor le plus 'grand'
    - dict: essaie clés usuelles ('feat','features','x','out')
    - Tensor: retourne tel quel
    """
    if torch.is_tensor(out):
        return out
    if isinstance(out, (list, tuple)):
        # garde le Tensor avec le plus grand nombre d'éléments
        tensors = [t for t in out if torch.is_tensor(t)]
        if len(tensors) == 0:
            raise TypeError("La sortie est une liste/tuple sans Tensor.")
        return max(tensors, key=lambda t: t.numel())
    if isinstance(out, dict):
        for k in ("feat", "features", "x", "out"):
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        # sinon, première valeur tensor
        for v in out.values():
            if torch.is_tensor(v):
                return v
        raise TypeError("Dict de sortie sans Tensor utilisable.")
    raise TypeError(f"Type de sortie non pris en charge: {type(out)}")

def extract_features(model, loader, device="cuda"):
    """
    Extrait des features (B, D) pour chaque batch du loader.
    Essaie model.forward_features(...) si dispo, sinon model.cnn(...).
    """
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            lbls = lbls.to(device)

            # 1) chemin 'propre' si dispo
            if hasattr(model, "forward_features"):
                out = model.forward_features(images)
            # 2) sinon, tenter le backbone CNN
            elif hasattr(model, "cnn"):
                out = model.cnn(images)
            else:
                # fallback: utiliser tout le modèle puis récupérer l'avant-dernière couche ?
                # Ici, on prend la sortie et on la traite quand même (pas idéal si ce sont des logits)
                out = model(images)

            # récupérer un Tensor depuis out (gère Tensor/list/tuple/dict)
            x = _pick_tensor_from_output(out)
            # mettre en (B,D)
            x = _to_2d_features(x)

            feats.append(x.cpu())
            labels.append(lbls.cpu())

    feats = torch.cat(feats, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return feats, labels

def compute_tsne(features, labels, class_names, out_path="tsne_test.png",
                 perplexity=30, n_iter=1000, random_state=42):
    # sécuriser perplexity (doit être < n_samples)
    n = features.shape[0]
    perp = max(5, min(perplexity, max(5, n//3)))
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=n_iter,
                init="pca", learning_rate="auto", random_state=random_state)
    feats_2d = tsne.fit_transform(features)

    plt.figure(figsize=(6,6))
    for i, cname in enumerate(class_names):
        idxs = labels == i
        if np.any(idxs):
            plt.scatter(feats_2d[idxs, 0], feats_2d[idxs, 1],
                        label=cname, alpha=0.75, s=10)
    plt.legend(frameon=False)
    plt.title("t-SNE des représentations (test)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"🖼️  t-SNE enregistré : {out_path}")

# ----

CLASS_NAMES = ["Autres", "Falciformes", "Normales"]
device = Config.DEVICE

features, labels = extract_features(model, test_loader, device=device)
compute_tsne(features, labels, CLASS_NAMES, out_path="tsne_test.png")

# ----

import umap.umap_ as umap

def compute_umap(features, labels, class_names,
                 n_neighbors=15, min_dist=0.1,
                 out_path="umap_test.png", random_state=42):
    """
    Projette les features en 2D avec UMAP et sauvegarde une figure.
    """
    reducer = umap.UMAP(n_components=2,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        random_state=random_state)
    feats_2d = reducer.fit_transform(features)

    plt.figure(figsize=(6,6))
    for i, cname in enumerate(class_names):
        idxs = labels == i
        if np.any(idxs):
            plt.scatter(feats_2d[idxs, 0], feats_2d[idxs, 1],
                        label=cname, alpha=0.75, s=10)
    plt.legend(frameon=False)
    plt.title("UMAP des représentations (test)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"🖼️  UMAP enregistré : {out_path}")

# ----

CLASS_NAMES = ["others", "Sickle Cells", "Normal"]

# Réutilise les mêmes features extraits pour t-SNE
features, labels = extract_features(model, test_loader, device=device)

# Visualisation UMAP
compute_umap(features, labels, CLASS_NAMES, out_path="umap_test.png")

# ----

# =========================
# t-SNE vs UMAP – Comparatif
# =========================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

try:
    import umap.umap_ as umap
except ImportError as e:
    raise ImportError("UMAP non installé. Fais:  pip install umap-learn") from e

def compute_tsne(features, perplexity=30, n_iter=1000, random_state=42):
    n = features.shape[0]
    perp = max(5, min(perplexity, max(5, n // 3)))  # sécuriser perplexity
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=n_iter,
                init="pca", learning_rate="auto", random_state=random_state)
    return tsne.fit_transform(features)

def compute_umap(features, n_neighbors=15, min_dist=0.1, random_state=42):
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=random_state)
    return reducer.fit_transform(features)

def plot_tsne_umap_side_by_side(tsne_2d, umap_2d, labels, class_names,
                                title_tsne="t-SNE (test)", title_umap="UMAP (test)",
                                out_path_png="tsne_umap_comparison.png",
                                out_path_tif="tsne_umap_comparison.tif",
                                s=10, alpha=0.75, dpi=300):
    """
    Affiche et sauvegarde une figure 1x2 : t-SNE | UMAP
    """
    # même limites d'axes par méthode (indépendantes) pour un cadrage propre
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    for ax, emb, ttl in zip(axes, [tsne_2d, umap_2d], [title_tsne, title_umap]):
        for i, cname in enumerate(class_names):
            idxs = (labels == i)
            if np.any(idxs):
                ax.scatter(emb[idxs, 0], emb[idxs, 1], s=s, alpha=alpha, label=cname)
        ax.set_title(ttl)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Légende unique en bas
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=len(class_names), frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)  # place pour la légende
    fig.savefig(out_path_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path_tif, dpi=dpi, bbox_inches="tight")
    plt.show()
    print(f"🖼️ Figure enregistrée : {out_path_png} et {out_path_tif} (dpi={dpi})")

# ========= Utilisation =========
CLASS_NAMES = ["Autres", "Falciformes", "Normales"]

# 1) Réutilise la fonction d’extraction que tu as déjà (features avant la dernière couche)
# features, labels = extract_features(model, test_loader, device=Config.DEVICE)

# 2) Embeddings
tsne_2d = compute_tsne(features, perplexity=30, n_iter=1000, random_state=42)
umap_2d = compute_umap(features, n_neighbors=15, min_dist=0.1, random_state=42)

# 3) Plot & export (PNG + TIFF, 300 dpi)
plot_tsne_umap_side_by_side(tsne_2d, umap_2d, labels, CLASS_NAMES,
                            title_tsne="t-SNE (Test)", title_umap="UMAP (Test)",
                            out_path_png="tsne_umap_comparison.png",
                            out_path_tif="tsne_umap_comparison.tif",
                            s=10, alpha=0.75, dpi=300)

# ----

import torch
import numpy as np

def collect_misclassified_with_cam(model, cam, loader, class_names,
                                   k_per_class=4, device="cuda",
                                   select="confident"):
    """
    Sélectionne des erreurs (par classe vraie), avec Grad-CAM.
    select: 'confident' = erreurs à plus forte confiance (classe prédite).
    """
    buckets = {c: [] for c in range(len(class_names))}
    model.eval()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # --- forward classique sans grads pour aller vite
        with torch.no_grad():
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

        wrong = (preds != labels)

        if select == "confident":
            conf_pred = get_confidence_vector(probs, preds=preds)  # numpy
            conf_pred_t = torch.tensor(conf_pred, device=device)
            idxs = torch.nonzero(wrong, as_tuple=False).flatten()
            if len(idxs) > 0:
                order = torch.argsort(conf_pred_t[idxs], descending=True)
                idxs = idxs[order]
        else:
            idxs = torch.nonzero(wrong, as_tuple=False).flatten()

        # --- pour chaque erreur sélectionnée: calculer Grad-CAM (avec gradients)
        for i in idxs.tolist():
            true_c = int(labels[i])
            if len(buckets[true_c]) >= k_per_class:
                continue

            img_i = images[i].unsqueeze(0)

            # Très important: réactiver les gradients ici
            with torch.enable_grad():
                # si tu utilises AMP ailleurs, on le coupe pour le CAM
                try:
                    from torch.cuda.amp import autocast
                    amp_cm = autocast(enabled=False)
                except Exception:
                    class _Dummy:
                        def __enter__(self): pass
                        def __exit__(self, *a): pass
                    amp_cm = _Dummy()

                with amp_cm:
                    # Optionnel: s'assurer que les poids peuvent backprop
                    for p in model.parameters():
                        if p.grad is None and not p.requires_grad:
                            p.requires_grad_(True)

                    hm_i, pred_i, probs_i = cam(img_i, targets=None, device=device)

            pred_i = int(pred_i.item())
            prob_pred_i = float(get_confidence_vector(probs_i, preds=torch.tensor([pred_i]))[0])

            buckets[true_c].append({
                "image": images[i].detach().cpu(),
                "label": int(labels[i]),
                "pred":  pred_i,
                "prob_pred": prob_pred_i,
                "heatmap": hm_i.squeeze(0).detach().cpu()
            })

        if all(len(buckets[c]) >= k_per_class for c in buckets):
            break

    # Rassembler
    imgs_list, labels_list, preds_list, probs_list, hmaps_list = [], [], [], [], []
    for c in range(len(class_names)):
        for item in buckets[c]:
            imgs_list.append(item["image"])
            labels_list.append(item["label"])
            preds_list.append(item["pred"])
            probs_list.append(item["prob_pred"])
            hmaps_list.append(item["heatmap"])

    if len(imgs_list) == 0:
        print("⚠️  Aucune erreur trouvée (ou pas assez dans ce loader).")
        return None

    imgs = torch.stack(imgs_list, dim=0)
    labels = torch.tensor(labels_list, dtype=torch.long)
    preds  = torch.tensor(preds_list,  dtype=torch.long)
    probs_pred = torch.tensor(probs_list, dtype=torch.float32)
    hmaps = torch.stack(hmaps_list, dim=0)
    indices = list(range(len(imgs)))
    return imgs, labels, preds, probs_pred, hmaps, indices

# ----

def collect_correct_low_conf_with_cam(model, cam, loader, class_names,
                                      k_per_class=4, device="cuda"):
    """
    Exemples bien classés mais à faible confiance (cas limites).
    """
    buckets = {c: [] for c in range(len(class_names))}
    model.eval()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # --- forward sans grads
        with torch.no_grad():
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

        conf_pred = get_confidence_vector(probs, preds=preds)  # numpy
        conf_pred_t = torch.tensor(conf_pred, device=device)

        correct = (preds == labels)
        idxs = torch.nonzero(correct, as_tuple=False).flatten()
        if len(idxs) > 0:
            order = torch.argsort(conf_pred_t[idxs], descending=False)  # faible confiance d'abord
            idxs = idxs[order]

        for i in idxs.tolist():
            true_c = int(labels[i])
            if len(buckets[true_c]) >= k_per_class:
                continue

            img_i = images[i].unsqueeze(0)

            # --- Grad-CAM avec gradients
            with torch.enable_grad():
                try:
                    from torch.cuda.amp import autocast
                    amp_cm = autocast(enabled=False)
                except Exception:
                    class _Dummy:
                        def __enter__(self): pass
                        def __exit__(self, *a): pass
                    amp_cm = _Dummy()

                with amp_cm:
                    for p in model.parameters():
                        if p.grad is None and not p.requires_grad:
                            p.requires_grad_(True)
                    hm_i, pred_i, probs_i = cam(img_i, targets=None, device=device)

            pred_i = int(pred_i.item())
            prob_pred_i = float(get_confidence_vector(probs_i, preds=torch.tensor([pred_i]))[0])

            buckets[true_c].append({
                "image": images[i].detach().cpu(),
                "label": int(labels[i]),
                "pred":  pred_i,
                "prob_pred": prob_pred_i,
                "heatmap": hm_i.squeeze(0).detach().cpu()
            })

        if all(len(buckets[c]) >= k_per_class for c in buckets):
            break

    imgs_list, labels_list, preds_list, probs_list, hmaps_list = [], [], [], [], []
    for c in range(len(class_names)):
        for item in buckets[c]:
            imgs_list.append(item["image"])
            labels_list.append(item["label"])
            preds_list.append(item["pred"])
            probs_list.append(item["prob_pred"])
            hmaps_list.append(item["heatmap"])

    if len(imgs_list) == 0:
        print("⚠️  Pas assez de cas corrects à faible confiance.")
        return None

    imgs = torch.stack(imgs_list, dim=0)
    labels = torch.tensor(labels_list, dtype=torch.long)
    preds  = torch.tensor(preds_list,  dtype=torch.long)
    probs_pred = torch.tensor(probs_list, dtype=torch.float32)
    hmaps = torch.stack(hmaps_list, dim=0)
    indices = list(range(len(imgs)))
    return imgs, labels, preds, probs_pred, hmaps, indices

# ----

# Erreurs haute confiance
res_err = collect_misclassified_with_cam(model, cam, test_loader, CLASS_NAMES,
                                         k_per_class=3, device=device, select="confident")
if res_err is not None:
    imgs_e, labels_e, preds_e, conf_e, hmaps_e, idxs_e = res_err
    save_gradcam_panel(imgs_e, hmaps_e,
                       true_labels=labels_e, pred_labels=preds_e,
                       class_names=CLASS_NAMES, indices=idxs_e,
                       scores_pred=conf_e,
                       out_path="errors_high_conf_gradcam.png",
                       cmap="magma", alpha=0.45, dpi=300)

# (Optionnel) Corrects faible confiance
res_low = collect_correct_low_conf_with_cam(model, cam, test_loader, CLASS_NAMES,
                                            k_per_class=3, device=device)
if res_low is not None:
    imgs_c, labels_c, preds_c, conf_c, hmaps_c, idxs_c = res_low
    save_gradcam_panel(imgs_c, hmaps_c,
                       true_labels=labels_c, pred_labels=preds_c,
                       class_names=CLASS_NAMES, indices=idxs_c,
                       scores_pred=conf_c,
                       out_path="correct_low_conf_gradcam.png",
                       cmap="magma", alpha=0.45, dpi=300)

# ----

import time, math, torch
from contextlib import nullcontext

# ========= 1) Compter les paramètres =========
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# ========= 2) Mesurer l’inférence sur un DataLoader =========
@torch.no_grad()
def benchmark_inference(model, loader, device="cuda", amp=False, warmup_batches=5):
    """
    Retourne: dict(total_imgs, total_time_s, imgs_per_s, ms_per_img)
    """
    device_str = str(device)  # <-- conversion
    was_training = model.training
    model.eval().to(device)

    # warmup
    it = iter(loader)
    for _ in range(min(warmup_batches, len(loader))):
        try:
            images, _ = next(it)
        except StopIteration:
            break
        images = images.to(device, non_blocking=True)
        if device_str.startswith("cuda"):
            torch.cuda.synchronize()
        with torch.autocast(device_type="cuda", dtype=torch.float16) if (amp and device_str.startswith("cuda")) else nullcontext():
            _ = model(images)
        if device_str.startswith("cuda"):
            torch.cuda.synchronize()

    # mesure
    total_imgs = 0
    if device_str.startswith("cuda"):
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        bs = images.size(0)
        with torch.autocast(device_type="cuda", dtype=torch.float16) if (amp and device_str.startswith("cuda")) else nullcontext():
            _ = model(images)
        total_imgs += bs

    if device_str.startswith("cuda"):
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    total_time = t_end - t_start

    imgs_per_s = total_imgs / total_time if total_time > 0 else float("inf")
    ms_per_img = (total_time / total_imgs) * 1000.0 if total_imgs > 0 else float("inf")

    if was_training:
        model.train()
    return {
        "total_imgs": total_imgs,
        "total_time_s": total_time,
        "imgs_per_s": imgs_per_s,
        "ms_per_img": ms_per_img,
        "amp_used": bool(amp),
        "device": device_str,
    }

def print_model_runtime_report(model, loader, img_size, device="cuda", amp=False, warmup_batches=5, in_channels=3):
    device_str = str(device)  # <-- conversion
    H, W = (img_size, img_size) if isinstance(img_size, int) else img_size
    total_params, trainable_params = count_parameters(model)
    print("=== PARAMÈTRES ===")
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")

    stats = benchmark_inference(model, loader, device=device, amp=amp, warmup_batches=warmup_batches)
    print("\n=== TEMPS D'INFERENCE (sur DataLoader) ===")
    print(f"Device           : {stats['device']}  | AMP: {stats['amp_used']}")
    print(f"Images traitées  : {stats['total_imgs']}")
    print(f"Temps total      : {stats['total_time_s']:.3f} s")
    print(f"Throughput       : {stats['imgs_per_s']:.2f} img/s")
    print(f"Latence moyenne  : {stats['ms_per_img']:.2f} ms / image")

    flops, msg = compute_flops_macs(model, input_shape=(1, in_channels, H, W), device=device)
    print("\n=== COMPLEXITÉ (thop) ===")
    if flops is None:
        print(msg)
    else:
        print(f"MACs             : {flops['MACs']:,}  (~{flops['GMACs']:.2f} GMACs)")
        print(f"GFLOPs (≈2*MACs) : {flops['GFLOPs_approx']:.2f}")
        print(f"Params (thop)    : {flops['params']:,}")


# ========= 5) Exemple d’utilisation =========
# Hypothèses:
# - model est déjà chargé et sur Config.DEVICE
# - test_loader (ou val_loader) est défini
# - Config.IMG_SIZE / Config.IN_CHANNELS existent (sinon remplace par tes valeurs)
try:
    in_ch = getattr(Config, "IN_CHANNELS", 3)
    img_sz = getattr(Config, "IMG_SIZE", 224)
except NameError:
    in_ch, img_sz = 3, 224

print_model_runtime_report(
    model=model,
    loader=test_loader,       # ou val_loader
    img_size=img_sz,          # ex: 224 ou (224,224)
    in_channels=in_ch,        # 1 si grayscale
    device=Config.DEVICE,
    amp=True,                 # True si tu infères en autocast FP16 sur GPU
    warmup_batches=5
)

# ----

@torch.no_grad()
def evaluate_accuracy(model, loader, device="cuda", amp=False):
    was_training = model.training
    model.eval().to(device)
    correct, total = 0, 0

    # petite accélération si GPU
    use_amp = (amp and str(device).startswith("cuda"))
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
        else:
            logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.numel()

    if was_training:
        model.train()
    return 100.0 * correct / max(1, total)

# ----

def gather_model_runtime_metrics(model, loader, img_size, in_channels=3,
                                 device="cuda", amp=True, warmup_batches=5):
    H, W = (img_size, img_size) if isinstance(img_size, int) else img_size
    total_params, trainable_params = count_parameters(model)

    # Inference bench
    stats = benchmark_inference(model, loader, device=device, amp=amp, warmup_batches=warmup_batches)
    # FLOPs/MACs (si thop installé)
    flops, msg = compute_flops_macs(model, input_shape=(1, in_channels, H, W), device=device)
    # Accuracy
    acc = evaluate_accuracy(model, loader, device=device, amp=amp)

    metrics = {
        "Model": getattr(model, "__class__", type("X",(object,),{})).__name__,
        "ImgSize": f"{H}x{W}",
        "InCh": in_channels,
        "Device": str(device),
        "AMP": bool(stats["amp_used"]),
        "TotalParams": int(total_params),
        "TrainableParams": int(trainable_params),
        "Images": int(stats["total_imgs"]),
        "TotalTime_s": float(stats["total_time_s"]),
        "Throughput_img_per_s": float(stats["imgs_per_s"]),
        "Latency_ms_per_img": float(stats["ms_per_img"]),
        "Accuracy_pct": float(acc),
        "MACs": None, "GMACs": None, "GFLOPs_approx": None,
    }
    if flops is None:
        metrics["MACs_msg"] = msg
    else:
        metrics["MACs"] = int(flops["MACs"])
        metrics["GMACs"] = float(flops["GMACs"])
        metrics["GFLOPs_approx"] = float(flops["GFLOPs_approx"])
        metrics["MACs_msg"] = ""
    return metrics

# ----

import csv
from textwrap import dedent

def save_metrics_csv(metrics: dict, path="runtime_metrics.csv", round_digits=3):
    keys = list(metrics.keys())
    # arrondir quelques colonnes numériques pour lisibilité
    for k in ["TotalTime_s", "Throughput_img_per_s", "Latency_ms_per_img", "GMACs", "GFLOPs_approx"]:
        if k in metrics and isinstance(metrics[k], (float, int)) and metrics[k] is not None:
            metrics[k] = round(float(metrics[k]), round_digits)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerow(metrics)
    print(f"📄 CSV écrit: {path}")

def save_metrics_latex(metrics: dict, path="runtime_metrics.tex", caption="Résumé des performances d'inférence", label="tab:runtime"):
    # mise en forme avec séparateurs de milliers
    def fmt(x, k):
        if x is None: return "--"
        if k in ("TotalParams", "TrainableParams", "MACs"):
            return f"{int(x):,}".replace(",", " ")
        if k in ("GMACs", "GFLOPs_approx", "TotalTime_s", "Throughput_img_per_s", "Latency_ms_per_img"):
            return f"{float(x):.3f}"
        return str(x)

    rows = [
        ("Modèle",                fmt(metrics["Model"], "Model")),
        ("Entrée (HxW, C)",      f'{metrics["ImgSize"]}, {metrics["InCh"]}'),
        ("Périphérique",         fmt(metrics["Device"], "Device")),
        ("AMP",                  "Oui" if metrics["AMP"] else "Non"),
        ("Paramètres (total)",   fmt(metrics["TotalParams"], "TotalParams")),
        ("Paramètres (appr.)",   fmt(metrics["TrainableParams"], "TrainableParams")),
        ("Images évaluées",      fmt(metrics["Images"], "Images")),
        ("Temps total (s)",      fmt(metrics["TotalTime_s"], "TotalTime_s")),
        ("Débit (img/s)",        fmt(metrics["Throughput_img_per_s"], "Throughput_img_per_s")),
        ("Latence (ms/img)",     fmt(metrics["Latency_ms_per_img"], "Latency_ms_per_img")),
        ("MACs",                 fmt(metrics["MACs"], "MACs")),
        ("GMACs",                fmt(metrics["GMACs"], "GMACs")),
        ("GFLOPs (≈2×MACs)",     fmt(metrics["GFLOPs_approx"], "GFLOPs_approx")),
    ]

    table = dedent(rf"""
    \begin{{table}}[t]
      \centering
      \caption{{{caption}}}
      \label{{{label}}}
      \begin{{tabular}}{{l r}}
        \toprule
        {" \\\\".join([f"{k} & {v}" for k,v in rows])} \\\\
        \bottomrule
      \end{{tabular}}
    \end{{table}}
    """).strip()

    # ajouter les packages si tu les utilises (booktabs)
    preamble_hint = "% Requiert \\usepackage{booktabs}\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(preamble_hint + table + "\n")
    print(f"📝 LaTeX écrit: {path}  (n'oublie pas \\usepackage{{booktabs}})")

# ----

# Définis in_ch et img_sz selon ta Config
try:
    in_ch = getattr(Config, "IN_CHANNELS", 3)
    img_sz = getattr(Config, "IMG_SIZE", 224)
except NameError:
    in_ch, img_sz = 3, 224

metrics = gather_model_runtime_metrics(
    model=model,
    loader=test_loader,
    img_size=img_sz,
    in_channels=in_ch,
    device=Config.DEVICE,  # torch.device OK
    amp=True,              # mets False si tu ne veux pas d'autocast
    warmup_batches=5
)

save_metrics_csv(metrics, path="runtime_metrics.csv")
save_metrics_latex(metrics, path="runtime_metrics.tex",
                   caption="Paramètres, temps d'inférence sur le test et complexité (MACs/GFLOPs).",
                   label="tab:runtime_overview")

# (Option) affichage rapide
from pprint import pprint
pprint(metrics)

# ----

import time, torch
from contextlib import nullcontext

def benchmark_split(model, loader, device="cuda", amp=True, warmup_batches=5):
    model.eval()
    iters, n_imgs, t0 = 0, 0, 0.0
    amp_ctx = torch.amp.autocast("cuda") if (amp and device.startswith("cuda")) else nullcontext()
    # --- warmup ---
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            with amp_ctx:
                _ = model(x)
            iters += 1; n_imgs += x.size(0)
            if iters >= warmup_batches: break
    if device.startswith("cuda"): torch.cuda.synchronize()
    # --- timed pass ---
    n_imgs = 0
    t_start = time.perf_counter()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            with amp_ctx:
                _ = model(x)
    if device.startswith("cuda"): torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_time = t_end - t_start
    total_imgs = len(loader.dataset)
    throughput = total_imgs / total_time
    latency_ms = 1000.0 / throughput
    return {"Images": total_imgs, "TotalTime_s": total_time,
            "Throughput_img_per_s": throughput, "Latency_ms_per_img": latency_ms}

# Exemple :
train_stats = benchmark_split(model, train_loader, device="cuda", amp=True)
val_stats   = benchmark_split(model, val_loader,   device="cuda", amp=True)
test_stats  = benchmark_split(model, test_loader,  device="cuda", amp=True)
print(train_stats, val_stats, test_stats)

# ----

import matplotlib.pyplot as plt

# ==== Données (édite ici) ====
model_name = "HybridModel (DrepaViT)"
hardware    = "Intel Core i5-8365U • 8GB RAM"
input_shape = "224×224×3"

params_total = 26_389_827         # paramètres
gflops_per_img = 9.221            # GFLOPs / image
gmacs_per_img  = 4.611            # GMACs / image

times_sec = {                      # temps totaux (secondes)
    "Train (N=2024)": 12.168114131000038,
    "Validation (N=577)": 1.9403658350001933,
    "Test (N=324)": 1.282117255999765,
}
throughput = {                     # img/s
    "Train (N=2024)": 166.33637539966574,
    "Validation (N=577)": 297.3666045815234,
    "Test (N=324)": 252.7069957789098,
}
latency_ms = {                     # ms / image
    "Train (N=2024)": 6.011914096343893,
    "Validation (N=577)": 3.3628524003469558,
    "Test (N=324)": 3.9571520246906324,
}

plt.rcParams.update({"font.size": 11})

def annotate(bars, fmt="{:.2f}", suffix="", fontsize=10):
    for r in bars:
        h = r.get_height()
        plt.text(r.get_x()+r.get_width()/2, h, fmt.format(h)+suffix,
                 ha="center", va="bottom", fontsize=fontsize)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))

# (a) Complexité
ax = axes[0]
labels_a = ["Parameters (M)", "GFLOPs / img", "GMACs / img"]
vals_a   = [params_total/1e6, gflops_per_img, gmacs_per_img]
bars = ax.bar(labels_a, vals_a)
annotate(bars, "{:.2f}")
ax.set_ylabel("Value")
ax.set_title("(a) Model Complexity — " + model_name)
ax.tick_params(axis="x", rotation=12)

# (b) Temps d'inférence
ax = axes[1]
labels_b = list(times_sec.keys())
vals_b   = [times_sec[k] for k in labels_b]
bars = ax.bar(labels_b, vals_b)
annotate(bars, "{:.2f}", "s")
ax.set_ylabel("Total time (s)")
ax.set_title("(b) Inference Time per Split")
ax.tick_params(axis="x", rotation=12)

# Petite légende texte (débit/latence)
cap = (f"Throughput / Latency — "
       f"Train: {throughput[labels_b[0]]:.1f} img/s; {latency_ms[labels_b[0]]:.2f} ms  |  "
       f"Val: {throughput[labels_b[1]]:.1f}; {latency_ms[labels_b[1]]:.2f} ms  |  "
       f"Test: {throughput[labels_b[2]]:.1f}; {latency_ms[labels_b[2]]:.2f} ms")
fig.suptitle(f"{model_name}  •  Input {input_shape}  •  {hardware}", y=1.03, fontsize=12)
fig.text(0.5, 0.01, cap, ha="center", fontsize=9)

plt.tight_layout(rect=[0, 0.04, 1, 0.98])
plt.savefig("fig_drepavit_complexity_ab.png", dpi=400, bbox_inches="tight")
plt.show()

# ----

import matplotlib.pyplot as plt

# ===== Réglages taille & résolution =====
FIG_W, FIG_H = 8, 5.2    # ← augmente FIG_W pour allonger la figure
SAVE_DPI     = 450        # résolution export
FONT_SIZE    = 11
plt.rcParams.update({"font.size": FONT_SIZE})

# ===== Données (édite si besoin) =====
model_name  = "HybridModel (DrepaViT)"
hardware    = "Intel Core i5-8365U • 8GB RAM"
input_shape = "224×224×3"

params_total   = 26_389_827         # paramètres absolus
gflops_per_img = 9.221              # GFLOPs / image
gmacs_per_img  = 4.611              # GMACs / image

times_sec = {                        # temps totaux (secondes)
    "Train (N=2024)": 12.168114131000038,
    "Validation (N=577)": 1.9403658350001933,
    "Test (N=324)": 1.282117255999765,
}
throughput = {                       # img/s
    "Train (N=2024)": 166.33637539966574,
    "Validation (N=577)": 297.3666045815234,
    "Test (N=324)": 252.7069957789098,
}
latency_ms = {                       # ms / image
    "Train (N=2024)": 6.011914096343893,
    "Validation (N=577)": 3.3628524003469558,
    "Test (N=324)": 3.9571520246906324,
}

# ===== Palettes (couleurs distinctes) =====
palette_a = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # (a) Params / GFLOPs / GMACs
palette_b = ["#4c72b0", "#dd8452", "#55a868"]   # (b) Train / Val / Test

# ===== Utilitaires =====
def annotate_fallback(ax, bars, labels, pad_frac=0.06, fontsize=9):
    """Annoter au-dessus des barres si Matplotlib<3.4 (sans bar_label)."""
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for r, txt in zip(bars, labels):
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h + pad_frac*span, txt,
                ha="center", va="bottom", fontsize=fontsize, clip_on=False)

# ===== Figure =====
fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
# Réserver de la place pour le supertitre et la légende bas
fig.subplots_adjust(top=0.86, bottom=0.26, wspace=0.30)

# ---------- (a) Complexité ----------
ax = axes[0]
labels_a = ["Parameters (M)", "GFLOPs / img", "GMACs / img"]
vals_a   = [params_total/1e6, gflops_per_img, gmacs_per_img]
bars_a   = ax.bar(labels_a, vals_a, color=palette_a)
ax.set_ylabel("Value")
ax.set_title("(a) Model Complexity — " + model_name)
ax.tick_params(axis="x", rotation=12)
ax.set_ylim(0, max(vals_a) * 1.40)  # headroom pour les labels
ax.margins(x=0.06)

# Labels au-dessus des barres
labels_text_a = [f"{v:.2f}" for v in vals_a]
if hasattr(ax, "bar_label"):
    ax.bar_label(bars_a, labels=labels_text_a, label_type="edge", padding=4, fontsize=9)
else:
    annotate_fallback(ax, bars_a, labels_text_a, pad_frac=0.08, fontsize=9)

# ---------- (b) Temps d'inférence ----------
ax = axes[1]
labels_b = list(times_sec.keys())
vals_b   = [times_sec[k] for k in labels_b]
bars_b   = ax.bar(labels_b, vals_b, color=palette_b)
ax.set_ylabel("Total time (s)")
ax.set_title("(b) Inference Time per Split")
ax.tick_params(axis="x", rotation=20)
ax.set_ylim(0, max(vals_b) * 1.40)  # headroom
ax.margins(x=0.08)

labels_text_b = [f"{v:.2f}s" for v in vals_b]
if hasattr(ax, "bar_label"):
    ax.bar_label(bars_b, labels=labels_text_b, label_type="edge", padding=4, fontsize=9)
else:
    annotate_fallback(ax, bars_b, labels_text_b, pad_frac=0.08, fontsize=9)

# ---------- Supertitre & légende bas ----------
fig.suptitle(f"{model_name}  •  Input {input_shape}  •  {hardware}",
             y=0.92, fontsize=12)

cap = (f"Throughput / Latency — "
       f"Train: {throughput[labels_b[0]]:.1f} img/s; {latency_ms[labels_b[0]]:.2f} ms  |  "
       f"Val: {throughput[labels_b[1]]:.1f} img/s; {latency_ms[labels_b[1]]:.2f} ms  |  "
       f"Test: {throughput[labels_b[2]]:.1f} img/s; {latency_ms[labels_b[2]]:.2f} ms")
fig.text(0.5, 0.08, cap, ha="center", va="bottom", fontsize=9)

# ---------- Export ----------
plt.tight_layout(rect=[0, 0.14, 1, 0.88])
plt.savefig("fig_drepavit_complexity_ab_colored_vfinal.png",
            dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.35)
plt.savefig("fig_drepavit_complexity_ab_colored_vfinal.pdf",
            dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.35)
plt.show()

# ----

def gather_multiple_models(models, loaders, img_size, in_channels=3, device="cuda", amp=True, warmup_batches=5):
    """
    models : dict { "Nom": model_obj }
    loaders: dict { "Nom": loader }  (si tu veux loader spécifique par modèle, sinon passe le même partout)
    Retourne: list[dict] (une ligne de métriques par modèle)
    """
    results = []
    for name, model in models.items():
        loader = loaders.get(name, list(loaders.values())[0])  # prend le premier loader si pas précisé
        print(f"\n🔎 Benchmark du modèle: {name}")
        m = gather_model_runtime_metrics(model, loader,
                                         img_size=img_size,
                                         in_channels=in_channels,
                                         device=device,
                                         amp=amp,
                                         warmup_batches=warmup_batches)
        m["Model"] = name  # forcer nom lisible
        results.append(m)
    return results

# ----

def save_metrics_multi_csv(results, path="runtime_metrics_multi.csv", round_digits=3):
    if not results:
        return
    keys = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            row = r.copy()
            for k in ["TotalTime_s", "Throughput_img_per_s", "Latency_ms_per_img", "GMACs", "GFLOPs_approx"]:
                if k in row and isinstance(row[k], (float, int)) and row[k] is not None:
                    row[k] = round(float(row[k]), round_digits)
            w.writerow(row)
    print(f"📄 CSV comparatif écrit: {path}")

# ----

def save_metrics_multi_latex(results, path="runtime_metrics_multi.tex",
                             caption="Comparaison des modèles sur temps d'inférence et complexité",
                             label="tab:runtime_multi"):
    if not results:
        return

    headers = ["Modèle", "Params", "Débit (img/s)", "Latence (ms)", "GMACs", "GFLOPs"]
    lines = []
    for r in results:
        params = f"{int(r['TotalParams']):,}".replace(",", " ")
        thr = f"{r['Throughput_img_per_s']:.2f}"
        lat = f"{r['Latency_ms_per_img']:.2f}"
        gmacs = f"{r['GMACs']:.2f}" if r["GMACs"] is not None else "--"
        gflops = f"{r['GFLOPs_approx']:.2f}" if r["GFLOPs_approx"] is not None else "--"
        lines.append(f"{r['Model']} & {params} & {thr} & {lat} & {gmacs} & {gflops} \\\\")

    table = dedent(rf"""
    \begin{{table}}[t]
      \centering
      \caption{{{caption}}}
      \label{{{label}}}
      \begin{{tabular}}{{l r r r r r}}
        \toprule
        {' & '.join(headers)} \\\\
        \midrule
        {'\n'.join(lines)}
        \bottomrule
      \end{{tabular}}
    \end{{table}}
    """).strip()

    with open(path, "w", encoding="utf-8") as f:
        f.write("% Requiert \\usepackage{booktabs}\n")
        f.write(table + "\n")
    print(f"📝 LaTeX comparatif écrit: {path}")

# ----

# Suppose que tu as deux modèles: baseline CNN et ton HybridModel
models = {
    "BaselineCNN": baseline_model,
    "HybridModel": model
}

# Si un seul loader pour tous (test_loader par ex.)
loaders = { "BaselineCNN": test_loader, "HybridModel": test_loader }

results = gather_multiple_models(models, loaders,
                                 img_size=img_sz,
                                 in_channels=in_ch,
                                 device=Config.DEVICE,
                                 amp=True,
                                 warmup_batches=5)

save_metrics_multi_csv(results, path="runtime_metrics_multi.csv")
save_metrics_multi_latex(results, path="runtime_metrics_multi.tex",
                         caption="Comparaison Baseline CNN vs HybridModel sur le test",
                         label="tab:runtime_comparison")

from pprint import pprint
pprint(results)

# ----

import matplotlib.pyplot as plt

# ====== Tes chiffres DrepaViT (tu peux ajuster ici) ======
drepavit = {
    "name": "DrepaViT@224",
    "acc": 0.99,
    "precision": 0.9872,
    "recall": 0.9864,          # = sensibilité
    "f1": 0.9867,
    "specificity": 0.9938,
    "mcc": 0.9807,
    "auc": 0.9996,
    "params": 26_389_827,
    "gflops": 9.288,
}

# ====== (A) Figure GFLOPs par modèle (avec DrepaViT ajouté) ======
models = [
    ("MobileNetV2@224",      2_261_827,   0.613),
    ("VGG16@224",          134_272_835,  30.952),
    ("VGG19@224",          139_582_531,  39.277),
    ("ResNet50@224",        23_593_859,   7.751),
    ("EfficientNetB0@224",   4_053_414,   0.800),
    ("InceptionV3@299",     21_808_931,  11.465),
    ("Xception@299",        20_867_627,  16.770),
    (drepavit["name"],       drepavit["params"], drepavit["gflops"]),  # DrepaViT ajouté
]

# tri croissant par GFLOPs
models_sorted = sorted(models, key=lambda x: x[2])
names   = [m[0] for m in models_sorted]
paramsM = [m[1] / 1e6 for m in models_sorted]
gflops  = [m[2] for m in models_sorted]

labels = [f"{n} ({p:.1f}M)" for n, p in zip(names, paramsM)]
palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728",
           "#9467bd","#8c564b","#e377c2","#17becf","#7f7f7f"]
colors  = [palette[i % len(palette)] for i in range(len(labels))]

plt.rcParams.update({"font.size": 11})
plt.figure(figsize=(13.5, 6))
ypos = range(len(labels))
bars = plt.barh(ypos, gflops, color=colors)

for y, v in zip(ypos, gflops):
    plt.text(v, y, f"  {v:.3f}", va="center", ha="left", fontsize=10)

plt.yticks(ypos, labels)
plt.xlabel("GFLOPs / image (224×224 ; 299×299 indiqué)")
plt.title("Comparaison des GFLOPs par modèle (incluant DrepaViT)")
plt.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("fig_sota_gflops_with_drepavit.png", dpi=450, bbox_inches="tight")
plt.show()

# ====== (B) Figure métriques DrepaViT (barres) ======
metric_names = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "MCC", "AUC"]
metric_vals  = [drepavit["acc"], drepavit["precision"], drepavit["recall"],
                drepavit["f1"], drepavit["specificity"], drepavit["mcc"], drepavit["auc"]]
metric_pct   = [100*x for x in metric_vals]

plt.figure(figsize=(11, 4.8))
bars = plt.bar(metric_names, metric_pct, color=["#1f77b4","#ff7f0e","#2ca02c","#d62728",
                                                "#9467bd","#8c564b","#17becf"])
try:
    # labels au-dessus
    plt.gca().bar_label(bars, labels=[f"{v:.2f}%" for v in metric_pct],
                        label_type="edge", padding=3, fontsize=9)
except Exception:
    ymax = max(metric_pct) * 1.2
    plt.ylim(0, ymax)
    for r, txt in zip(bars, [f"{v:.2f}%" for v in metric_pct]):
        h = r.get_height()
        plt.text(r.get_x()+r.get_width()/2, h + 0.02*ymax, txt,
                 ha="center", va="bottom", fontsize=9)

plt.ylabel("Score (%)")
plt.title("DrepaViT — Métriques sur l'ensemble évalué")
plt.ylim(0, 105)
plt.grid(axis="y", linestyle="--", alpha=0.25)
plt.tight_layout()
plt.savefig("fig_drepavit_metrics_bar.png", dpi=450, bbox_inches="tight")
plt.show()

# ====== (C) Mini-card Params & GFLOPs (2 barres) ======
plt.figure(figsize=(6.8, 4.2))
labels = ["Parameters (M)", "GFLOPs / image"]
values = [drepavit["params"]/1e6, drepavit["gflops"]]
bars = plt.bar(labels, values, color=["#1f77b4", "#ff7f0e"])

try:
    plt.gca().bar_label(bars, labels=[f"{values[0]:.2f}M", f"{values[1]:.3f}"],
                        label_type="edge", padding=4, fontsize=10)
except Exception:
    ymax = max(values) * 1.35
    plt.ylim(0, ymax)
    for r, txt in zip(bars, [f"{values[0]:.2f}M", f"{values[1]:.3f}"]):
        h = r.get_height()
        plt.text(r.get_x()+r.get_width()/2, h + 0.03*ymax, txt,
                 ha="center", va="bottom", fontsize=10)

plt.ylabel("Value")
plt.title("DrepaViT — Parameters & GFLOPs @224×224")
plt.tight_layout()
plt.savefig("fig_drepavit_params_gflops.png", dpi=450, bbox_inches="tight")
plt.show()

# ----

# === Figure composite (a/b/c) : SOTA GFLOPs + métriques DrepaViT + Params/GFLOPs DrepaViT ===
import matplotlib.pyplot as plt
from PIL import Image
import os

# -----------------------
# 0) Données à personnaliser
# -----------------------
# SOTA + DrepaViT (GFLOPs)
models = [
    ("MobileNetV2@224",        2_261_827,   0.613),
    ("VGG16@224",            134_272_835,  30.952),
    ("VGG19@224",            139_582_531,  39.277),
    ("ResNet50@224",          23_593_859,   7.751),
    ("EfficientNetB0@224",     4_053_414,   0.800),
    ("InceptionV3@299",       21_808_931,  11.465),
    ("Xception@299",          20_867_627,  16.770),
    ("DrepaViT@224 (proposé)", 26_389_827,   9.288),  # DrepaViT inséré
]

# Métriques DrepaViT (valeurs 0..1)
drepavit = {
    "name": "DrepaViT",
    "acc": 0.99,
    "precision": 0.9872,
    "recall": 0.9864,          # Sensibilité
    "f1": 0.9867,
    "specificity": 0.9938,
    "mcc": 0.9807,
    "auc": 0.9996,
    "params": 26_389_827,
    "gflops": 9.288,
}

# -----------------------
# 1) Panneau (a) : GFLOPs par modèle (barres horizontales)
# -----------------------
def panel_a(path="panel_a_gflops.png", dpi=450):
    models_sorted = sorted(models, key=lambda x: x[2])
    names   = [m[0] for m in models_sorted]
    paramsM = [m[1] / 1e6 for m in models_sorted]
    gflops  = [m[2] for m in models_sorted]
    labels  = [f"{n} ({p:.1f}M)" for n, p in zip(names, paramsM)]

    # Palette simple (couleurs distinctes)
    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728",
               "#9467bd","#8c564b","#e377c2","#17becf","#7f7f7f"]
    colors = [palette[i % len(palette)] for i in range(len(labels))]

    plt.rcParams.update({"font.size": 11})
    fig_h = 6 + 0.25 * max(0, len(labels) - 7)
    plt.figure(figsize=(13.0, fig_h))
    ypos = range(len(labels))
    bars = plt.barh(ypos, gflops, color=colors)

    # Annotations au bout des barres
    for y, v in zip(ypos, gflops):
        plt.text(v, y, f"  {v:.3f}", va="center", ha="left", fontsize=10)

    plt.yticks(ypos, labels)
    plt.xlabel("GFLOPs / image (224×224 ; 299×299 indiqué)")
    plt.title("(a) GFLOPs par modèle (incluant DrepaViT)")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path

# -----------------------
# 2) Panneau (b) : Métriques DrepaViT (barres)
# -----------------------
def panel_b(path="panel_b_metrics.png", dpi=450):
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "MCC", "AUC"]
    metric_vals  = [drepavit["acc"], drepavit["precision"], drepavit["recall"],
                    drepavit["f1"], drepavit["specificity"], drepavit["mcc"], drepavit["auc"]]
    metric_pct   = [100*x for x in metric_vals]

    plt.rcParams.update({"font.size": 11})
    plt.figure(figsize=(11.5, 4.6))
    bars = plt.bar(metric_names, metric_pct,
                   color=["#1f77b4","#ff7f0e","#2ca02c","#d62728",
                          "#9467bd","#8c564b","#17becf"])
    # Labels au-dessus
    try:
        plt.gca().bar_label(bars, labels=[f"{v:.2f}%" for v in metric_pct],
                            label_type="edge", padding=3, fontsize=9)
    except Exception:
        ymax = max(metric_pct) * 1.2
        plt.ylim(0, ymax)
        for r, txt in zip(bars, [f"{v:.2f}%" for v in metric_pct]):
            h = r.get_height()
            plt.text(r.get_x()+r.get_width()/2, h + 0.02*ymax, txt,
                     ha="center", va="bottom", fontsize=9)

    plt.ylabel("Score (%)")
    plt.title("(b) Métriques du modèle DrepaViT")
    plt.ylim(0, 105); plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path

# -----------------------
# 3) Panneau (c) : Params & GFLOPs DrepaViT (2 barres)
# -----------------------
def panel_c(path="panel_c_params_gflops.png", dpi=450):
    plt.rcParams.update({"font.size": 11})
    plt.figure(figsize=(7.2, 4.2))
    labels = ["Parameters (M)", "GFLOPs / image"]
    values = [drepavit["params"]/1e6, drepavit["gflops"]]
    bars = plt.bar(labels, values, color=["#1f77b4", "#ff7f0e"])

    try:
        plt.gca().bar_label(bars, labels=[f"{values[0]:.2f}M", f"{values[1]:.3f}"],
                            label_type="edge", padding=4, fontsize=10)
    except Exception:
        ymax = max(values) * 1.35
        plt.ylim(0, ymax)
        for r, txt in zip(bars, [f"{values[0]:.2f}M", f"{values[1]:.3f}"]):
            h = r.get_height()
            plt.text(r.get_x()+r.get_width()/2, h + 0.03*ymax, txt,
                     ha="center", va="bottom", fontsize=10)

    plt.ylabel("Value")
    plt.title("(c) DrepaViT — Parameters & GFLOPs @224×224")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path

# -----------------------
# 4) Génération des 3 panneaux
# -----------------------
a_path = panel_a()
b_path = panel_b()
c_path = panel_c()

# -----------------------
# 5) Fusion en figure composite (a/b/c)
# -----------------------
def fuse_horiz(images, out_png="fig_composite_abc.png", out_pdf="fig_composite_abc.pdf", gap=28, bg="white"):
    ims = [Image.open(p).convert("RGB") for p in images]
    H = max(im.height for im in ims)

    # mettre toutes à la même hauteur (conservation des proportions)
    resized = []
    for im in ims:
        if im.height != H:
            new_w = int(im.width * (H / im.height))
            im = im.resize((new_w, H), Image.LANCZOS)
        resized.append(im)

    total_w = sum(im.width for im in resized) + gap*(len(resized)-1)
    canvas = Image.new("RGB", (total_w, H), color=bg)

    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width + gap

    canvas.save(out_png)
    try:
        canvas.save(out_pdf)  # PIL peut aussi exporter en PDF
    except Exception:
        pass
    return out_png, out_pdf

out_png, out_pdf = fuse_horiz([a_path, b_path, c_path],
                              out_png="fig_drepavit_composite_abc.png",
                              out_pdf="fig_drepavit_composite_abc.pdf")

print("Composite saved:", os.path.abspath(out_png), "|", os.path.abspath(out_pdf))
