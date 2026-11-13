
import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
import kagglehub

# 1. 재현성 설정
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("사용 장치:", device)

# 2. 데이터셋
root = kagglehub.dataset_download(
    "kaustubhdikshit/neu-surface-defect-database"
)
data_root = os.path.join(root, "NEU-DET")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(
    os.path.join(data_root, "train", "images"),
    transform=transform
)
val_dataset = datasets.ImageFolder(
    os.path.join(data_root, "validation", "images"),
    transform=transform
)

assert train_dataset.classes == val_dataset.classes

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=16, shuffle=False,
    num_workers=0
)

classes = train_dataset.classes
print("클래스:", classes)
print("학습 이미지:", len(train_dataset))
print("검증 이미지:", len(val_dataset))

# 3. CNN 모델
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

model = SimpleCNN(len(classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)

# 4. 학습 및 검증
epochs = 5
train_losses = []
train_accuracies = []
val_accuracies = []

best_acc = -1
best_state = None

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (
            outputs.argmax(1) == labels
        ).sum().item()
        total += labels.size(0)

    train_loss = total_loss / total
    train_acc = 100 * correct / total

    model.eval()
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            val_correct += (
                outputs.argmax(1) == labels
            ).sum().item()

    val_acc = 100 * val_correct / len(val_dataset)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Acc: {val_acc:.2f}%",
        flush=True
    )

# 5. 최적 모델 저장
os.makedirs("results", exist_ok=True)

model.load_state_dict(best_state)

torch.save({
    "model_state_dict": best_state,
    "classes": classes
}, "results/cnn_model.pth")

# 6. 최적 모델 평가
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(1).cpu().tolist()

        y_true.extend(labels.tolist())
        y_pred.extend(predictions)

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")

print(f"\nCNN Accuracy: {accuracy*100:.2f}%")
print(f"CNN Macro F1-score: {f1:.4f}")
print(classification_report(
    y_true, y_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0
))

# 7. 그래프 저장
x = range(1, epochs + 1)

plt.figure(figsize=(8, 5))
plt.plot(x, train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("CNN Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/cnn_training_loss.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(x, train_accuracies, marker="o", label="Train")
plt.plot(x, val_accuracies, marker="s", label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("CNN Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/cnn_training_accuracy.png", dpi=200)
plt.close()

cm = confusion_matrix(
    y_true, y_pred,
    labels=list(range(len(classes)))
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.tight_layout()
plt.savefig("results/cnn_confusion_matrix.png", dpi=200)
plt.close()

print("\nCNN 학습 및 평가 완료!")
print("결과 저장 위치: results/")
