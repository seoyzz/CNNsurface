
import kagglehub
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ResNetForImageClassification

# 데이터셋
path = kagglehub.dataset_download(
    "kaustubhdikshit/neu-surface-defect-database"
)
data_dir = Path(path) / "NEU-DET"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_data = datasets.ImageFolder(
    data_dir / "train" / "images",
    transform=transform
)

train_loader = DataLoader(
    train_data, batch_size=16, shuffle=True, num_workers=0
)

# 기존 1 Epoch 학습 모델 불러오기
model = ResNetForImageClassification.from_pretrained(
    "deepinspect_model"
)

device = torch.device("cpu")
model.to(device)

# 분류 계층만 학습
for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    model.classifier.parameters(), lr=0.0005
)

# 5 Epoch 추가 학습
epochs = 5
loss_history = []
accuracy_history = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values=images, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total * 100

    loss_history.append(avg_loss)
    accuracy_history.append(accuracy)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {avg_loss:.4f} | "
        f"Accuracy: {accuracy:.2f}%",
        flush=True
    )

# 모델 저장
model.save_pretrained("deepinspect_model")
print("5 Epoch 추가 학습 완료!")

# 학습 그래프
plt.figure(figsize=(10, 4))
plt.plot(range(1, epochs+1), loss_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("DeepInspect Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=200)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(range(1, epochs+1), accuracy_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy (%)")
plt.title("DeepInspect Training Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_accuracy.png", dpi=200)
plt.show()
