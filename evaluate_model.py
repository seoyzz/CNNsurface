
import kagglehub
import torch
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ResNetForImageClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# 데이터셋 불러오기
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

val_data = datasets.ImageFolder(
    data_dir / "validation" / "images",
    transform=transform
)

val_loader = DataLoader(
    val_data,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# 학습한 모델 불러오기
model = ResNetForImageClassification.from_pretrained(
    "deepinspect_model"
)

model.eval()

true_labels = []
pred_labels = []

# 예측 실행
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(pixel_values=images)
        predictions = outputs.logits.argmax(dim=1)

        true_labels.extend(labels.tolist())
        pred_labels.extend(predictions.tolist())

# 성능 평가
accuracy = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels, average="macro")

print("정확도:", round(accuracy * 100, 2), "%")
print("F1-score:", round(f1, 4))

print(classification_report(
    true_labels,
    pred_labels,
    target_names=val_data.classes,
    zero_division=0
))

# 혼동행렬
cm = confusion_matrix(true_labels, pred_labels)

fig, ax = plt.subplots(figsize=(10, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=val_data.classes
)

disp.plot(ax=ax, xticks_rotation=45, cmap="Blues")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()
