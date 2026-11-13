
import kagglehub
from pathlib import Path
from torchvision import datasets, transforms

path = kagglehub.dataset_download(
    "kaustubhdikshit/neu-surface-defect-database"
)

data_dir = Path(path) / "NEU-DET"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(
    root=data_dir / "train" / "images",
    transform=transform
)

val_data = datasets.ImageFolder(
    root=data_dir / "validation" / "images",
    transform=transform
)

print("결함 종류:", train_data.classes)
print("학습 이미지:", len(train_data))
print("검증 이미지:", len(val_data))

image, label = train_data[0]
print("이미지 크기:", image.shape)
print("첫 번째 이미지 정답:", train_data.classes[label])
