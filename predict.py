
import kagglehub
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import ResNetForImageClassification

# 1. 학습된 모델 불러오기
model = ResNetForImageClassification.from_pretrained(
    "deepinspect_model"
)
model.eval()

# 2. 결함 종류
classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# 3. 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 4. 예측할 이미지 가져오기
path = kagglehub.dataset_download(
    "kaustubhdikshit/neu-surface-defect-database"
)

image_path = (
    Path(path) / "NEU-DET" / "validation"
    / "images" / "scratches" / "scratches_241.jpg"
)

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)

# 5. AI 예측
with torch.no_grad():
    outputs = model(pixel_values=image_tensor)
    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted = probabilities.argmax(dim=1).item()
    confidence = probabilities[0][predicted].item() * 100

# 6. 결과 출력
print("실제 결함: scratches")
print("AI 예측:", classes[predicted])
print(f"예측 확률: {confidence:.2f}%")
