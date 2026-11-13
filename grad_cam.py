
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from transformers import ResNetForImageClassification
from torchvision import transforms

# 1. 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNetForImageClassification.from_pretrained(
    "deepinspect_model"
).to(device)

model.eval()

# 2. 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 3. 이미지 경로 설정
image_path = input("분석할 이미지 경로를 입력하세요: ").strip().strip('"')

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# 4. Grad-CAM 대상 계층 설정
target_layer = model.resnet.encoder.stages[-1].layers[-1]

activations = None
gradients = None

def forward_hook(module, inputs, output):
    global activations
    activations = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

forward_handle = target_layer.register_forward_hook(forward_hook)
backward_handle = target_layer.register_full_backward_hook(backward_hook)

# 5. 예측 및 역전파
model.zero_grad()

outputs = model(pixel_values=input_tensor)
logits = outputs.logits

predicted_class = logits.argmax(dim=1).item()
confidence = F.softmax(logits, dim=1)[0, predicted_class].item()

logits[0, predicted_class].backward()

# 6. Grad-CAM 계산
weights = gradients.mean(dim=(2, 3), keepdim=True)

cam = (weights * activations).sum(dim=1, keepdim=True)
cam = F.relu(cam)

cam = F.interpolate(
    cam,
    size=(224, 224),
    mode="bilinear",
    align_corners=False
)

cam = cam.squeeze().detach().cpu().numpy()

cam = cam - cam.min()
cam = cam / (cam.max() + 1e-8)

# 7. 시각화
original = image.resize((224, 224))
original_array = np.array(original)

class_names = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

class_name = class_names[predicted_class]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

axes[0].imshow(original_array)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(original_array)
axes[1].imshow(cam, cmap="jet", alpha=0.45)
axes[1].set_title(
    f"Grad-CAM: {class_name}\nConfidence: {confidence:.2%}"
)
axes[1].axis("off")

plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/grad_cam.png", dpi=200, bbox_inches="tight")
plt.show()

forward_handle.remove()
backward_handle.remove()

print("예측 결함:", class_name)
print(f"예측 신뢰도: {confidence:.2%}")
print("Grad-CAM 저장 완료: results/grad_cam.png")
