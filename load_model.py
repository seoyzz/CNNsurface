
from transformers import AutoImageProcessor, ResNetForImageClassification

model_name = "microsoft/resnet-18"

processor = AutoImageProcessor.from_pretrained(model_name)

model = ResNetForImageClassification.from_pretrained(
    model_name,
    ignore_mismatched_sizes=True,
    num_labels=6
)

print("ResNet18 모델 불러오기 성공!")
print("출력 클래스 수:", model.config.num_labels)
