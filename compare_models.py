
import os
import numpy as np
import matplotlib.pyplot as plt

# DeepInspect 모델별 검증 결과
models = ["CNN", "ResNet18"]

accuracy = [68.06, 97.78]
macro_f1 = [66.97, 98.00]

x = np.arange(len(models))
width = 0.32

fig, ax = plt.subplots(figsize=(9, 6))

bars1 = ax.bar(
    x - width / 2,
    accuracy,
    width,
    label="Accuracy",
    color="#4472C4"
)

bars2 = ax.bar(
    x + width / 2,
    macro_f1,
    width,
    label="Macro F1-score",
    color="#ED7D31"
)

ax.set_title(
    "DeepInspect: CNN vs ResNet18",
    fontsize=16,
    fontweight="bold"
)

ax.set_ylabel("Score (%)")
ax.set_ylim(0, 110)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis="y", alpha=0.25)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=11
        )

plt.tight_layout()

os.makedirs("results", exist_ok=True)

plt.savefig(
    "results/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("모델 성능 비교 그래프 저장 완료!")
print("저장 위치: results/model_comparison.png")
