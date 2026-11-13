
import kagglehub
import os

path = kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database")
print("다운로드 완료!")
print("저장 위치:", path)
print("폴더 목록:", os.listdir(path))

for root, dirs, files in os.walk(path):
    print("폴더:", root)
    print("하위 폴더:", dirs)
    print("파일 개수:", len(files))
    print("파일 예시:", files[:3])
    print("--------------------")
