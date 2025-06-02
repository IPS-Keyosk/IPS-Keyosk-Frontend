import os
import numpy as np
from PIL import Image
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from skimage.io import imread

# === 설정 ===
MODEL_PATH = "models/my_faceage_model.h5"
BASE_IMAGE_FOLDER = "images"
model_input_size = (160, 160)

# 가장 최근 생성된 user 폴더 찾기
def get_latest_user_folder(base_path):
    user_folders = [
        os.path.join(base_path, d)
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d)) and d.startswith('user')
    ]
    if not user_folders:
        raise ValueError("user 폴더가 없습니다.")
    return max(user_folders, key=os.path.getctime)

# 얼굴 감지 및 전처리 함수
def get_face_bbox(image_path):
    img = imread(image_path)
    try:
        box = MTCNN().detect_faces(img)[0]['box']
        return img, box
    except:
        print(f" 얼굴 인식 실패: {image_path}")
        return None, None

def preprocess_face(img, box):
    x1, y1, w, h = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + w, y1 + h
    face = img[y1:y2, x1:x2]
    face = Image.fromarray(np.uint8(face)).convert('RGB').resize(model_input_size)
    face = np.asarray(face) / 255.0
    return face

# === 메인 ===
if __name__ == "__main__":
    latest_folder = get_latest_user_folder(BASE_IMAGE_FOLDER)
    print(f" 가장 최근 user 폴더: {latest_folder}")

    model = load_model(MODEL_PATH, compile=False)

    face_inputs = []
    file_names = []

    for fname in os.listdir(latest_folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(latest_folder, fname)
            img, box = get_face_bbox(img_path)
            if img is None:
                continue
            input_face = preprocess_face(img, box)
            face_inputs.append(input_face)
            file_names.append(fname)

    if not face_inputs:
        print(" 예측 가능한 이미지가 없습니다.")
        exit()

    # 배치 예측
    batch_input = np.array(face_inputs).reshape(-1, 160, 160, 3)
    predicted_ages = model.predict(batch_input).flatten()

    for fname, age in zip(file_names, predicted_ages):
        print(f"{fname}: {age:.2f}")

    print("\n 예측 완료")
    print(f" 평균 나이: {np.mean(predicted_ages):.2f}")
    print(f" 표준편차: ±{np.std(predicted_ages):.2f}")
