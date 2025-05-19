# train_face.py - 얼굴 인식 모델 학습 관련 함수 (수정 버전)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam  # legacy 제거
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback  # Callback 클래스 추가
import numpy as np
import os
import pickle
import cv2

# 콜백 클래스 정의
class TrainingProgressCallback(Callback):
    def __init__(self, status_dict):
        super(TrainingProgressCallback, self).__init__()
        self.status_dict = status_dict
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # 현재 에포크 종료 시 진행 상태 업데이트
        total_epochs = self.status_dict.get('total_epochs', 10)
        progress = int(((epoch + 1) / total_epochs) * 100)
        
        # 상태 딕셔너리 업데이트
        self.status_dict['progress'] = progress
        self.status_dict['current_epoch'] = epoch + 1
        self.status_dict['current_accuracy'] = float(logs.get('accuracy', 0))
        self.status_dict['current_val_accuracy'] = float(logs.get('val_accuracy', 0))
        
        print(f"에포크 {epoch+1}/{total_epochs} 완료: 정확도={logs.get('accuracy', 0):.4f}, 진행률={progress}%")

def train_mobilenet_model(folder_name, status_dict=None):
    """
    MobileNetV2를 이용한 얼굴 인식 모델 학습 함수
    
    Args:
        folder_name: 학습할 사용자 폴더 이름
        status_dict: 학습 상태를 저장할 딕셔너리
    
    Returns:
        학습 결과 정보 (딕셔너리)
    """
    print(f"[INFO] {folder_name}에 대한 얼굴 인식 모델 학습 시작...")
    
    # 모델 파일과 레이블 인코더 파일 경로
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_path = os.path.join(models_dir, f"{folder_name}_model.h5")
    label_encoder_path = os.path.join(models_dir, f"{folder_name}_encoder.pkl")
    
    # 이전 모델 파일이 있으면 삭제
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"[INFO] 기존 모델 파일 삭제: {model_path}")
    
    if os.path.exists(label_encoder_path):
        os.remove(label_encoder_path)
        print(f"[INFO] 기존 인코더 파일 삭제: {label_encoder_path}")
    
    # 초기 학습률, 에포크 수, 배치 크기 설정
    INIT_LR = 1e-4
    EPOCHS = 15  # 에포크 수 증가
    BS = 4       # 배치 크기 감소
    
    # 상태 딕셔너리가 제공된 경우 초기화
    if status_dict is not None:
        status_dict['total_epochs'] = EPOCHS
        status_dict['current_epoch'] = 0
        status_dict['progress'] = 0
    
    # 데이터셋 디렉토리 설정
    DIRECTORY = "datasets"
    
    # unknown 클래스 추가를 위한 설정
    CATEGORIES = [folder_name, "unknown"]
    
    # 데이터와 레이블을 초기화
    data = []
    labels = []
    
    # 사용자 이미지 로드
    user_path = os.path.join(DIRECTORY, folder_name)
    image_count = 0
    
    if not os.path.exists(user_path):
        return {"success": False, "error": f"사용자 폴더 {folder_name}을 찾을 수 없습니다."}
    
    for img in os.listdir(user_path):
        if not img.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(user_path, img)
        try:
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            
            data.append(image)
            labels.append(folder_name)
            image_count += 1
        except Exception as e:
            print(f"[WARN] 이미지 로드 실패: {img_path}, 오류: {e}")
    
    print(f"[INFO] {folder_name} 클래스에서 {image_count}개 이미지 로드됨")
    
    # unknown 클래스를 위한 가짜 데이터 생성
    # 원본 이미지 회전, 반전 등으로 "unknown" 클래스용 가짜 이미지 생성
    print("[INFO] unknown 클래스를 위한 가짜 데이터 생성 중...")
    unknown_count = 0
    
    # 기본 haarcascade를 사용하여 랜덤한 얼굴 이미지 생성
    try:
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # 기본 얼굴 데이터를 사용하여 unknown 클래스 생성 (원본 이미지 변형)
        for img in os.listdir(user_path):
            if not img.endswith(('.jpg', '.jpeg', '.png')) or unknown_count >= 20:
                continue
                
            img_path = os.path.join(user_path, img)
            try:
                # 원본 이미지 로드
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # 다양한 변형 적용하여 "unknown" 클래스 생성
                # 1. 좌우 반전
                flipped = cv2.flip(image, 1)
                flipped = cv2.resize(flipped, (224, 224))
                flipped = img_to_array(flipped)
                flipped = preprocess_input(flipped)
                data.append(flipped)
                labels.append("unknown")
                unknown_count += 1
                
                # 2. 흐림 효과
                blurred = cv2.GaussianBlur(image, (15, 15), 0)
                blurred = cv2.resize(blurred, (224, 224))
                blurred = img_to_array(blurred)
                blurred = preprocess_input(blurred)
                data.append(blurred)
                labels.append("unknown")
                unknown_count += 1
                
                # 3. 회전
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, 45, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h))
                rotated = cv2.resize(rotated, (224, 224))
                rotated = img_to_array(rotated)
                rotated = preprocess_input(rotated)
                data.append(rotated)
                labels.append("unknown")
                unknown_count += 1
                
            except Exception as e:
                print(f"[WARN] unknown 이미지 생성 실패: {img_path}, 오류: {e}")
    except Exception as e:
        print(f"[WARN] unknown 클래스 생성 중 오류: {e}")
    
    print(f"[INFO] unknown 클래스에 {unknown_count}개 이미지 생성됨")
    
    # unknown 폴더가 있으면 추가 이미지 로드
    unknown_path = os.path.join(DIRECTORY, "unknown")
    if os.path.exists(unknown_path):
        for img in os.listdir(unknown_path):
            if not img.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(unknown_path, img)
            try:
                image = load_img(img_path, target_size=(224, 224))
                image = img_to_array(image)
                image = preprocess_input(image)
                
                data.append(image)
                labels.append("unknown")
                unknown_count += 1
            except Exception as e:
                print(f"[WARN] 이미지 로드 실패: {img_path}, 오류: {e}")
    
    # 충분한 이미지가 있는지 확인
    total_images = len(data)
    if total_images < 10:
        return {"success": False, "error": "학습에 필요한 충분한 이미지가 없습니다."}
    
    print(f"[INFO] 총 {total_images}개의 이미지 로드 완료 ({folder_name}: {image_count}, unknown: {unknown_count})")
    
    # 레이블 인코딩
    lb = LabelEncoder()
    labels = lb.fit_transform(labels)
    
    # 데이터를 학습 및 테스트 세트로 분할
    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    
    # 데이터가 적을 경우 테스트 세트 비율을 줄임
    test_size = 0.25 if total_images >= 40 else 0.2
    (trainX, testX, trainY, testY) = train_test_split(data, labels, 
                                                     test_size=test_size, 
                                                     stratify=labels, 
                                                     random_state=42)
    
    print(f"[INFO] 학습 세트: {len(trainX)}개, 테스트 세트: {len(testX)}개")
    
    # 더 강력한 데이터 증강 설정
    aug = ImageDataGenerator(
        rotation_range=30,        # 회전 각도 범위 확대
        zoom_range=0.2,           # 확대/축소 범위 확대
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],  # 밝기 조절 추가
        fill_mode="nearest")
    
    # MobileNetV2 모델 로드
    print("[INFO] MobileNetV2 기본 모델 로드 중...")
    baseModel = MobileNetV2(weights="imagenet", include_top=False, 
                           input_tensor=Input(shape=(224, 224, 3)))
    
    # 최상단 모델 구성
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.7)(headModel)  # 드롭아웃 비율 증가
    
    # 출력 레이어 수정 - 클래스 수에 맞게
    num_classes = len(lb.classes_)
    headModel = Dense(num_classes, activation="softmax")(headModel)
    
    # 최상단 모델을 기본 모델 위에 쌓기
    model = Model(inputs=baseModel.input, outputs=headModel)
    
    # 기본 모델의 모든 레이어를 고정하여 학습되지 않도록 설정
    for layer in baseModel.layers:
        layer.trainable = False
    
    # 모델 컴파일
    print("[INFO] 모델 컴파일 중...")
    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="sparse_categorical_crossentropy", 
                 optimizer=opt, 
                 metrics=["accuracy"])
    
    # 콜백 생성
    callbacks = []
    if status_dict is not None:
        progress_callback = TrainingProgressCallback(status_dict)
        callbacks.append(progress_callback)
    
    # 모델 학습
    print("[INFO] 모델 학습 시작...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=max(len(trainX) // BS, 1),
        validation_data=(testX, testY),
        validation_steps=max(len(testX) // BS, 1),
        epochs=EPOCHS,
        callbacks=callbacks)
    
    # 모델 평가
    print("[INFO] 모델 평가 중...")
    predIdxs = model.predict(testX, batch_size=BS)
    predIdxs = np.argmax(predIdxs, axis=1)
    
    # 모델 저장
    print(f"[INFO] 모델 저장 중: {model_path}")
    model.save(model_path)
    
    # LabelEncoder 객체 저장
    print(f"[INFO] 레이블 인코더 저장 중: {label_encoder_path}")
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(lb, f)
    
    # 학습 결과 반환
    val_accuracy = np.max(H.history["val_accuracy"])
    print(f"[INFO] 학습 완료! 최대 검증 정확도: {val_accuracy:.4f}")
    
    return {
        "success": True,
        "model_path": model_path,
        "label_encoder_path": label_encoder_path,
        "accuracy": float(val_accuracy),
        "epochs": EPOCHS,
        "classes": list(lb.classes_)
    }

# 단독 실행 테스트용
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        folder_name = sys.argv[1]
        print(f"폴더 '{folder_name}'에 대한 모델 학습 시작")
        result = train_mobilenet_model(folder_name)
        print(f"학습 결과: {result}")
    else:
        print("사용법: python train_face.py <폴더명>")