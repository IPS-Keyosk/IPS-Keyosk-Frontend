import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
from PIL import ImageFont, ImageDraw, Image
import pickle
import glob

# 시스템 폰트 경로 - 윈도우/리눅스/맥에 맞게 조정 필요
system_font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"  # Mac 기본 폰트
if not os.path.exists(system_font_path):
    # 윈도우용 대체 폰트
    system_font_path = "C:\\Windows\\Fonts\\malgun.ttf"
    if not os.path.exists(system_font_path):
        # 리눅스용 대체 폰트
        system_font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# 얼굴 감지 모델 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 사용자 모델 및 레이블 인코더 로드
# 사용자 모델 및 레이블 인코더 로드
def load_user_model():
    """
    모델 폴더에서 가장 최근 모델과 레이블 인코더를 로드
    """
    try:
        models_dir = "models"
        if not os.path.exists(models_dir):
            print(f"[ERROR] 모델 디렉토리가 존재하지 않습니다: {models_dir}")
            os.makedirs(models_dir, exist_ok=True)
            return None, None
            
        model_files = glob.glob(os.path.join(models_dir, "*_model.h5"))
        
        if not model_files:
            print("[INFO] 학습된 모델을 찾을 수 없습니다. 사용자 등록 후 다시 시도하세요.")
            return None, None
        
        # 가장 최근 모델 파일 찾기
        latest_model = max(model_files, key=os.path.getctime)
        model_name = os.path.basename(latest_model).split('_model.h5')[0]
        encoder_path = os.path.join(models_dir, f"{model_name}_encoder.pkl")
        
        if not os.path.exists(encoder_path):
            print(f"[ERROR] 레이블 인코더를 찾을 수 없습니다: {encoder_path}")
            return None, None
        
        # 모델 및 레이블 인코더 로드
        try:
            model = load_model(latest_model)
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            
            print(f"[INFO] 모델 로드 성공: {latest_model}")
            print(f"[INFO] 클래스: {label_encoder.classes_}")
            return model, label_encoder
        except Exception as e:
            print(f"[ERROR] 모델 로드 실패: {e}")
            return None, None
    except Exception as e:
        print(f"[ERROR] 모델 로드 중 예외 발생: {e}")
        return None, None

# 얼굴 감지 및 인식 함수
# 얼굴 감지 및 인식 함수
def detect_and_recognize_face(frame, model, label_encoder):
    """
    프레임에서 얼굴을 감지하고 인식
    """
    try:
        if model is None or label_encoder is None:
            return [], []
        
        # 프레임 크기 확인 및 조정
        height, width = frame.shape[:2]
        if width > 1280:  # 이미지가 너무 크면 리사이징
            scale = 1280 / width
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            
        # 흑백 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 밝기 정규화 (선택적)
        gray = cv2.equalizeHist(gray)
        
        # 얼굴 감지 - 파라미터 조정으로 민감도 증가
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,  # 5에서 4로 낮춤 (더 많은 얼굴 후보 검출)
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        locs = []
        preds = []
        
        for (x, y, w, h) in faces:
            # 얼굴 영역을 약간 확장 (더 나은 인식을 위해)
            y_offset = int(h * 0.1)
            x_offset = int(w * 0.05)
            
            # 이미지 경계 확인
            y1 = max(0, y - y_offset)
            y2 = min(frame.shape[0], y + h + y_offset)
            x1 = max(0, x - x_offset)
            x2 = min(frame.shape[1], x + w + x_offset)
            
            face_roi = frame[y1:y2, x1:x2]
            
            # 얼굴 영역 확인
            if face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                continue
            
            # MobileNetV2 모델 입력을 위한 전처리
            try:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                face_rgb = cv2.resize(face_rgb, (224, 224))
                face_array = img_to_array(face_rgb)
                face_preproc = preprocess_input(face_array)
                face_preproc = np.expand_dims(face_preproc, axis=0)
                
                # 예측
                prediction = model.predict(face_preproc)
                locs.append((x1, y1, x2, y2))
                preds.append(prediction[0])
            except Exception as e:
                print(f"얼굴 처리 오류: {e}")
                continue
        
        return locs, preds
    except Exception as e:
        print(f"얼굴 감지 및 인식 중 오류: {e}")
        return [], []


def generate_frames():
    """
    실시간 비디오 스트림에서 얼굴 인식 프레임 생성 - 텍스트 표시 없이
    """
    # 모델 및 레이블 인코더 로드
    model, label_encoder = load_user_model()
    
    # 비디오 스트림 시작
    print("[INFO] 카메라 스트림 시작...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)  # 카메라 예열 시간
    
    # 미등록자 카운터 초기화
    unknown_counter = 0
    unknown_threshold = 50  # 연속으로 미등록자를 감지할 횟수
    
    while True:
        # 프레임 읽기
        frame = vs.read()
        if frame is None:
            continue
        
        # 프레임 크기 조정
        frame = imutils.resize(frame, width=640)
        
        # 얼굴 감지 및 인식 (여전히 수행하지만 텍스트 표시 없이)
        if model is not None and label_encoder is not None:
            locs, preds = detect_and_recognize_face(frame, model, label_encoder)
            
            # 감지된 얼굴에 대한 처리 - 텍스트 표시 없이 사각형만 표시
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                
                # 예측 결과에서 클래스 확인
                max_index = np.argmax(pred)
                confidence = pred[max_index]
                
                # 신뢰도 임계값 기반 색상 설정
                if confidence < 0.5:
                    color = (0, 0, 255)  # 빨간색 (미등록자)
                    unknown_counter += 1
                else:
                    color = (0, 255, 0)  # 녹색 (등록자)
                    unknown_counter = 0
                
                # 얼굴 영역만 표시 (텍스트 없이)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        # JPEG 인코딩
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        # 프레임 바이트로 변환
        frame_bytes = jpeg.tobytes()
        
        # 비디오 스트림 형식으로 프레임 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # 비디오 스트림 종료
    vs.stop()