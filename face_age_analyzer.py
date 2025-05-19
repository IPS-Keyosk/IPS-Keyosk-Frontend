# face_age_analyzer.py
import os
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import h5py

class FaceAgeAnalyzer:
    def __init__(self):
        # MTCNN 얼굴 검출기 초기화 (이 부분이 누락되어 있었습니다)
        self.face_detector = MTCNN()
        print("[INFO] MTCNN 얼굴 검출기 초기화 완료")
        
        # 절대 경로 사용
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(BASE_DIR, 'models', 'faceage')
        model_path = os.path.join(model_dir, 'faceage_model.h5')
        
        # 디버그 정보 출력
        print(f"[DEBUG] 현재 작업 디렉토리: {os.getcwd()}")
        print(f"[DEBUG] 기본 디렉토리: {BASE_DIR}")
        print(f"[DEBUG] 모델 파일 경로: {model_path}")
        print(f"[DEBUG] 모델 파일 존재 여부: {os.path.exists(model_path)}")
        
        # 노인 판단 기준 나이 (65세 이상을 노인으로 간주)
        self.elderly_age_threshold = 65
        
        try:
            # tf.keras.experimental.load_from_saved_model 사용
            self.age_model = tf.keras.experimental.load_from_saved_model(model_path)
            print("[INFO] experimental.load_from_saved_model로 모델 로드 완료!")
        except Exception as e1:
            print(f"[WARNING] experimental load 실패: {e1}")
            try:
                # SavedModel 형식으로 로드 시도
                self.age_model = tf.saved_model.load(model_path)
                print("[INFO] saved_model.load로 모델 로드 완료!")
            except Exception as e2:
                print(f"[WARNING] saved_model.load 실패: {e2}")
                try:
                    # 일반적인 방법으로 시도
                    self.age_model = tf.keras.models.load_model(model_path, compile=False)
                    print("[INFO] 일반 모델 로드 완료!")
                except Exception as e3:
                    print(f"[ERROR] 모든 로드 방법 실패: {e3}")
                    self.age_model = None
            
    def preprocess_image(self, image):
        """이미지 전처리"""
        # OpenCV 이미지 형식 확인 및 변환 (BGR -> RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        return image_rgb
        
    def extract_face(self, image):
        """이미지에서 얼굴 추출"""
        # 이미지 전처리
        image_rgb = self.preprocess_image(image)
        
        # MTCNN을 사용하여 얼굴 검출
        try:
            faces = self.face_detector.detect_faces(image_rgb)
            print(f"[INFO] 감지된 얼굴 수: {len(faces)}")
        except Exception as e:
            print(f"[ERROR] 얼굴 검출 중 오류: {e}")
            return []
        
        # 감지된 얼굴이 없으면 빈 리스트 반환
        if not faces:
            print("[INFO] 얼굴이 감지되지 않았습니다.")
            return []
            
        # 모든 얼굴에 대한 데이터 추출
        face_images = []
        for face in faces:
            # 얼굴 영역 추출
            x, y, width, height = face['box']
            # 음수 값 방지
            x, y = max(0, x), max(0, y)
            
            # 이미지 경계 확인
            if x >= image_rgb.shape[1] or y >= image_rgb.shape[0]:
                continue
                
            # 얼굴 영역 추출
            face_img = image_rgb[y:y+height, x:x+width]
            
            # 이미지 크기 조정 (160x160 크기로)
            try:
                face_img = cv2.resize(face_img, (160, 160))
                # 정규화 (픽셀 값을 0~1 범위로)
                face_img = face_img / 255.0
                face_images.append({
                    'face': face_img,
                    'box': face['box']
                })
            except Exception as e:
                print(f"[ERROR] 얼굴 이미지 처리 중 오류: {e}")
                continue
                
        return face_images
        
    def predict_age(self, face_img):
        """FaceAge 모델을 사용하여 생물학적 나이 예측"""
        if self.age_model is None:
            return None
            
        try:
            # 배치 차원 추가
            face_batch = np.expand_dims(face_img, axis=0)
            
            # 나이 예측
            predicted_age = self.age_model.predict(face_batch, verbose=0)[0][0]
            return float(predicted_age)
        except Exception as e:
            print(f"[ERROR] 나이 예측 중 오류: {e}")
            return None
    
    def analyze_age(self, image):
        """
        이미지에서 얼굴 감지 및 나이 분석 - FaceAge 모델 기반
        
        Returns:
            annotated_image: 결과가 표시된 이미지
            results: 분석 결과 리스트
        """
        # 얼굴 추출
        face_data_list = self.extract_face(image)
        
        if not face_data_list:
            return image, []  # 얼굴이 감지되지 않은 경우
        
        results = []
        annotated_image = image.copy()
        
        for face_data in face_data_list:
            face_img = face_data['face']
            box = face_data['box']
            
            # FaceAge 모델로 생물학적 나이 예측
            predicted_age = self.predict_age(face_img)
            
            # 나이를 기준으로 노인 여부 판단
            if predicted_age is not None:
                is_elderly = predicted_age >= self.elderly_age_threshold
                
                # 신뢰도 점수 계산 (나이와 임계값의 차이에 기반)
                confidence = min(abs(predicted_age - self.elderly_age_threshold) / 20.0, 0.95)
                if is_elderly:
                    confidence = 0.5 + confidence / 2  # 0.5-0.975 범위
                else:
                    confidence = 0.5 - confidence / 2  # 0.025-0.5 범위
                
                # 나이 그룹 결정
                if predicted_age >= self.elderly_age_threshold:
                    age_group = "노년층"
                elif predicted_age >= 40:
                    age_group = "중년층"
                else:
                    age_group = "청년층"
            else:
                # FaceAge 모델이 없는 경우, 기본값 사용
                is_elderly = False
                confidence = 0.5
                age_group = "알 수 없음"
                predicted_age = None
            
            # 결과 저장
            result = {
                'box': box,
                'is_elderly': is_elderly,
                'age_group': age_group,
                'weighted_score': confidence,
                'predicted_age': predicted_age
            }
            results.append(result)
            
            # 이미지에 결과 표시
            x, y, w, h = box
            color = (0, 0, 255) if is_elderly else (0, 255, 0)
            
            # 라벨 생성
            label = f"{age_group}"
            if predicted_age is not None:
                label += f" ({predicted_age:.1f}세)"
                
            cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(annotated_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_image, results