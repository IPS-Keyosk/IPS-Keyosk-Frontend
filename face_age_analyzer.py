import cv2
import dlib
import numpy as np
from scipy.spatial import distance

class FaceAgeAnalyzer:
    def __init__(self):
    # 변경할 코드
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")
        self.predictor = dlib.shape_predictor(model_path)
        
        # 얼굴 감지기 추가 (초기화 시 누락되어 있었음)
        self.detector = dlib.get_frontal_face_detector()
        
        # 노인/젊은이 구분을 위한 임계값 설정 (논문 기반) - 미세 조정
        self.thresholds = {
            'AU06': 0.85,  # 볼 올리기 (노인: 높음) - 임계값 소폭 상향
            'AU07': 1.9,   # 눈꺼풀 조이기 (노인: 높음) - 임계값 소폭 상향
            'AU10': 0.75,  # 윗입술 올리기 (노인: 높음) - 임계값 소폭 상향
            'AU12': 0.65,  # 입꼬리 당기기 (노인: 높음) - 임계값 소폭 하향 (젊은 사람의 웃는 경향 반영)
            'AU14': 0.65,  # 볼 오목하게 하기 (노인: 높음) - 임계값 소폭 상향
            'AU45': 1.45,  # 눈 깜빡임 (노인: 낮음, 그래서 이 값보다 작으면 노인) - 임계값 소폭 하향
        }
        
        # 특성별 가중치 (적응형 LASSO 모델에서 선택된 중요 특성에 더 높은 가중치)
        self.weights = {
            'AU06': 1.7,  # 중립 표정에서 중요 - 가중치 상향
            'AU07': 1.2,  # 가중치 소폭 상향
            'AU10': 1.5,  # 여러 감정에서 중요
            'AU12': 0.8,  # 가중치 소폭 하향 (웃는 표정 영향 감소)
            'AU14': 1.4,  # 슬픔, 놀람에서 중요 - 가중치 소폭 상향
            'AU45': 1.7,  # 중립, 슬픔에서 중요 - 가중치 상향
        }
        
        # 종합 점수 임계값 (이 값 이상이면 노인으로 분류)
        self.elderly_threshold = 0.65  # 65%로 소폭 상향
    
    def shape_to_np(self, shape, dtype="int"):
        """dlib 얼굴 랜드마크를 numpy 배열로 변환"""
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    
    def eye_aspect_ratio(self, eye):
        """눈의 가로/세로 비율(EAR) 계산 - AU45(눈 깜빡임) 근사"""
        # 세로 방향 거리
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        # 가로 방향 거리
        C = distance.euclidean(eye[0], eye[3])
        # EAR 계산
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calc_AU06(self, landmarks):
        """AU06 (볼 올리기) 추정 - 눈 끝과 볼 부분 랜드마크 분석"""
        # 오른쪽 눈 끝과 볼 사이 거리 변화
        right_eye_corner = landmarks[36]
        right_cheek = landmarks[31]
        
        # 왼쪽 눈 끝과 볼 사이 거리 변화
        left_eye_corner = landmarks[45]
        left_cheek = landmarks[35]
        
        # 거리 계산 및 정규화 (얼굴 크기에 따라)
        face_width = distance.euclidean(landmarks[0], landmarks[16])
        right_dist = distance.euclidean(right_eye_corner, right_cheek) / face_width
        left_dist = distance.euclidean(left_eye_corner, left_cheek) / face_width
        
        # 볼 올림 정도 추정 (거리가 줄어들면 볼이 올라간 것)
        # 기준 거리에서 현재 거리를 뺀 값 (볼이 올라갈수록 값이 커짐)
        standard_dist = 0.15  # 기준 거리 (조정 필요)
        au06_right = max(0, standard_dist - right_dist) * 10
        au06_left = max(0, standard_dist - left_dist) * 10
        
        return max(au06_right, au06_left)
    
    def calc_AU07(self, landmarks):
        """AU07 (눈꺼풀 조이기) 추정 - 눈썹과 눈 사이의 거리 측정"""
        # 오른쪽 눈과 눈썹 사이 거리
        right_eye_top = landmarks[37]
        right_brow_bottom = landmarks[20]
        
        # 왼쪽 눈과 눈썹 사이 거리
        left_eye_top = landmarks[44]
        left_brow_bottom = landmarks[23]
        
        # 얼굴 크기로 정규화
        face_height = distance.euclidean(landmarks[8], landmarks[27])
        right_dist = distance.euclidean(right_eye_top, right_brow_bottom) / face_height
        left_dist = distance.euclidean(left_eye_top, left_brow_bottom) / face_height
        
        # 눈꺼풀 조임 정도 추정 (거리가 줄어들면 눈꺼풀이 조여진 것)
        standard_dist = 0.05  # 기준 거리 (조정 필요)
        au07_right = max(0, standard_dist - right_dist) * 40
        au07_left = max(0, standard_dist - left_dist) * 40
        
        return max(au07_right, au07_left)
    
    def calc_AU10(self, landmarks):
        """AU10 (윗입술 올리기) 추정 - 코 끝과 윗입술 사이의 거리 측정"""
        nose_tip = landmarks[33]
        upper_lip = landmarks[51]
        
        # 얼굴 크기로 정규화
        face_height = distance.euclidean(landmarks[8], landmarks[27])
        dist = distance.euclidean(nose_tip, upper_lip) / face_height
        
        # 윗입술 올림 정도 추정 (거리가 줄어들면 윗입술이 올라간 것)
        standard_dist = 0.12  # 기준 거리 (조정 필요)
        au10 = max(0, standard_dist - dist) * 15
        
        return au10
    
    def calc_AU12(self, landmarks):
        """AU12 (입꼬리 당기기) 추정 - 입 양쪽 끝의 수평 위치 측정"""
        mouth_left = landmarks[48]
        mouth_right = landmarks[54]
        mouth_center_top = landmarks[51]
        mouth_center_bottom = landmarks[57]
        
        # 입의 기준 너비
        neutral_width = distance.euclidean(landmarks[36], landmarks[45]) * 0.8
        
        # 현재 입 너비
        current_width = distance.euclidean(mouth_left, mouth_right)
        
        # 입 높이 (웃을 때 입이 약간 벌어짐)
        mouth_height = distance.euclidean(mouth_center_top, mouth_center_bottom)
        
        # 웃는 정도 추정 (너비가 늘어나면 웃는 것)
        au12 = max(0, (current_width / neutral_width - 0.9) * 2.5)
        
        return au12
    
    def calc_AU14(self, landmarks):
        """AU14 (볼 오목하게 하기) 추정 - 입 주변 볼 부분의 랜드마크 분석"""
        # 입 양쪽 끝 좌표
        mouth_left = landmarks[48]
        mouth_right = landmarks[54]
        
        # 볼 부분 좌표
        left_cheek = landmarks[31]
        right_cheek = landmarks[35]
        
        # 입과 볼 사이의 거리 변화 측정
        face_width = distance.euclidean(landmarks[0], landmarks[16])
        left_dist = distance.euclidean(mouth_left, left_cheek) / face_width
        right_dist = distance.euclidean(mouth_right, right_cheek) / face_width
        
        # 볼 오목함 추정 (거리가 줄어들면 볼이 오목해진 것)
        standard_dist = 0.2  # 기준 거리 (조정 필요)
        au14_left = max(0, standard_dist - left_dist) * 5
        au14_right = max(0, standard_dist - right_dist) * 5
        
        return max(au14_left, au14_right)
    
    def calc_AU45(self, landmarks):
        """AU45 (눈 깜빡임) 추정 - 눈의 가로/세로 비율 계산"""
        # 왼쪽 눈, 오른쪽 눈 랜드마크
        left_eye = landmarks[42:48]
        right_eye = landmarks[36:42]
        
        # 양쪽 눈의 EAR 계산
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        
        # 두 눈의 평균 EAR
        ear = (left_ear + right_ear) / 2.0
        
        # 눈의 개방 정도를 AU45에 맞게 변환 (값이 클수록 눈이 더 열려있음)
        au45 = ear * 5
        
        return au45
    
    def analyze_age(self, image):
        """이미지에서 얼굴을 감지하고 노인/젊은이 구분 분석 수행"""
        # 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 감지
        faces = self.detector(gray, 0)
        
        result = []
        
        # 각 얼굴에 대해 분석 수행
        for face in faces:
            # 얼굴 랜드마크 추출
            shape = self.predictor(gray, face)
            landmarks = self.shape_to_np(shape)
            
            # AU 값 계산
            au_values = {
                'AU06': self.calc_AU06(landmarks),
                'AU07': self.calc_AU07(landmarks),
                'AU10': self.calc_AU10(landmarks),
                'AU12': self.calc_AU12(landmarks),
                'AU14': self.calc_AU14(landmarks),
                'AU45': self.calc_AU45(landmarks)
            }
            
            # 노인/젊은이 특성 점수 계산
            elderly_scores = {}
            total_score = 0
            total_weight = 0
            
            for au, value in au_values.items():
                weight = self.weights[au]
                total_weight += weight
                
                # AU45는 반대로 (낮을수록 노인 특성)
                if au == 'AU45':
                    if value < self.thresholds[au]:
                        score = 1.0
                    else:
                        score = max(0, 1.0 - (value - self.thresholds[au]) / self.thresholds[au])
                else:
                    # 다른 AU는 높을수록 노인 특성
                    if value > self.thresholds[au]:
                        score = 1.0
                    else:
                        score = value / self.thresholds[au]
                
                elderly_scores[au] = score
                total_score += score * weight
            
            # 가중 평균 점수 계산
            weighted_score = total_score / total_weight
            
            # 노인/젊은이 분류
            is_elderly = weighted_score >= self.elderly_threshold
            
            # 결과 저장
            face_result = {
                'rect': (face.left(), face.top(), face.right(), face.bottom()),
                'au_values': au_values,
                'elderly_scores': elderly_scores,
                'weighted_score': weighted_score,
                'is_elderly': is_elderly,
                'age_group': 'elderly' if is_elderly else 'young',
                'landmarks': landmarks.tolist()
            }
            
            result.append(face_result)
            
            # 결과를 이미지에 시각화 (디버깅용)
            self.visualize_result(image, face_result)
        
        return image, result
    
    def visualize_result(self, image, face_result):
        """분석 결과를 이미지에 시각화"""
        x1, y1, x2, y2 = face_result['rect']
        age_group = face_result['age_group']
        score = face_result['weighted_score']
        
        # 얼굴 주변에 경계 상자 그리기
        if age_group == 'elderly':
            color = (0, 0, 255)  # 노인: 빨간색
        else:
            color = (0, 255, 0)  # 젊은이: 녹색
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 텍스트 표시
        text = f"{age_group.upper()} ({score:.2f})"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 랜드마크 표시 (디버깅용)
        landmarks = np.array(face_result['landmarks'])
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
        
        # AU 값을 이미지에 표시 (디버그 모드)
        y_offset = y2 + 20
        for au, value in face_result['au_values'].items():
            text = f"{au}: {value:.2f}"
            cv2.putText(image, text, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

# Flask 통합을 위한 함수
def analyze_image(image_path):
    """이미지 파일 경로에서 얼굴 연령 분석 수행"""
    analyzer = FaceAgeAnalyzer()
    image = cv2.imread(image_path)
    if image is None:
        return None, "이미지를 불러올 수 없습니다."
    
    result_image, analysis = analyzer.analyze_age(image)
    return result_image, analysis

def analyze_webcam_frame(frame):
    """웹캠 프레임에서 얼굴 연령 분석 수행"""
    analyzer = FaceAgeAnalyzer()
    result_image, analysis = analyzer.analyze_age(frame)
    return result_image, analysis

# 테스트 코드
if __name__ == "__main__":
    # 웹캠에서 실시간 분석 예시
    analyzer = FaceAgeAnalyzer()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result_frame, analysis = analyzer.analyze_age(frame)
        
        # 결과 표시
        cv2.imshow("Face Age Analysis", result_frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()