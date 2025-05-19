from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
import os
import base64
import time
import threading
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import traceback
from face_age_analyzer import FaceAgeAnalyzer
import datetime
from train_face import train_mobilenet_model  # 모델 학습 모듈 import
from detect_face import generate_frames, load_user_model, detect_and_recognize_face  # 얼굴 인식 함수 import

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 세션 관리를 위한 시크릿 키

# 훈련 상태를 저장할 전역 변수
training_status = {
    'is_training': False,
    'completed': False,
    'error': None,
    'progress': 0,
    'current_epoch': 0,
    'total_epochs': 10,
    'current_accuracy': 0,
    'current_val_accuracy': 0
}

# 모든 사용자 데이터에 대한 모델 학습 함수
def train_all_users():
    global training_status
    
    if not os.path.exists('datasets'):
        print("[INFO] datasets 폴더가 없습니다. 학습을 건너뜁니다.")
        return
        
    users = [d for d in os.listdir('datasets') if os.path.isdir(os.path.join('datasets', d))]
    
    if not users:
        print("[INFO] 등록된 사용자가 없습니다. 학습을 건너뜁니다.")
        return
        
    print(f"[INFO] {len(users)}명의 사용자에 대한 자동 학습을 시작합니다...")
    
    for user in users:
        folder_path = os.path.join('datasets', user)
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) < 10:
            print(f"[WARNING] {user}의 이미지가 부족합니다 ({len(image_files)}개). 학습을 건너뜁니다.")
            continue
            
        try:
            print(f"[INFO] {user}에 대한 모델 학습 시작...")
            training_status = {
                'is_training': True,
                'completed': False,
                'error': None,
                'progress': 0,
                'current_epoch': 0,
                'total_epochs': 10,
                'current_accuracy': 0,
                'current_val_accuracy': 0
            }
            
            # train_mobilenet_model 함수 호출
            result = train_mobilenet_model(user, training_status)
            
            if result['success']:
                print(f"[INFO] {user}에 대한 모델 학습 완료 (정확도: {result['accuracy']:.4f})")
            else:
                print(f"[ERROR] {user}에 대한 모델 학습 실패: {result.get('error', '알 수 없는 오류')}")
                
        except Exception as e:
            print(f"[ERROR] {user}에 대한 모델 학습 중 오류 발생: {str(e)}")
    
    # 모든 사용자 학습 완료 후 통합 모델 로드
    try:
        model, label_encoder = load_user_model()
        if model is not None and label_encoder is not None:
            print("[INFO] 모든 사용자 모델 학습 및 로드 완료")
        else:
            print("[WARNING] 통합 모델 로드에 실패했습니다.")
    except Exception as e:
        print(f"[ERROR] 통합 모델 로드 중 오류 발생: {str(e)}")
    
    # 학습 상태 초기화
    training_status = {
        'is_training': False,
        'completed': True,
        'error': None,
        'progress': 100,
        'current_epoch': 0,
        'total_epochs': 10
    }
    
    print("[INFO] 자동 학습 프로세스 완료")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

# 전역 변수로 FaceAgeAnalyzer 객체 생성
age_analyzer = FaceAgeAnalyzer()

# 얼굴 연령 분석 API 엔드포인트
@app.route('/analyze_age', methods=['POST'])
def analyze_age():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': '이미지 데이터가 없습니다.'}), 400
        
        image_data = data.get('image')
        
        # 이미지 디코딩
        try:
            header, encoded = image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
        except Exception as e:
            print(f"이미지 디코딩 오류: {e}")
            return jsonify({'success': False, 'message': '이미지 형식이 올바르지 않습니다.'}), 400
        
        # 이미지 처리
        try:
            image = Image.open(BytesIO(image_bytes))
            open_cv_image = np.array(image)
            if open_cv_image.shape[2] == 4:  # RGBA 이미지인 경우
                open_cv_image = open_cv_image[:, :, :3]  # RGB로 변환
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"이미지 처리 오류: {e}")
            return jsonify({'success': False, 'message': '이미지 처리 중 오류가 발생했습니다.'}), 400
        
        # 연령 분석 실행
        _, analysis_result = age_analyzer.analyze_age(open_cv_image)
        
        if not analysis_result or len(analysis_result) == 0:
            return jsonify({'success': False, 'message': '얼굴을 찾을 수 없습니다.'}), 404
        
        # 가장 큰 얼굴에 대한 분석 결과 사용 (여러 얼굴이 있을 경우)
        face_result = analysis_result[0]  # 첫 번째 얼굴 결과 사용
        
        # 터미널에 AU 값 출력
        print("\n--- 얼굴 연령 분석 결과 ---")
        print(f"연령 구분: {'노인' if face_result['is_elderly'] else '일반 사용자'} (신뢰도: {face_result['weighted_score']:.2f})")
        print("액션 유닛(AU) 값:")
        for au, value in face_result['au_values'].items():
            threshold = age_analyzer.thresholds[au]
            if au == 'AU45':
                comparison = "< 노인" if value < threshold else "> 일반 사용자" 
            else:
                comparison = "> 노인" if value > threshold else "< 일반 사용자"
            print(f"  {au}: {value:.2f} ({comparison}, 임계값: {threshold})")
        
        # 결과 반환
        return jsonify({
            'success': True,
            'is_elderly': face_result['is_elderly'],
            'age_group': face_result['age_group'],
            'confidence': face_result['weighted_score'],
            'au_values': face_result['au_values']
        })
            
    except Exception as e:
        print(f"연령 분석 처리 중 예외 발생: {str(e)}")
        return jsonify({'success': False, 'message': f'서버 오류가 발생했습니다: {str(e)}'}), 500

def convert_numpy_types(obj):
    """NumPy 타입을 Python 기본 타입으로 재귀적으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(i) for i in obj)
    else:
        return obj


@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    try:
        print("\n--- 얼굴 인식 API 호출 시작 ---")
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': '이미지 데이터가 없습니다.'}), 400
        
        image_data = data.get('image')
        print("이미지 데이터 수신 완료 (Base64 인코딩)")
        
        # 이미지 디코딩
        try:
            header, encoded = image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            print("이미지 디코딩 완료")
        except Exception as e:
            print(f"이미지 디코딩 오류: {e}")
            return jsonify({'success': False, 'message': '이미지 형식이 올바르지 않습니다.'}), 400
        
        # 이미지 처리
        try:
            image = Image.open(BytesIO(image_bytes))
            open_cv_image = np.array(image)
            
            # 이미지 형식 검사 및 변환
            if len(open_cv_image.shape) < 3:
                # 그레이스케일 이미지를 RGB로 변환
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2BGR)
                print("그레이스케일 이미지를 BGR로 변환")
            elif open_cv_image.shape[2] == 4:
                # RGBA 이미지를 RGB로 변환
                open_cv_image = open_cv_image[:, :, :3]
                print("RGBA 이미지를 RGB로 변환")
                
            # RGB를 BGR로 변환 (OpenCV 요구사항)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            print(f"이미지 처리 완료: 크기 {open_cv_image.shape}")
        except Exception as e:
            print(f"이미지 처리 오류: {e}")
            return jsonify({'success': False, 'message': '이미지 처리 중 오류가 발생했습니다.'}), 400
        
        # try-except 블록 안에 연령 분석 코드 감싸기
        try:
            print("연령 분석 시작...")
            _, age_analysis = age_analyzer.analyze_age(open_cv_image.copy())
            print("연령 분석 완료")
        except Exception as e:
            print(f"연령 분석 중 오류 발생: {e}")
            traceback.print_exc()
            # 연령 분석 실패 시에도 계속 진행 (빈 결과로)
            age_analysis = []

        # 연령 분석 결과 초기화
        is_elderly = False
        age_confidence = 0.0
        au_values = {}
        au_explanations = {}  # AU 값 설명 추가

        if age_analysis and len(age_analysis) > 0:
            try:
                # NumPy 타입을 Python 기본 타입으로 변환
                face_result = convert_numpy_types(age_analysis[0])
                is_elderly = face_result['is_elderly']
                age_confidence = face_result['weighted_score']
                au_values = face_result['au_values'] 
                
                # AU 값 설명 생성
                au_explanations = {}
                for au, value in au_values.items():
                    threshold = age_analyzer.thresholds[au]
                    if au == 'AU45':
                        comparison = "< 노인" if value < threshold else "> 일반 사용자" 
                        explanation = "눈 깜빡임 (낮을수록 노인 특성)"
                    else:
                        comparison = "> 노인" if value > threshold else "< 일반 사용자"
                        if au == 'AU06':
                            explanation = "볼 올리기 (높을수록 노인 특성)"
                        elif au == 'AU07':
                            explanation = "눈꺼풀 조이기 (높을수록 노인 특성)"
                        elif au == 'AU10':
                            explanation = "윗입술 올리기 (높을수록 노인 특성)"
                        elif au == 'AU12':
                            explanation = "입꼬리 당기기 (높을수록 노인 특성)"
                        elif au == 'AU14':
                            explanation = "볼 오목하게 하기 (높을수록 노인 특성)"
                        else:
                            explanation = "기타 특성"
                    
                    au_explanations[au] = {
                        "value": value,
                        "threshold": threshold,
                        "comparison": comparison,
                        "explanation": explanation
                    }
                
                # 터미널에 AU 값 출력
                print("\n--- 얼굴 연령 분석 결과 ---")
                print(f"연령 구분: {'노인' if is_elderly else '일반 사용자'} (신뢰도: {age_confidence:.2f})")
                print("액션 유닛(AU) 값:")
                for au, value in au_values.items():
                    threshold = age_analyzer.thresholds[au]
                    if au == 'AU45':
                        comparison = "< 노인" if value < threshold else "> 일반 사용자" 
                    else:
                        comparison = "> 노인" if value > threshold else "< 일반 사용자"
                    print(f"  {au}: {value:.2f} ({comparison}, 임계값: {threshold})")
            except Exception as e:
                print(f"연령 분석 결과 처리 중 오류: {e}")
                traceback.print_exc()
        else:
            print("얼굴 연령 분석 결과가 없습니다 (얼굴을 찾지 못함)")
        
        # 사용자 폴더 확인
        users_folders = []
        try:
            if os.path.exists('datasets'):
                users_folders = [d for d in os.listdir('datasets') if os.path.isdir(os.path.join('datasets', d))]
                print(f"등록된 사용자 수: {len(users_folders)}")
            else:
                print("datasets 폴더가 존재하지 않습니다")
        except Exception as e:
            print(f"사용자 폴더 확인 중 오류: {e}")
        
        # 등록된 사용자가 없는 경우
        if not users_folders:
            print("등록된 사용자가 없습니다")
            return jsonify({
                'success': False, 
                'message': '등록된 사용자가 없습니다.',
                'is_elderly': is_elderly,
                'age_confidence': age_confidence,
                'au_values': au_values,
                'au_explanations': au_explanations  # AU 설명 추가
            }), 200
        
        # 모델 및 레이블 인코더 로드
        try:
            print("얼굴 인식 모델 로드 시작...")
            model, label_encoder = load_user_model()
            if model is not None and label_encoder is not None:
                print("모델 로드 성공")
            else:
                print("모델 로드 실패: 모델 또는 레이블 인코더가 None입니다")
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            traceback.print_exc()
            model, label_encoder = None, None
        
        # 모델 로드 실패 시
        if model is None or label_encoder is None:
            return jsonify({
                'success': False, 
                'message': '인식 모델이 준비되지 않았습니다.',
                'is_elderly': is_elderly,
                'age_confidence': age_confidence,
                'au_values': au_values,
                'au_explanations': au_explanations  # AU 설명 추가
            }), 200
        
        # 얼굴 감지 및 인식 (try-except로 감싸기)
        try:
            print("얼굴 감지 및 인식 시작...")
            locs, preds = detect_and_recognize_face(open_cv_image, model, label_encoder)
            print(f"얼굴 감지 결과: {len(locs) if locs else 0}개의 얼굴 감지됨")
        except Exception as e:
            print(f"얼굴 감지 중 오류 발생: {e}")
            traceback.print_exc()
            locs, preds = None, None
        
        # 얼굴을 찾지 못한 경우
        if not locs or not preds:
            print("얼굴이 감지되지 않았거나 예측 값이 없습니다")
            return jsonify({
                'success': False, 
                'message': '얼굴을 찾을 수 없습니다. 카메라를 정면으로 봐주세요.',
                'is_elderly': is_elderly,
                'age_confidence': age_confidence,
                'au_values': au_values,
                'au_explanations': au_explanations  # AU 설명 추가
            }), 200
        
        # 예측 결과 처리 (try-except로 감싸기)
        try:
            # 가장 신뢰도가 높은 얼굴 선택
            max_confidence = 0
            recognized_name = None
            
            for pred in preds:
                max_index = np.argmax(pred)
                confidence = float(pred[max_index])  # NumPy float -> Python float
                
                # 임계값을 0.5에서 0.7로 높임 (더 엄격한 인식 기준)
                if confidence > 0.7 and confidence > max_confidence:  # 신뢰도 임계값 수정
                    max_confidence = confidence
                    recognized_name = label_encoder.classes_[max_index]
            
            print(f"인식 결과: {recognized_name}, 신뢰도: {max_confidence:.2f}")
        except Exception as e:
            print(f"얼굴 인식 결과 처리 중 오류: {e}")
            traceback.print_exc()
            recognized_name = None
            max_confidence = 0
        
        if recognized_name:
            # 등록된 사용자 이름과 일치하는지 확인
            if recognized_name in users_folders:
                print(f"등록된 사용자 인식 성공: {recognized_name}")
                return jsonify({
                    'success': True, 
                    'name': recognized_name, 
                    'confidence': max_confidence,
                    'is_elderly': is_elderly,
                    'age_confidence': age_confidence,
                    'au_values': au_values,
                    'au_explanations': au_explanations  # AU 설명 추가
                })
            else:
                print(f"인식된 사용자({recognized_name})가 폴더와 일치하지 않음")
                return jsonify({
                    'success': False, 
                    'message': '인식된 사용자가 폴더와 일치하지 않습니다.',
                    'is_elderly': is_elderly,
                    'age_confidence': age_confidence,
                    'au_values': au_values,
                    'au_explanations': au_explanations  # AU 설명 추가
                }), 200
        else:
            print("얼굴은 감지되었으나 등록된 사용자로 인식되지 않음")
            return jsonify({
                'success': False, 
                'message': '얼굴을 인식했으나 등록된 사용자가 아닙니다.',
                'is_elderly': is_elderly,
                'age_confidence': age_confidence,
                'au_values': au_values,
                'au_explanations': au_explanations  # AU 설명 추가
            }), 200
            
    except Exception as e:
        print(f"얼굴 인식 처리 중 예외 발생: {str(e)}")
        traceback.print_exc()  # 상세 오류 추적 출력
        return jsonify({
            'success': False, 
            'message': f'서버 오류가 발생했습니다: {str(e)}',
            'is_elderly': False,
            'age_confidence': 0.0,
            'au_values': {},
            'au_explanations': {}  # 빈 AU 설명 추가
        }), 500
    
# 실시간 얼굴 인식 비디오 스트림 라우트 추가
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/signup-check')
def signup_check():
    return render_template('signup-check.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

# signup-register 라우트 - 하나로 통합
@app.route('/signup-register')
def signup_register():
    # 사용자 이름 가져오기 (URL 파라미터에서)
    folder_name = request.args.get('name', '')
    return render_template('signup-register.html', folder_name=folder_name)

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/menu-existing')
def menu_existing():
    return render_template('menu-existing.html')

@app.route('/menu-new')
def menu_new():
    # URL 파라미터에서 사용자 이름 가져오기
    user_name = request.args.get('user_name', '')
    return render_template('menu-new.html', user_name=user_name)

@app.route('/menu-simple')
def menu_simple():
    return render_template('menu-simple.html')
@app.route('/menu-simple-new')
def menu_simple_new():
    return render_template('menu-simple-new.html')

@app.route('/checkout')
def checkout():
    return render_template('checkout.html')

@app.route('/checkout-new')
def checkout_new():
    return render_template('checkout-new.html')


# 사용자 목록 가져오기 라우트 추가
@app.route('/get_users', methods=['GET'])
def get_users():
    # datasets 폴더에서 사용자 목록 가져오기
    if not os.path.exists('datasets'):
        return jsonify({'users': []})
    
    users = [d for d in os.listdir('datasets') if os.path.isdir(os.path.join('datasets', d))]
    return jsonify({'users': users})

@app.route('/create-folder', methods=['POST'])
def create_folder():
    data = request.get_json()
    name = data.get('name')
    phone = data.get('phone')
    is_elderly = data.get('is_elderly', False)  # 노인 여부 정보 추가
    
    if not name:
        return jsonify({'error': 'Name is required'}), 400
    
    # 폴더 생성 경로 (datasets/{name})
    folder_path = os.path.join('datasets', name)
    
    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # 폴더에 데이터 저장 (data.txt)
    with open(os.path.join(folder_path, 'data.txt'), 'w') as f:
        f.write(f"Name: {name}\nPhone: {phone}\nIsElderly: {is_elderly}")
    
    return jsonify({'message': 'Folder created and data saved successfully'})

# 이미지 업로드 라우트
@app.route('/upload_image', methods=['POST'])
def upload_image():
    data = request.get_json()
    image_data = data.get('image')
    folder_name = data.get('folder')
    
    if not image_data or not folder_name:
        return jsonify({'error': '이미지 데이터 또는 폴더 이름이 없습니다.'}), 400
    
    # 폴더 경로 설정
    folder_path = os.path.join('datasets', folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # 이미지 디코딩
    header, encoded = image_data.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    
    # 이미지 처리
    image = Image.open(BytesIO(image_bytes))
    open_cv_image = np.array(image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    # 얼굴 인식
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # 얼굴 부분만 저장
    face_count = 0
    for (x, y, w, h) in faces:
        face = open_cv_image[y:y+h, x:x+w]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = os.path.join(folder_path, f'face_{timestamp}.jpg')
        cv2.imwrite(save_path, face)
        face_count += 1
    
    if face_count > 0:
        return jsonify({'success': True, 'message': f'{face_count}개의 얼굴이 {folder_name} 폴더에 저장되었습니다.'})
    else:
        # 얼굴이 감지되지 않은 경우 원본 이미지 저장 (선택 사항)
        image_filename = os.path.join(folder_path, f'image_{len(os.listdir(folder_path)) + 1}.jpg')
        with open(image_filename, 'wb') as f:
            f.write(image_bytes)
        return jsonify({'success': False, 'message': '얼굴이 감지되지 않았습니다. 원본 이미지를 저장했습니다.'})

# 모델 학습 함수 (백그라운드 스레드에서 실행)
def train_model_task(folder_name):
    global training_status
    
    try:
        training_status['is_training'] = True
        training_status['completed'] = False
        training_status['error'] = None
        training_status['progress'] = 0
        
        # 모바일넷 모델 학습 함수 호출 (train_face.py 모듈에서 가져옴)
        result = train_mobilenet_model(folder_name, training_status)
        
        if result['success']:
            # 학습 성공 시 상태 업데이트
            training_status['completed'] = True
            training_status['model_path'] = result['model_path']
            training_status['label_encoder_path'] = result['label_encoder_path']
            training_status['accuracy'] = result['accuracy']
            training_status['progress'] = 100  # 100% 완료
            print(f"[INFO] {folder_name}에 대한 모델 학습 완료 (정확도: {result['accuracy']:.4f})")
        else:
            # 학습 실패 시 오류 상태 업데이트
            training_status['error'] = result.get('error', '알 수 없는 오류가 발생했습니다.')
            print(f"[ERROR] 모델 학습 실패: {training_status['error']}")
            
    except Exception as e:
        # 오류 발생 시 상태 업데이트
        training_status['error'] = str(e)
        print(f"[ERROR] 모델 학습 오류: {str(e)}")
    
    finally:
        training_status['is_training'] = False

# 모델 학습 시작 라우트
@app.route('/train_model', methods=['POST'])
def train_model():
    global training_status
    
    # 이미 학습 중이면 오류 반환
    if training_status['is_training']:
        return jsonify({'error': '이미 모델 학습이 진행 중입니다.'}), 400
    
    # 세션에서 현재 사용자 폴더 이름 가져오기 또는 요청 데이터에서 가져오기
    folder_name = request.json.get('folder') or request.args.get('folder') or session.get('current_user')
    
    if not folder_name:
        return jsonify({'error': '사용자 정보를 찾을 수 없습니다.'}), 400
    
    # 사용자 데이터 폴더 확인
    folder_path = os.path.join('datasets', folder_name)
    if not os.path.exists(folder_path):
        return jsonify({'error': '사용자 데이터 폴더를 찾을 수 없습니다.'}), 404
    
    # 이미지 파일 개수 확인
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) < 10:  # 최소 이미지 개수 확인
        return jsonify({'error': '학습을 위한 이미지가 충분하지 않습니다.'}), 400
    
    # 학습 상태 초기화
    training_status = {
        'is_training': True,
        'completed': False,
        'error': None,
        'progress': 0,
        'current_epoch': 0,
        'total_epochs': 10,
        'current_accuracy': 0,
        'current_val_accuracy': 0
    }
    
    # 백그라운드 스레드에서 모델 학습 실행
    training_thread = threading.Thread(target=train_model_task, args=(folder_name,))
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({'message': f'{folder_name}에 대한 모델 학습이 시작되었습니다.'})

# 학습 상태 확인 라우트
@app.route('/training_status', methods=['POST'])
def check_training_status():
    global training_status
    
    # 오류가 있으면 오류 반환
    if training_status['error']:
        return jsonify({
            'status': 'error',
            'error': training_status['error']
        }), 500
    
    # 학습 완료 여부에 따라 상태 반환
    if training_status['completed']:
        return jsonify({
            'status': 'completed',
            'accuracy': training_status.get('accuracy', 0),
            'model_path': training_status.get('model_path', '')
        })
    elif training_status['is_training']:
        return jsonify({
            'status': 'in_progress',
            'progress': training_status.get('progress', 0),
            'current_epoch': training_status.get('current_epoch', 0),
            'total_epochs': training_status.get('total_epochs', 10),
            'current_accuracy': training_status.get('current_accuracy', 0)
        })
    else:
        return jsonify({'status': 'not_started'})

# 학습 완료 확인 및 다음 페이지로 리다이렉트
@app.route('/confirm_training_complete', methods=['POST'])
def confirm_training_complete():
    global training_status
    
    # 학습이 완료되지 않았으면 오류 반환
    if not training_status['completed']:
        return jsonify({'error': '모델 학습이 아직 완료되지 않았습니다.'}), 400
    
    # 사용자 이름 가져오기
    folder_name = request.json.get('folder') or request.args.get('folder') or session.get('current_user')
    
    # 학습 상태 초기화
    training_status = {
        'is_training': False,
        'completed': False,
        'error': None,
        'progress': 0,
        'current_epoch': 0,
        'total_epochs': 10
    }
    
    # 사용자 이름과 함께 메뉴 페이지로 리다이렉트
    return redirect(url_for('menu_new', user_name=folder_name))


# 디지털 약자용 결제 페이지 라우트 추가
@app.route('/checkout-simple')
def checkout_simple():
    return render_template('checkout-simple.html')

@app.route('/get_user_info', methods=['POST'])
def get_user_info():
    data = request.get_json()
    user_name = data.get('name')
    
    if not user_name:
        return jsonify({'success': False, 'message': '사용자 이름이 필요합니다.'}), 400
    
    # 사용자 폴더 경로
    folder_path = os.path.join('datasets', user_name)
    data_file = os.path.join(folder_path, 'data.txt')
    
    if not os.path.exists(data_file):
        return jsonify({'success': False, 'message': '사용자 정보 파일을 찾을 수 없습니다.'}), 404
    
    # data.txt 파일 읽기
    try:
        with open(data_file, 'r') as f:
            content = f.read()
            
        # IsElderly 정보 추출
        is_elderly = False
        for line in content.split('\n'):
            if line.startswith('IsElderly:'):
                is_elderly = line.replace('IsElderly:', '').strip().lower() == 'true'
                break
        
        return jsonify({
            'success': True,
            'name': user_name,
            'is_elderly': is_elderly
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'오류가 발생했습니다: {str(e)}'}), 500

@app.route('/checkout-simple-new')
def checkout_simple_new():
    return render_template('checkout-simple-new.html')

if __name__ == '__main__':
    # models 폴더 생성 (없는 경우)
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # datasets 폴더 생성 (없는 경우)
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    
    # 백그라운드 스레드에서 모든 사용자 모델 학습 시작
    print("[INFO] 애플리케이션 시작 시 자동 모델 학습 시작…")
    training_thread = threading.Thread(target=train_all_users)
    training_thread.daemon = True
    training_thread.start()

    
    app.run(debug=True, port=5001)

