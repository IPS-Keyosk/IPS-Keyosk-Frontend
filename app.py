from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
import os
import base64
import time
import threading
import numpy as np
import cv2
import sqlite3
from PIL import Image
from io import BytesIO
import traceback
import datetime
from train_face import train_mobilenet_model  # 모델 학습 모듈 import
from detect_face import generate_frames, load_user_model, detect_and_recognize_face  # 얼굴 인식 함수 import
import requests

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 세션 관리를 위한 시크릿 키

def check_user_info(phoneNum):
    res = requests.get(f"http://localhost:8080/api/info?phoneNum={phoneNum}")
    return res.json()

# 사용자의 스탬프 수 가져오기 (data.txt 파일에서)
def get_user_stamps(user_name):
    if not user_name:
        return 0
    
    folder_path = os.path.join('datasets', user_name)
    data_file = os.path.join(folder_path, 'data.txt')
    
    if not os.path.exists(data_file):
        return 0
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        for line in content.split('\n'):
            if line.startswith('Stamps:'):
                try:
                    return int(line.replace('Stamps:', '').strip())
                except ValueError:
                    return 0
    except Exception as e:
        print(f"스탬프 정보 읽기 오류: {e}")
        
    return 0

# 사용자의 스탬프 수 업데이트 (data.txt 파일에서)
def update_user_stamps(user_name, new_stamp_count):
    if not user_name:
        return False
    
    folder_path = os.path.join('datasets', user_name)
    data_file = os.path.join(folder_path, 'data.txt')
    
    if not os.path.exists(data_file):
        return False
    
    try:
        # 기존 데이터 읽기
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 스탬프 정보 업데이트
        found_stamps = False
        updated_lines = []
        
        for line in lines:
            if line.startswith('Stamps:'):
                updated_lines.append(f'Stamps: {new_stamp_count}\n')
                found_stamps = True
            else:
                updated_lines.append(line)
        
        # 스탬프 정보가 없으면 추가
        if not found_stamps:
            updated_lines.append(f'Stamps: {new_stamp_count}\n')
        
        # 업데이트된 데이터 저장
        with open(data_file, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
            
        return True
    except Exception as e:
        print(f"스탬프 정보 업데이트 오류: {e}")
        return False

@app.route('/user')
def user_page():
    user_name = request.args.get('user_name')
    stamp_count = get_user_stamps(user_name)
    return render_template('user.html', user_name=user_name, stamp=stamp_count)

@app.route('/update_stamps', methods=['POST'])
def update_stamps():
    data = request.json
    user_name = data.get('user_name')
    new_stamps = data.get('stamps')

    if not user_name:
        return jsonify({'result': 'error', 'message': '사용자 이름이 필요합니다.'}), 400

    if update_user_stamps(user_name, new_stamps):
        return jsonify({'result': 'success'})
    else:
        return jsonify({'result': 'error', 'message': '스탬프 업데이트 중 오류가 발생했습니다.'}), 500

# 연동 부분 코드
def get_user_by_phone(phone):
    with open('users.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                name, saved_phone, face_path, stamp_count = parts
                if saved_phone == phone:
                    return {
                        "name": name,
                        "phone": saved_phone,
                        "stamp_count": int(stamp_count)
                    }
    return None
# 몰래 캡쳐 관련 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SECRET_CAPTURE_PATH = os.path.join(BASE_DIR, 'images')
secret_capture_counter = 1
current_session_user_folder = None  # 현재 세션의 사용자 폴더
session_capture_completed = False   # 현재 세션 캡쳐 완료 여부
session_active = False              # 세션 활성화 여부

# predict_age.py 실행 상태 관리
age_prediction_status = {
    'is_running': False,
    'completed': False,
    'user_folder': None,
    'results': None,
    'error': None,
    'average_age': None,
    'is_elderly_by_age': False
}

# 몰래 캡쳐 폴더 생성
def ensure_secret_capture_folder():
    if not os.path.exists(SECRET_CAPTURE_PATH):
        os.makedirs(SECRET_CAPTURE_PATH)
        print(f"[INFO] 몰래 캡쳐 폴더 생성: {SECRET_CAPTURE_PATH}")

# 다음 사용자 번호 찾기
def get_next_user_number():
    if not os.path.exists(SECRET_CAPTURE_PATH):
        return 1
    
    existing_folders = [d for d in os.listdir(SECRET_CAPTURE_PATH) 
                       if os.path.isdir(os.path.join(SECRET_CAPTURE_PATH, d)) and d.startswith('user')]
    
    if not existing_folders:
        return 1
    
    numbers = []
    for folder in existing_folders:
        try:
            num = int(folder.replace('user', ''))
            numbers.append(num)
        except ValueError:
            continue
    
    if numbers:
        return max(numbers) + 1
    else:
        return 1

# 현재 사용자가 등록된 사용자인지 확인하는 함수
def is_registered_user():
    """현재 몰래 캡쳐 중인 사용자가 등록된 사용자인지 확인"""
    try:
        # 최근 얼굴 인식 결과를 확인하는 로직
        # 여기서는 간단히 session이나 다른 방법으로 확인
        # 실제로는 최근 recognize_face 호출 결과를 저장해둬야 함
        return getattr(is_registered_user, '_last_recognition_success', False)
    except:
        return False

# predict_age.py 실행 함수 (백그라운드)
def run_predict_age(user_folder):
    global age_prediction_status
    
    try:
        print(f"[AGE] {user_folder}에 대한 나이 예측 시작...")
        age_prediction_status = {
            'is_running': True,
            'completed': False,
            'user_folder': user_folder,
            'results': None,
            'error': None,
            'average_age': None,
            'is_elderly_by_age': False
        }
        
        # predict_age.py 실행
        import subprocess
        result = subprocess.run(
            ["python", "predict_age.py"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=60  # 60초 타임아웃
        )
        
        if result.returncode == 0:
            # 성공적으로 실행됨
            print(f"[AGE] {user_folder} 나이 예측 완료")
            print("[AGE] 예측 결과:\n" + result.stdout)
            
            # ★★★ 얼굴 인식 실패 확인 ★★★
            if "예측 가능한 이미지가 없습니다." in result.stdout:
                print(f"[AGE] {user_folder} 얼굴 인식 실패 - 반응속도 측정으로 전환")
                age_prediction_status.update({
                    'is_running': False,
                    'completed': True,
                    'results': result.stdout,
                    'average_age': None,
                    'is_elderly_by_age': None,  # null로 설정하여 반응속도 측정 필요함을 표시
                    'error': "face_detection_failed"  # 특별한 에러 코드
                })
                return
            
            # ★★★ 평균 나이 추출 로직 강화 ★★★
            average_age = None
            try:
                lines = result.stdout.split('\n')
                for line in lines:
                    if '평균 나이:' in line:
                        # "평균 나이: 67.34" 형태에서 숫자 추출
                        age_str = line.split(':')[1].strip()
                        average_age = float(age_str)
                        print(f"[AGE] 추출된 평균 나이: {average_age}세")
                        break
                
                if average_age is None:
                    print("[AGE] 평균 나이를 찾을 수 없어 기본값 35세로 설정")
                    average_age = 35.0  # 기본값
                    
            except Exception as e:
                print(f"[AGE] 평균 나이 추출 실패: {e}")
                average_age = 35.0  # 기본값
            
            # 65세 이상 여부 판정
            is_elderly_by_age = average_age >= 65.0 if average_age else False
            
            print(f"[AGE] 최종 결과 - 평균 나이: {average_age}세, 노인 여부: {is_elderly_by_age}")
            
            age_prediction_status.update({
                'is_running': False,
                'completed': True,
                'results': result.stdout,
                'average_age': average_age,
                'is_elderly_by_age': is_elderly_by_age,
                'error': None
            })
        else:
            # 실행 중 오류 발생
            print(f"[AGE] 나이 예측 실행 오류: {result.stderr}")
            age_prediction_status.update({
                'is_running': False,
                'completed': True,
                'results': None,
                'average_age': 35.0,  # 기본값
                'is_elderly_by_age': False,
                'error': result.stderr
            })
            
    except subprocess.TimeoutExpired:
        print("[AGE] 나이 예측 시간 초과")
        age_prediction_status.update({
            'is_running': False,
            'completed': True,
            'results': None,
            'average_age': 35.0,  # 기본값
            'is_elderly_by_age': False,
            'error': "시간 초과"
        })
    except Exception as e:
        print(f"[AGE] 나이 예측 중 오류 발생: {e}")
        age_prediction_status.update({
            'is_running': False,
            'completed': True,
            'results': None,
            'average_age': 35.0,  # 기본값
            'is_elderly_by_age': False,
            'error': str(e)
        })

# ★★★ 새로운 세션 초기화 API ★★★
@app.route('/initialize_session', methods=['POST'])
def initialize_session():
    global current_session_user_folder, session_capture_completed, session_active
    
    try:
        # 새로운 세션 시작
        current_session_user_folder = f"user{secret_capture_counter}"
        session_capture_completed = False
        session_active = True
        
        print(f"[SESSION] 새 세션 초기화: {current_session_user_folder}")
        
        return jsonify({
            'success': True,
            'message': '새 세션이 초기화되었습니다.',
            'user_folder': current_session_user_folder
        })
        
    except Exception as e:
        print(f"[SESSION] 세션 초기화 오류: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ★★★ 세션 종료 API ★★★
@app.route('/end_session', methods=['POST'])
def end_session():
    global current_session_user_folder, session_capture_completed, session_active, secret_capture_counter
    
    try:
        if session_active:
            print(f"[SESSION] 세션 종료: {current_session_user_folder}")
            
            # 다음 사용자를 위해 카운터 증가
            secret_capture_counter += 1
            
            # 세션 상태 리셋
            current_session_user_folder = None
            session_capture_completed = False
            session_active = False
            
            print(f"[SESSION] 다음 사용자: user{secret_capture_counter}")
        
        return jsonify({
            'success': True,
            'message': '세션이 종료되었습니다.',
            'next_user': f'user{secret_capture_counter}'
        })
        
    except Exception as e:
        print(f"[SESSION] 세션 종료 오류: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# 몰래 캡쳐 API 엔드포인트 (수정됨)
@app.route('/secret_capture', methods=['POST'])
def secret_capture():
    global current_session_user_folder, session_capture_completed
    
    try:
        # 세션이 활성화되지 않았다면 오류 반환
        if not session_active or current_session_user_folder is None:
            return jsonify({
                'success': False, 
                'message': '활성화된 세션이 없습니다.'
            }), 400
        
        # ★★★ 이미 현재 세션에서 캡쳐가 완료되었다면 무시 ★★★
        if session_capture_completed:
            return jsonify({
                'success': True, 
                'message': '이미 캡쳐가 완료된 세션입니다.',
                'user_folder': current_session_user_folder,
                'image_count': 8
            })
        
        data = request.get_json()
        if not data or 'image' not in data:
            print("[SECRET] 이미지 데이터가 없습니다")
            return jsonify({'success': False, 'message': '이미지 데이터가 없습니다.'}), 400
        
        image_data = data.get('image')
        timestamp = data.get('timestamp', int(time.time() * 1000))
        
        # 몰래 캡쳐 폴더 확인
        ensure_secret_capture_folder()
        
        # 이미지 디코딩
        try:
            header, encoded = image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            print(f"[SECRET] 이미지 디코딩 완료, 크기: {len(image_bytes)} bytes")
        except Exception as e:
            print(f"[SECRET] 몰래 캡쳐 이미지 디코딩 오류: {e}")
            return jsonify({'success': False, 'message': '이미지 형식이 올바르지 않습니다.'}), 400
        
        # 현재 세션 사용자 폴더 사용
        user_folder = current_session_user_folder
        user_folder_path = os.path.join(SECRET_CAPTURE_PATH, user_folder)
        
        if not os.path.exists(user_folder_path):
            os.makedirs(user_folder_path)
            print(f"[SECRET] 새 사용자 폴더 생성: {user_folder}")
        
        # 이미지 파일명 생성 (타임스탬프 기반)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초까지
        filename = f"capture_{current_time}.jpg"
        file_path = os.path.join(user_folder_path, filename)
        
        # 이미지 저장
        with open(file_path, 'wb') as f:
            f.write(image_bytes)
        print(f"[SECRET] 이미지 저장 완료: {file_path}")
        
        # 폴더 내 이미지 개수 확인
        image_count = len([f for f in os.listdir(user_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"[SECRET] {user_folder}에 이미지 저장됨: {filename} (총 {image_count}개)")
        
        # 이미지가 충분히 모이면 (8개) 다음 처리
        if image_count >= 8:
            print(f"[SECRET] {user_folder} 캡쳐 완료.")
            
            # ★★★ 세션 캡쳐 완료 표시 ★★★
            session_capture_completed = True
            
            # 등록된 사용자인지 확인
            if is_registered_user():
                print(f"[SECRET] {user_folder}는 등록된 사용자입니다. predict_age.py를 실행하지 않습니다.")
            else:
                print(f"[SECRET] {user_folder}는 새로운 사용자입니다. predict_age.py를 실행합니다.")
                # 백그라운드에서 predict_age.py 실행
                prediction_thread = threading.Thread(target=run_predict_age, args=(user_folder,))
                prediction_thread.daemon = True
                prediction_thread.start()
        
        return jsonify({
            'success': True, 
            'message': f'이미지가 {user_folder}에 저장되었습니다.',
            'user_folder': user_folder,
            'image_count': image_count
        })
        
    except Exception as e:
        print(f"[SECRET] 몰래 캡쳐 처리 중 오류: {str(e)}")
        traceback.print_exc()  # 상세 오류 추적 추가
        return jsonify({'success': False, 'message': f'저장 중 오류가 발생했습니다: {str(e)}'}), 500
    
    

# predict_age.py 상태 확인 API 수정
@app.route('/check_age_prediction_status', methods=['GET'])
def check_age_prediction_status():
    """나이 예측 상태를 확인하는 API"""
    global age_prediction_status
    
    return jsonify({
        'is_running': age_prediction_status['is_running'],
        'completed': age_prediction_status['completed'],
        'user_folder': age_prediction_status['user_folder'],
        'results': age_prediction_status['results'],
        'error': age_prediction_status['error'],
        'average_age': age_prediction_status['average_age'],  # ★★★ 추가 ★★★
        'is_elderly_by_age': age_prediction_status['is_elderly_by_age']  # ★★★ 추가 ★★★
    })

# 나이 예측 상태 초기화 API 추가
@app.route('/reset_age_prediction_status', methods=['POST'])
def reset_age_prediction_status():
    """나이 예측 상태를 초기화하는 API"""
    global age_prediction_status
    
    age_prediction_status = {
        'is_running': False,
        'completed': False,
        'user_folder': None,
        'results': None,
        'error': None,
        'average_age': None,
        'is_elderly_by_age': False
    }
    
    return jsonify({'success': True, 'message': '나이 예측 상태가 초기화되었습니다.'})

# 몰래 캡쳐된 이미지 관리 API 엔드포인트들 (기존 라우트들 끝부분에 추가)
@app.route('/secret_capture_status', methods=['GET'])
def secret_capture_status():
    """몰래 캡쳐 상태 조회"""
    try:
        ensure_secret_capture_folder()
        
        # 현재 사용자 폴더들 조회
        user_folders = []
        if os.path.exists(SECRET_CAPTURE_PATH):
            folders = [d for d in os.listdir(SECRET_CAPTURE_PATH) 
                      if os.path.isdir(os.path.join(SECRET_CAPTURE_PATH, d)) and d.startswith('user')]
            
            for folder in sorted(folders):
                folder_path = os.path.join(SECRET_CAPTURE_PATH, folder)
                image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                user_folders.append({
                    'folder_name': folder,
                    'image_count': len(image_files),
                    'last_modified': os.path.getmtime(folder_path)
                })
        
        return jsonify({
            'success': True,
            'current_user': current_session_user_folder,
            'session_active': session_active,
            'total_folders': len(user_folders),
            'folders': user_folders
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/reset_secret_capture', methods=['POST'])
def reset_secret_capture():
    """몰래 캡쳐 카운터 리셋"""
    global secret_capture_counter
    
    try:
        secret_capture_counter = get_next_user_number()
        return jsonify({
            'success': True, 
            'message': f'캡쳐 카운터가 user{secret_capture_counter}로 리셋되었습니다.',
            'current_user': f'user{secret_capture_counter}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    
@app.route('/login', methods=['POST'])
def login():
    phone = request.json.get('phone')  # 얼굴 인식 후 받은 전화번호
    
    if not phone:
        return jsonify({
            "success": False,
            "message": "전화번호가 제공되지 않았습니다.",
            "data": None
        }), 400

    user = get_user_by_phone(phone)

    if user:
        return jsonify({
            "success": True,
            "message": "로그인에 성공하였습니다.",
            "data": {
                "name": user["name"],
                "phone": user["phone"],
                "stamp_count": user["stamp_count"]
            }
        })
    else:
        return jsonify({
            "success": False,
            "message": "등록되지 않은 사용자입니다.",
            "data": None
        })

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
    # URL 파라미터에서 사용자 정보와 스탬프 정보 가져오기
    user_name = request.args.get('user_name', '')
    stamp_count = request.args.get('stamp', 0)
    
    # 인덱스 페이지 렌더링, 사용자 정보와 스탬프 수 전달
    return render_template('index.html', user_name=user_name, stamp=stamp_count)

@app.route('/camera')
def camera():
    return render_template('camera.html')


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
            # 새로운 사용자로 처리
            is_registered_user._last_recognition_success = False
            return jsonify({
                'success': False, 
                'message': '등록된 사용자가 없습니다.'
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
            # 새로운 사용자로 처리
            is_registered_user._last_recognition_success = False
            return jsonify({
                'success': False, 
                'message': '인식 모델이 준비되지 않았습니다.'
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
            # 새로운 사용자로 처리
            is_registered_user._last_recognition_success = False
            return jsonify({
                'success': False, 
                'message': '얼굴을 찾을 수 없습니다. 카메라를 정면으로 봐주세요.'
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
                # 등록된 사용자로 표시
                is_registered_user._last_recognition_success = True
                # return jsonify({
                #     'success': True, 
                #     'name': recognized_name, 
                #     'confidence': max_confidence
                # })
                user_folder = os.path.join('datasets', recognized_name)
                data_file = os.path.join(user_folder, 'data.txt')
                user_phone = None
                try:
                    with open(data_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.startswith("Phone:"):
                                user_phone = line.replace("Phone:", "").strip()
                                break
                except Exception as e:
                    print(f"[ERROR] 전화번호 불러오기 실패: {e}")

                return jsonify({
                    'success': True,
                    'name': recognized_name,
                    'confidence': max_confidence,
                    'phone': user_phone  # <- 프론트로 전화번호 전달!
                })
            else:
                print(f"인식된 사용자({recognized_name})가 폴더와 일치하지 않음")
                # 새로운 사용자로 처리
                is_registered_user._last_recognition_success = False
                return jsonify({
                    'success': False, 
                    'message': '인식된 사용자가 폴더와 일치하지 않습니다.'
                }), 200
        else:
            print("얼굴은 감지되었으나 등록된 사용자로 인식되지 않음")
            # 새로운 사용자로 처리
            is_registered_user._last_recognition_success = False
            return jsonify({
                'success': False, 
                'message': '얼굴을 인식했으나 등록된 사용자가 아닙니다.'
            }), 200
            
    except Exception as e:
        print(f"얼굴 인식 처리 중 예외 발생: {str(e)}")
        traceback.print_exc()  # 상세 오류 추적 출력
        # 새로운 사용자로 처리
        is_registered_user._last_recognition_success = False
        return jsonify({
            'success': False, 
            'message': f'서버 오류가 발생했습니다: {str(e)}'
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
    # URL 파라미터에서 사용자 정보 가져오기
    user_name = request.args.get('user_name', '')
    
    # 스탬프 정보 가져오기
    stamp_count = get_user_stamps(user_name)
    
    # 체크아웃 페이지 렌더링, 스탬프 수 전달
    return render_template('checkout.html', stamp=stamp_count, user_name=user_name)


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
    
    # if not name:
    #     return jsonify({'error': 'Name is required'}), 400
    
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

# @app.route('/get_user_info', methods=['POST'])
# def get_user_info():
#     data = request.get_json()
#     user_name = data.get('name')
    
#     if not user_name:
#         return jsonify({'success': False, 'message': '사용자 이름이 필요합니다.'}), 400
    
#     # 사용자 폴더 경로
#     folder_path = os.path.join('datasets', user_name)
#     data_file = os.path.join(folder_path, 'data.txt')
    
#     if not os.path.exists(data_file):
#         return jsonify({'success': False, 'message': '사용자 정보 파일을 찾을 수 없습니다.'}), 404
    
#     # data.txt 파일 읽기
#     try:
#         with open(data_file, 'r') as f:
#             content = f.read()
            
#         # IsElderly 정보 추출
#         is_elderly = False
#         for line in content.split('\n'):
#             if line.startswith('IsElderly:'):
#                 is_elderly = line.replace('IsElderly:', '').strip().lower() == 'true'
#                 break
        
#         return jsonify({
#             'success': True,
#             'name': user_name,
#             'is_elderly': is_elderly
#         })
#     except Exception as e:
#         return jsonify({'success': False, 'message': f'오류가 발생했습니다: {str(e)}'}), 500

@app.route('/get_user_info', methods=['POST'])
def get_user_info():
    data = request.get_json()
    name = data.get('name')  # 사용자 이름으로 폴더 찾기
    user_folder_path = os.path.join("datasets", name)  # 폴더 경로 예시

    data_file_path = os.path.join(user_folder_path, "data.txt")
    if not os.path.exists(data_file_path):
        return jsonify({'success': False, 'message': 'data.txt not found'}), 404

    try:
        with open(data_file_path, 'r', encoding='utf-8') as f:
            content = f.readlines()
        info = {}
        for line in content:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                info[key.strip()] = value.strip()

        return jsonify({
            'success': True,
            'userName': info.get('Name'),
            'phone': info.get('Phone'),
            'isElderly': info.get('IsElderly') == 'True'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

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
    
    # 몰래 캡쳐 폴더 생성 (먼저 실행)
    ensure_secret_capture_folder()
    
    # 몰래 캡쳐 카운터 초기화
    secret_capture_counter = get_next_user_number()
    print(f"[INFO] 몰래 캡쳐 시작 사용자: user{secret_capture_counter}")
    
    # 백그라운드 스레드에서 모든 사용자 모델 학습 시작
    print("[INFO] 애플리케이션 시작 시 자동 모델 학습 시작…")
    training_thread = threading.Thread(target=train_all_users)
    training_thread.daemon = True
    training_thread.start()
    
    app.run(debug=True, port=5001)

