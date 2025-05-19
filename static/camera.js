// DOM 요소
const video = document.getElementById('webcam');
const loadingIndicator = document.getElementById('loading-indicator');
const resultContainer = document.getElementById('result-container');
const resultMessage = document.getElementById('result-message');
const newUserBtn = document.getElementById('new-user-btn');
const existingUserBtn = document.getElementById('existing-user-btn');

// 웹캠 시작
async function startWebcam() {
    try {
        // 카메라 접근 요청
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480,
                facingMode: "user" // 전면 카메라 사용 (셀피 모드)
            }
        });
        
        // 비디오 요소에 스트림 연결
        video.srcObject = stream;
        
        // 비디오가 로드되면 로딩 인디케이터 숨기기
        video.onloadedmetadata = () => {
            // 로딩 인디케이터 숨김
            loadingIndicator.classList.add('hidden');
            
            // 5초 후 얼굴 인식 완료로 가정하고 결과 표시
            setTimeout(showOptions, 5000);
        };
        
    } catch (error) {
        console.error('카메라 접근 에러:', error);
        loadingIndicator.classList.add('hidden');
        resultMessage.textContent = '카메라를 사용할 수 없습니다. 권한을 확인해주세요.';
        resultContainer.classList.remove('hidden');
    }
}

// 얼굴 인식 옵션 표시 (실제로는 인식 로직이 여기에 들어갈 수 있음)
function showOptions() {
    resultMessage.textContent = '얼굴이 확인되었습니다. 회원 유형을 선택해주세요.';
    resultContainer.classList.remove('hidden');
    resultContainer.classList.add('fade-in');
}

// 버튼 이벤트 핸들러
newUserBtn.addEventListener('click', () => {
    // 신규 회원 페이지로 이동
    window.location.href = 'register.html'; // 실제 신규 회원 페이지 URL로 변경
});

existingUserBtn.addEventListener('click', () => {
    // 기존 회원 페이지로 이동
    window.location.href = 'member.html'; // 실제 기존 회원 페이지 URL로 변경
});

// 페이지 로드 시 실행
document.addEventListener('DOMContentLoaded', () => {
    // 웹캠 시작
    startWebcam();
});