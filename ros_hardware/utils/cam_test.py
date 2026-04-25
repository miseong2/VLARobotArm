import cv2 # 이미지 저장을 위해 OpenCV 라이브러리 추가
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

config = OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480)

with OpenCVCamera(config) as camera:
    # 1. 카메라로부터 프레임 읽기
    frame = camera.read()
    print(f"이미지 캡처 성공! 이미지 크기: {frame.shape}")

    # 2. [중요] 색상 변환 (LeRobot은 RGB를 쓰지만, OpenCV 저장용은 BGR이어야 색이 제대로 나옵니다)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 3. 이미지 파일로 저장하기
    # 파일명은 'test_shot.jpg'로 저장됩니다.
    file_name = "test_shot.jpg"
    cv2.imwrite(file_name, frame_bgr)
    
    print(f"✅ 이미지가 '{file_name}' 파일로 저장되었습니다!")
    print("💡 이제 VS Code 탐색기나 호스트 PC의 공유 폴더에서 확인하세요.")