import cv2
import numpy as np
import time
import os
import sys

# LeRobot 카메라 관련 임포트
try:
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
except ImportError:
    print("❌ 에러: LeRobot 라이브러리를 찾을 수 없습니다. 도커 환경인지 확인하세요.")
    sys.exit(1)

def main():
    # 듀얼 카메라 설정 (제공된 코드 기준: 0-Top, 2-Wrist)
    fps, w, h = 30, 640, 480
    cfg_top = OpenCVCameraConfig(index_or_path=0, fps=fps, width=w, height=h)
    cfg_wrist = OpenCVCameraConfig(index_or_path=2, fps=fps, width=w, height=h)

    print(f"🚀 Dual-Cam 실시간 연결 확인 (VLA 추론 제외)")
    print(f"종료: 영상 창이 선택된 상태에서 'q' 누름")

    try:
        # 카메라 열기
        with OpenCVCamera(cfg_top) as cam_top, OpenCVCamera(cfg_wrist) as cam_wrist:
            print("✅ 카메라 연결 성공. 영상을 표시합니다...")
            while True:
                # (1) 프레임 읽기 (RGB 포맷으로 들어옴)
                frame_top = cam_top.read()
                frame_wrist = cam_wrist.read()
                
                if frame_top is None or frame_wrist is None:
                    continue

                # (2) 시각화 (RGB -> BGR 변환 후 가로로 합침)
                vis_top = cv2.cvtColor(frame_top, cv2.COLOR_RGB2BGR)
                vis_wrist = cv2.cvtColor(frame_wrist, cv2.COLOR_RGB2BGR)
                
                # 영상 두 개를 가로로 붙이기
                combined = np.hstack((vis_top, vis_wrist))

                # 디버그 정보 표시 (해상도 및 안내 문구)
                cv2.putText(combined, "Top Camera (0)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(combined, "Wrist Camera (2)", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # (3) 화면에 띄우기
                cv2.imshow("Dual Camera Check (Top | Wrist)", combined)
                
                # 'q' 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"🚨 에러 발생: {e}")
    finally:
        cv2.destroyAllWindows()
        print("🔌 테스트 종료.")

if __name__ == "__main__":
    main()
