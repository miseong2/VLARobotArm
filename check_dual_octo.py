import cv2
import numpy as np
import time
import os
import sys
from PIL import Image

# [주의] 이 클라이언트 파일에는 jax, tensorflow, octo가 절대 임포트되면 안 됩니다!

try:
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
except ImportError:
    print("❌ 에러: LeRobot 라이브러리를 찾을 수 없습니다. (lerobot 가상환경인지 확인하세요)")
    sys.exit(1)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

proto_dir = os.path.join(current_dir, "proto")
if proto_dir not in sys.path:
    sys.path.append(proto_dir)

from agents.remote_agent import RemoteAgent

def main():
    SERVER_IP = "localhost" 
    SERVER_PORT = 50051
    agent = RemoteAgent(target_ip=SERVER_IP, target_port=SERVER_PORT)

    fps, w, h = 30, 640, 480
    cfg_top = OpenCVCameraConfig(index_or_path=0, fps=fps, width=w, height=h)
    cfg_wrist = OpenCVCameraConfig(index_or_path=2, fps=fps, width=w, height=h)

    instruction = "pick up the orange cube"

    print(f"🚀 Octo Dual-Cam 실시간 추론 클라이언트 시작 (연결: {SERVER_IP}:{SERVER_PORT})")
    print(f"명령어: '{instruction}' | 종료: 'q' 누름")
    print("-" * 70)

    step_count = 0 # 추론 횟수 카운터
    # dummy_raw_state = [24.307, -103.428, 96.131, 80.351, -105.098, 10.489]
    try:
        with OpenCVCamera(cfg_top) as cam_top, OpenCVCamera(cfg_wrist) as cam_wrist:
            while True:
                loop_start_time = time.time()

                frame_top = cam_top.read()
                frame_wrist = cam_wrist.read()
                
                if frame_top is None or frame_wrist is None:
                    continue

                img_top_pil = Image.fromarray(frame_top)
                img_wrist_pil = Image.fromarray(frame_wrist)

                start_time = time.time()
                actions = agent.predict(
                    image=img_top_pil, 
                    instruction=instruction, 
                    wrist_image=img_wrist_pil,
                    state = None
                )
                latency = (time.time() - start_time) * 1000
                step_count += 1 # 추론 1회 완료 시 +1

                vis_top = cv2.cvtColor(frame_top, cv2.COLOR_RGB2BGR)
                vis_wrist = cv2.cvtColor(frame_wrist, cv2.COLOR_RGB2BGR)
                combined = np.hstack((vis_top, vis_wrist))

                pos = np.round(actions[:3], 3)
                rot = np.round(actions[3:6], 3)
                grip = actions[6]

                # end='\r' 옵션을 제거하여 매번 새로운 줄에 로그가 찍히도록 변경
                print(f"[Step {step_count:04d}] ⏱️ {latency:5.1f}ms | 📍 Pos: {pos} | 🔄 Rot: {rot} | ✊ Grip: {grip:5.2f}")

                cv2.imshow("Octo Client (Top | Wrist)", combined)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # 3. 10Hz(0.1초) 주기를 맞추기 위한 정밀 대기
                elapsed = time.time() - loop_start_time
                if elapsed < 0.8:
                    time.sleep(0.8 - elapsed)
                
                # [선택] 실제 제어 주기가 얼마나 유지되는지 확인하는 로그
                print(f"⏱️ 실제 주기: {time.time() - loop_start_time:.3f}s")

    except Exception as e:
        print(f"\n🚨 클라이언트 에러 발생: {e}")
    finally:
        cv2.destroyAllWindows()
        print("🔌 테스트 종료.")

if __name__ == "__main__":
    main()