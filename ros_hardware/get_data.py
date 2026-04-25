import time
from lerobot.configs.types import RobotType
from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

print("🤖 [Sim-to-Real] LeRobot 핵심 엔진 부팅 중...")

# ==========================================
# 1. 완벽하게 세팅된 카메라 2대 (Top & Wrist) 연결
# ==========================================
config_top = OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480)
config_wrist = OpenCVCameraConfig(index_or_path=2, fps=30, width=640, height=480)
top_cam = OpenCVCamera(config_top)
wrist_cam = OpenCVCamera(config_wrist)

top_cam.connect()
wrist_cam.connect()
print("✅ 카메라(OpenCV) 연결 완료!")

# ==========================================
# 2. LeRobot 공식 클래스로 로봇팔(Follower) 연결
# ==========================================
# 우리가 터미널에서 쳤던 옵션들을 그대로 딕셔너리로 넣어줍니다.
# calibration_dir가 자동으로 잡히므로, 에러 없이 완벽한 0~4095(또는 도 단위) 각도가 나옵니다.
follower = SOFollower(
    port="/dev/ttyACM0",
    id="my_follower",
    use_degrees=False # True로 바꾸면 0~360도 각도로 나옵니다.
)

follower.connect()
print("✅ 로봇팔(SO-Follower) 공식 엔진 연결 완료!")

print("\n" + "="*70)
print("🚀 [완벽한 실시간 데이터 스트림 시작] (Ctrl+C로 종료)")
print("="*70)

try:
    while True:
        # [Step 1: 카메라 영상 추출]
        img_top = top_cam.read_latest()
        img_wrist = wrist_cam.read_latest()
        
        # [Step 2: 로봇팔 관절 각도 추출 (LeRobot 공식 API)]
        # get_state()를 쓰면 캘리브레이션이 적용된 완벽하고 깔끔한 숫자가 딕셔너리로 나옵니다.
        state = follower.get_state()
        
        # 각도 리스트만 뽑아내기 (Present Position)
        # state 구조: {'present_position': [val1, val2...], 'present_velocity': [...]}
        positions = state['present_position']
        
        # [Step 3: 출력 (소수점 1자리까지만 깔끔하게)]
        pos_str = [f"{p:.1f}" for p in positions]
        print(f"\r📸 Top:{img_top.shape} | Wrist:{img_wrist.shape} | 🦾 Joints:{pos_str}", end="   ", flush=True)
        
        time.sleep(0.1) # 10 FPS 출력

except KeyboardInterrupt:
    print("\n\n🛑 데이터 추출을 종료합니다.")

finally:
    top_cam.disconnect()
    wrist_cam.disconnect()
    follower.disconnect()
    print("🔌 모든 장치 안전하게 해제 완료.")