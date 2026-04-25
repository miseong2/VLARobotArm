import time
import torch
import numpy as np
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

print("🤖 [Sim-to-Real] 정밀 제어 모듈 부팅 중... (Port: /dev/ttyACM0)")

JOINT_KEYS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
HOME_ANGLES = np.array([-3.2, -104.8, 105.8, 78.6, 0.3, 2.3])

try:
    # 1. 로봇 설정 (⭐ [수정됨] 사용자 요청대로 ACM0 사용)
    robot_config = SOFollowerRobotConfig(
        port="/dev/ttyACM0",
        id="my_follower",
        use_degrees=True
    )

    # 2. 로봇 연결
    robot = SOFollower(robot_config)
    robot.connect() 
    print("✅ 로봇팔 연결 성공!")

    # 하드웨어 가속도 최적화
    SLOW_ACCEL = 15
    robot.bus.configure_motors(maximum_acceleration=SLOW_ACCEL, acceleration=SLOW_ACCEL)
    
    # 3. 현재 상태 확인
    obs = robot.get_observation()
    current_angles = np.array([obs[k + '.pos'] for k in JOINT_KEYS])
    print(f"📊 [현재 관절 상태] \n{np.round(current_angles, 1)}")
    
    time.sleep(1)

    # 4. 이동 명령 생성
    target_angles = current_angles.copy()

    # [Step 1] 이동 테스트
    print("\n▶ [Step 1] 동작 중...")
    target_angles[2] -= 30.0  
    robot.send_action({f"{k}.pos": torch.tensor(target_angles[i]) for i, k in enumerate(JOINT_KEYS)})
    time.sleep(2)

    # ---------------------------------------------------------
    # ⭐ [수정됨] [Step 5] 지정된 초기 위치(HOME)로 복귀
    # ---------------------------------------------------------
    print("\n🏠 [Step 5] 초기 위치(HOME)로 복귀 중...")
    robot.send_action({f"{k}.pos": torch.tensor(HOME_ANGLES[i]) for i, k in enumerate(JOINT_KEYS)})
    
    time.sleep(4.0)
    print("✅ 모든 테스트 완료 및 홈 복귀 성공!")

except Exception as e:
    print(f"\n❌ 에러 발생: {e}")
    print("💡 팁: 로봇이 /dev/ttyACM0이 아닌 /dev/ttyACM1에 연결되어 있는지 확인해 보세요.")

finally:
    if 'robot' in locals() and robot.is_connected:
        robot.disconnect()
        print("🔌 안전하게 종료 완료.")
