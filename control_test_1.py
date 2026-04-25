import time
import torch
import numpy as np

# 경로 확인 완료
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

print("🤖 [Sim-to-Real] 정밀 제어 모듈 부팅 중...")

# 관절 데이터가 저장된 키 목록
JOINT_KEYS = [
    'shoulder_pan', 
    'shoulder_lift', 
    'elbow_flex', 
    'wrist_flex', 
    'wrist_roll', 
    'gripper'
]

try:
    # 1. 로봇 설정 (확인된 ACM1 포트 사용)
    robot_config = SOFollowerRobotConfig(
        port="/dev/ttyACM1",
        id="my_follower",
        use_degrees=True
    )

    # 2. 로봇 연결
    robot = SOFollower(robot_config)
    robot.connect() 
    print("✅ 로봇팔 연결 및 토크 ON 완료!")

    # 3. 각 관절의 '실제 제어 가능 범위' 출력
    print("\n" + "🔒"*25)
    print("🔒 [관절별 안전 제어 범위 (Limits)]")
    for name in JOINT_KEYS:
        calib = robot.calibration[name]
        deg_min = (calib.range_min - 2048 - calib.homing_offset) * (360.0 / 4096.0)
        deg_max = (calib.range_max - 2048 - calib.homing_offset) * (360.0 / 4096.0)
        real_min, real_max = min(-deg_min, -deg_max), max(-deg_min, -deg_max)
        if name == 'gripper':
            print(f" - {name:15}: 0.0 % ~ 100.0 %")
        else:
            print(f" - {name:15}: {real_min:>6.1f} 도 ~ {real_max:>6.1f} 도")
    print("🔒"*25 + "\n")

    # 4. 현재 관절 정보 가져오기
    obs = robot.get_observation()
    current_angles = np.array([obs[k + '.pos'] for k in JOINT_KEYS])
    print("="*50)
    print(f"📊 [현재 관절 상태] \n{np.round(current_angles, 1)}")
    print("="*50)
    time.sleep(1)

    # 5. 하드웨어 가속도 낮추기 (정밀 제어용)
    SLOW_ACCEL = 15
    print(f"\n🐌 하드웨어 가속도를 {SLOW_ACCEL}로 낮춥니다...")
    robot.bus.configure_motors(maximum_acceleration=SLOW_ACCEL, acceleration=SLOW_ACCEL)
    
    # 6. 이동 명령 시퀀스 시작
    target_angles = current_angles.copy()

    # [Step 1] 3번(인덱스 2) -30도
    print("\n▶ [Step 1] 팔꿈치(2) 이동...")
    target_angles[2] -= 30.0  
    action_dict = {f"{name}.pos": torch.tensor(target_angles[i], dtype=torch.float32) for i, name in enumerate(JOINT_KEYS)}
    robot.send_action(action_dict)
    time.sleep(2)

    # [Step 2] 2번(인덱스 1) +50도, 4번(인덱스 3) -50도
    print("\n▶ [Step 2] 어깨(1)/손목(3) 이동...")
    target_angles[1] += 50.0
    target_angles[3] -= 50.0
    action_dict = {f"{name}.pos": torch.tensor(target_angles[i], dtype=torch.float32) for i, name in enumerate(JOINT_KEYS)}
    robot.send_action(action_dict)
    time.sleep(2)

    # [Step 3] 6번(인덱스 5) +50% (그리퍼 열기)
    print("\n▶ [Step 3] 그리퍼(5) 열기...")
    target_angles[5] += 50.0
    action_dict = {f"{name}.pos": torch.tensor(target_angles[i], dtype=torch.float32) for i, name in enumerate(JOINT_KEYS)}
    robot.send_action(action_dict)
    time.sleep(2)

    # [Step 
    print("\n▶ [Step 3] 그리퍼(5) 열기...")
    target_angles[4] += 30.0
    action_dict = {f"{name}.pos": torch.tensor(target_angles[i], dtype=torch.float32) for i, name in enumerate(JOINT_KEYS)}
    robot.send_action(action_dict)
    time.sleep(2)

    # 7. 최종 상태 확인
    final_obs = robot.get_observation()
    final_angles = np.array([final_obs[k + '.pos'] for k in JOINT_KEYS])
    print(f"✅ [이동 완료 후 각도] \n{np.round(final_angles, 1)} 도")

except Exception as e:
    print(f"\n❌ 제어 중 에러 발생: {e}")

finally:
    if 'robot' in locals() and robot.is_connected:
        print("\n🔌 로봇팔 안전하게 연결 해제 중...")
        robot.disconnect()
        print("✅ 종료 완료. 수고하셨습니다!")