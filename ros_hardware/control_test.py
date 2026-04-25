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
    # 1. 로봇 설정 (ACM1 포트 사용)
    robot_config = SOFollowerRobotConfig(
        port="/dev/ttyACM0",
        id="my_follower",
        use_degrees=True
    )

    # 2. 로봇 연결
    robot = SOFollower(robot_config)
    robot.connect() 
    print("✅ 로봇팔 연결 및 토크 ON 완료!")

    # =========================================================================
    # ⭐ [새로운 기능] 각 관절의 '실제 제어 가능 범위' 완벽 계산 및 출력
    # (에러 없는 깨끗한 버전)
    # =========================================================================
    print("\n" + "🔒"*25)
    print("🔒 [관절별 안전 제어 범위 (Limits)]")
    
    limits_dict = {}

    for name in JOINT_KEYS:
        # [핵심] 아까 학생이 찾아준 진짜 은신처(robot.calibration)에서 값을 꺼냅니다!
        calib = robot.calibration[name]
        
        # 1. 영점 이동 및 각도 변환
        deg_min = (calib.range_min - 2048 - calib.homing_offset) * (360.0 / 4096.0)
        deg_max = (calib.range_max - 2048 - calib.homing_offset) * (360.0 / 4096.0)
        
        # 2. 직관성을 위해 부호 반전 (현재 각도와 맞추기 위함)
        real_min = min(-deg_min, -deg_max)
        real_max = max(-deg_min, -deg_max)
        
        limits_dict[name] = (real_min, real_max)

        if name == 'gripper':
            print(f" - {name:15}: 0.0 % ~ 100.0 %")
        else:
            print(f" - {name:15}: {real_min:>6.1f} 도  ~  {real_max:>6.1f} 도")
            
    print("🔒"*25 + "\n")
    
    time.sleep(1)

    # 3. 현재 관절 정보 가져오기
    obs = robot.get_observation()
    
    # 키 이름에 .pos를 붙여서 값을 빼옵니다. (에러 수정 완료)
    current_angles = np.array([obs[k + '.pos'] for k in JOINT_KEYS])
    
    print("="*50)
    print(f"📊 [현재 관절 상태] \n{np.round(current_angles, 1)}")
    print("="*50)
    
    time.sleep(1)

    # =========================================================================
    # ⭐ [하드웨어 튜닝] LeRobot 공식 함수로 6개 모터의 가속도를 한 방에 낮춤!
    # =========================================================================
    SLOW_ACCEL = 15  # 254가 최대, 15는 아주 부드럽고 느린 가속도입니다.
    
    print(f"\n🐌 하드웨어 가속도를 {SLOW_ACCEL}로 낮춰서 부드럽게 움직이게 세팅합니다...")
    # Follower의 bus 객체에 있는 공식 함수를 호출합니다!
    robot.bus.configure_motors(maximum_acceleration=SLOW_ACCEL, acceleration=SLOW_ACCEL)
    
    # 4. 이동 명령 생성 =================================

    # from the bottom, motor num is <<0 1 2 3 4 5>>
    #                               pan         gripper

    # 0: shoulder pan
    # -115, 0, 115 (+ goes right)

    # 1: shoulder lift
    # -100, -100, 100 (+ goes forward)

    # 2: elbow flex
    # -95, 95, 95 (+ goes forward)

    # 3: wrist flex
    # -100, 80, 100 (+ goes forward)

    # 4: wrist toll
    # 80, 0, 105 (+ goes right, and it rotates in left over -180. so 80 = -260)

    # 5: gripper
    # 5, 5, 95 (+ goes open)
    
    target_angles = current_angles.copy()

    # ---------------------------------------------------------
    # [Step 1] 3번 모터(인덱스 2) -30도
    # ---------------------------------------------------------
    print("\n▶ [Step 1] 이동...")
    target_angles[2] -= 30.0  
    
    # 딕셔너리를 만들 때 계속 'target_angles' 하나만 씁니다.
    action_dict = {f"{name}.pos": torch.tensor(target_angles[i], dtype=torch.float32) for i, name in enumerate(JOINT_KEYS)}
    robot.send_action(action_dict)
    time.sleep(2)

    # ---------------------------------------------------------
    # [Step 2] 2번(인덱스 1) +50도, 4번(인덱스 3) -50도
    # ---------------------------------------------------------
    print("\n▶ [Step 2] 이동...")
    # 복사할 필요 없이, 방금 전 상태에 그대로 각도만 더해줍니다!
    target_angles[1] += 50.0
    target_angles[3] -= 50.0
    
    action_dict = {f"{name}.pos": torch.tensor(target_angles[i], dtype=torch.float32) for i, name in enumerate(JOINT_KEYS)}
    robot.send_action(action_dict)
    time.sleep(2)

    # ---------------------------------------------------------
    # [Step 3] 6번(인덱스 5) +50도
    # ---------------------------------------------------------
    print("\n▶ [Step 3] 이동...")
    # 역시 그대로 누적해서 더해줍니다!
    target_angles[5] += 50.0
    
    action_dict = {f"{name}.pos": torch.tensor(target_angles[i], dtype=torch.float32) for i, name in enumerate(JOINT_KEYS)}
    robot.send_action(action_dict)
    time.sleep(2)

    # ---------------------------------------------------------
    # [Step 4] 6번(인덱스 5) -50도 (원위치)
    # ---------------------------------------------------------
    print("\n▶ [Step 4] 이동...")
    target_angles[5] -= 50.0
    
    action_dict = {f"{name}.pos": torch.tensor(target_angles[i], dtype=torch.float32) for i, name in enumerate(JOINT_KEYS)}
    robot.send_action(action_dict)
    time.sleep(2)
    
    
    
    # 6. 이동 후 최종 상태 확인
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