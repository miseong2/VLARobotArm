import time
import numpy as np
import torch
from envs.real_env import RealRobotEnv
from controllers.ik_ctrl import IKController

def main():
    # 1. 하드웨어 환경 초기화
    try:
        env = RealRobotEnv(port="/dev/ttyACM0")
    except Exception as e:
        print(f"❌ 로봇 연결 실패: {e}")
        return

    # 2. IK 제어기 초기화
    # Shadow 로봇은 GenesisEnv와 동일한 URDF를 쓰므로 IKController를 그대로 사용 가능
    controller = IKController(
        robot=env.shadow_robot, 
        ee_link=env.ee_link, 
        action_scaling=0.05, # 테스트 시에는 작게 시작
        smoothing_alpha=0.5
    )

    print("\n" + "="*50)
    print("🚀 하드웨어 IK 테스트 시작")
    print("명령어: 1(전진), 2(후진), 3(왼쪽), 4(오른쪽), 5(위), 6(아래), 0(종료)")
    print("="*50)

    try:
        while True:
            cmd = input("입력 (0~6): ")
            
            # 하드코딩된 액션 정의 [x, y, z, roll, pitch, yaw, gripper]
            # Octo/OpenVLA의 Raw Action 형태를 모사함
            raw_action = np.zeros(7)
            
            if cmd == '1': raw_action[0] = 0.5   # Forward
            elif cmd == '2': raw_action[0] = -0.5 # Backward
            elif cmd == '3': raw_action[1] = 0.5  # Left
            elif cmd == '4': raw_action[1] = -0.5 # Right
            elif cmd == '5': raw_action[2] = 0.5  # Up
            elif cmd == '6': raw_action[2] = -0.5 # Down
            elif cmd == '0': break
            else: continue

            # 3. 현재 실제 로봇 상태 확인 (Shadow 로봇과 동기화됨)
            current_state = env.get_state()
            print(f"📊 현재 위치: {np.round(current_state['pos'], 3)}")

            # 4. IK 계산 (Shadow 로봇 이용)
            q_target, gripper_target = controller.get_joint_targets(raw_action, current_state)

            # 5. 실제 로봇에게 명령 전송
            print(f"▶ 이동 명령 전송... (Action: {raw_action[:3]})")
            env.step(q_target, gripper_target)
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n테스트를 중단합니다.")
    finally:
        env.disconnect()
        print("🔌 로봇 연결 해제 완료.")

if __name__ == "__main__":
    main()
