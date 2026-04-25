import time
import torch
import numpy as np
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

def main():
    print("🏠 [Home] 로봇을 초기 자세(All Zero)로 이동합니다...")
    
    # 조인트 키 정의
    JOINT_KEYS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    
    try:
        # 1. 로봇 설정 및 연결 (확인된 ACM1 포트 사용)
        config = SOFollowerRobotConfig(
            port="/dev/ttyACM1", 
            id="my_follower", 
            use_degrees=True
        )
        robot = SOFollower(config)
        robot.connect()
        print("✅ 로봇 연결 성공!")

        # 2. 이동 속도 및 가속도 설정 (안전을 위해 부드럽게 설정)
        # maximum_acceleration: 1~254 (낮을수록 부드러움)
        robot.bus.configure_motors(maximum_acceleration=20, acceleration=20)

        # 3. 홈 자세 정의 (사용자 제공 초기 각도)
        home_angles = [-3.2, -104.8, 105.8, 78.6, 0.3, 2.3] 
        
        print(f"▶ 목표 각도 전송: {home_angles}")
        
        # 4. 명령 전송
        action_dict = {
            f"{name}.pos": torch.tensor(home_angles[i], dtype=torch.float32) 
            for i, name in enumerate(JOINT_KEYS)
        }
        
        robot.send_action(action_dict)
        
        # 이동 완료를 위해 충분히 대기 (4초)
        time.sleep(4.0)
        
        # 현재 상태 최종 확인
        final_obs = robot.get_observation()
        final_q = [final_obs[k + '.pos'] for k in JOINT_KEYS]
        print(f"✅ 홈 위치 도착 완료. 현재 각도: {np.round(final_q, 1)}")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
    finally:
        if 'robot' in locals() and robot.is_connected:
            # 5. 안전하게 연결 해제
            robot.disconnect()
            print("🔌 안전하게 연결 해제 완료.")

if __name__ == "__main__":
    main()
