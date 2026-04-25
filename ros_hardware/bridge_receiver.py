import socket
import json
import torch
import numpy as np
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

def main():
    print("🤖 [Receiver] 도커 로봇 제어 브릿지 가동 중 (IP: 172.17.0.2)")
    
    # 1. 하드웨어 초기화 (Port 0)
    try:
        config = SOFollowerRobotConfig(port="/dev/ttyACM0", id="my_follower", use_degrees=True)
        robot = SOFollower(config)
        robot.connect()
        # 부드러운 움직임을 위해 가속도 설정
        robot.bus.configure_motors(maximum_acceleration=20, acceleration=20)
        print("✅ 로봇팔 하드웨어 연결 성공!")
    except Exception as e:
        print(f"❌ 하드웨어 연결 실패: {e}")
        return

    JOINT_KEYS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

    # 2. UDP 소켓 서버 설정 (포트 9999)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 9999))
    print("📡 포트 9999에서 호스트(서버)로부터의 IK 각도 수신 대기 중...")

    try:
        while True:
            data, addr = sock.recvfrom(2048) # 넉넉하게 수신
            raw_msg = data.decode()

            # ⭐ [추가] 상태 요청 처리 (호스트 동기화용)
            if raw_msg == "GET_STATE":
                try:
                    obs = robot.get_observation()
                    current_q = [float(obs[k + '.pos']) for k in JOINT_KEYS]
                    sock.sendto(json.dumps(current_q).encode(), addr)
                except Exception as e:
                    print(f"⚠️ 상태 읽기 실패: {e}")
                continue

            try:
                # 수신 데이터: [q0, q1, q2, q3, q4, gripper_val] (Degrees)
                q_target = json.loads(raw_msg)
                
                # 하드웨어 명령 전송 (Action Dictionary 생성)
                action_dict = {
                    f"{name}.pos": torch.tensor(q_target[i], dtype=torch.float32) 
                    for i, name in enumerate(JOINT_KEYS)
                }
                
                robot.send_action(action_dict)
                # print(f"📥 수신 및 실행: {np.round(q_target, 1)}")

            except Exception as e:
                print(f"⚠️ 데이터 처리 오류: {e}")

    except KeyboardInterrupt:
        print("\n🛑 브릿지 수동 중단.")
    finally:
        if 'robot' in locals() and robot.is_connected:
            robot.disconnect()
        sock.close()
        print("🔌 브릿지 안전 종료.")

if __name__ == "__main__":
    main()
