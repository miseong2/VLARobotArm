import zmq
import torch
import numpy as np
import time

# LeRobot 모듈 임포트
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

def main():
    print("🤖 [Hardware Server] SO-ARM100 전용 ZMQ 제어 서버 부팅 중...")

    # =========================================================================
    # 1. 하드웨어 연결 및 초기화
    # =========================================================================
    robot_config = SOFollowerRobotConfig(
        port="/dev/ttyACM1", # 현재 사용 중인 포트 확인
        id="my_follower",
        use_degrees=True 
    )

    robot = SOFollower(robot_config)
    robot.connect() 
    print("✅ 로봇 연결 및 토크 ON 완료!")

    # [수정] 'gripper_link' -> 'gripper'로 명칭 통일
    joint_keys = [
        'shoulder_pan', 'shoulder_lift', 'elbow_flex', 
        'wrist_flex', 'wrist_roll', 'gripper'
    ]

    # =========================================================================
    # 2. [안전 로직 1] 관절별 안전 제어 범위(Limits) 동적 계산
    # =========================================================================
    print("\n" + "🔒"*25)
    print("🔒 [관절별 안전 제어 범위 (Safety Limits)]")
    
    safe_limits = {}

    for name in joint_keys:
        calib = robot.calibration[name]
        
        # [수정] 그리퍼 예외 없이 모든 관절에 동일한 Degree 변환 공식 적용
        deg_min = (calib.range_min - 2048 - calib.homing_offset) * (360.0 / 4096.0)
        deg_max = (calib.range_max - 2048 - calib.homing_offset) * (360.0 / 4096.0)
        
        real_min = min(-deg_min, -deg_max)
        real_max = max(-deg_min, -deg_max)
        
        safe_limits[name] = (real_min, real_max)
        print(f" - {name:15}: {real_min:>6.1f} 도  ~  {real_max:>6.1f} 도")
            
    print("🔒"*25 + "\n")

    # =========================================================================
    # 3. [안전 로직 2] 하드웨어 가속도 튜닝
    # =========================================================================
    SLOW_ACCEL = 15 
    print(f"🐌 하드웨어 가속도 제한: {SLOW_ACCEL}")
    robot.bus.configure_motors(maximum_acceleration=SLOW_ACCEL, acceleration=SLOW_ACCEL)
    time.sleep(1)

    # =========================================================================
    # 4. ZMQ 통신 서버 오픈
    # =========================================================================
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    print("🚀 [ZMQ Server] Port 5555에서 명령 대기 중...\n")

    try:
        while True:
            message = socket.recv_json()
            cmd = message.get("cmd")

            if cmd == "get_state":
                obs = robot.get_observation()
                q_deg = [float(obs[f"{k}.pos"]) for k in joint_keys]
                socket.send_json({"status": "ok", "q_deg": q_deg})
                # get_state() 안에
                obs = robot.get_observation()
                gripper_raw = obs["gripper.pos"]
                print(f"[HW] 그리퍼 현재 관측값: {gripper_raw}")  # raw unit인지 degree인지 확인

            elif cmd == "step":
                q_target_deg = message.get("q_target_deg")
                gripper_val = message.get("gripper_val")

                action_dict = {}
                print(f"[HW] Gripper 수신값: {gripper_val:.2f}도 | 범위: {safe_limits['gripper']}")

                for i, name in enumerate(joint_keys):
                    val = gripper_val if name == 'gripper' else q_target_deg[i]
                    
                    # [안전장치] 모든 관절에 대해 물리적 한계 Clipping 적용
                    min_l, max_l = safe_limits[name]
                    safe_val = np.clip(val, min_l, max_l)
                    
                    action_dict[f"{name}.pos"] = torch.tensor(safe_val, dtype=torch.float32)
                    if name == 'gripper':
                        print(f"[HW] Gripper clip 후: {safe_val:.2f}도")
                        
                print(f"[HW] send_action에 보내는 gripper값: {safe_val}")
                robot.send_action(action_dict)
                socket.send_json({"status": "moved"})

    except KeyboardInterrupt:
        print("\n🛑 서버 종료 신호 감지")
    finally:
        if robot.is_connected:
            robot.disconnect()
        socket.close()
        context.term()
        print("🔌 연결 종료 완료")

if __name__ == "__main__":
    main()