import os
import sys
import time
import numpy as np
import cv2
from PIL import Image

# -------------------------------------------------------------------
# [중요] 우리가 새로 만든 클라이언트와 컨트롤러 임포트
# -------------------------------------------------------------------
from envs.real_env_client import RealRobotEnvClient
from controllers.ik_ctrl import IKController

# gRPC proto 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
proto_dir = os.path.join(current_dir, "proto")
if proto_dir not in sys.path:
    sys.path.append(proto_dir)

# ===================================================================
# CONFIG
# ===================================================================
USE_REMOTE_AGENT = True
USE_REAL_CAMERA  = True   # 실제 카메라 사용
PRETRAINED_MODE  = False

CAM_TOP_INDEX    = 2
CAM_WRIST_INDEX  = 0

DEFAULT_URDF = "/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
URDF_PATH = os.getenv("ROBOT_URDF", DEFAULT_URDF)

INSTRUCTIONS = [
    "pick up the orange cube",
]
CURRENT_TASK_IDX = 0
# ===================================================================

def main():
    instruction = INSTRUCTIONS[CURRENT_TASK_IDX]

    # 1. 에이전트 초기화 (Octo gRPC)
    if USE_REMOTE_AGENT:
        from agents.remote_agent import RemoteAgent
        agent = RemoteAgent(target_ip="localhost", target_port=50051)
    else:
        from agents.openvla_agent import OpenVLAAgent
        agent = OpenVLAAgent()

    # ------------------------------------------------------------------
    # 2. 하드웨어 통신용 클라이언트 환경 초기화
    # ------------------------------------------------------------------
    print("🔌 하드웨어 서버(ZMQ)에 연결을 시도합니다...")
    env = RealRobotEnvClient(target_ip="localhost", target_port=5555, urdf_path=URDF_PATH)

    # 3. IK 제어기 초기화 (클라이언트 내부의 Shadow 로봇 사용)
    controller = IKController(
        robot=env.shadow_robot,
        ee_link=env.ee_link,
        action_scaling=1.0,
        smoothing_alpha=0.6,
    )

    # ------------------------------------------------------------------
    # 4. [수정됨] 실제 카메라 초기화 (순수 OpenCV 사용)
    # ------------------------------------------------------------------
    print(f"📷 카메라 연결 중: top({CAM_TOP_INDEX}), wrist({CAM_WRIST_INDEX})")
    
    cam_top = cv2.VideoCapture(CAM_TOP_INDEX)
    cam_top.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam_top.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam_top.set(cv2.CAP_PROP_FPS, 30)
    # V4L2 default buffer=4 frames → 30fps 카메라를 10Hz로 read 시 항상 ~133ms stale.
    # BUFFERSIZE=1로 강제하여 latest frame만 큐에 유지.
    cam_top.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cam_wrist = cv2.VideoCapture(CAM_WRIST_INDEX)
    cam_wrist.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam_wrist.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam_wrist.set(cv2.CAP_PROP_FPS, 30)
    cam_wrist.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cam_top.isOpened() or not cam_wrist.isOpened():
        print("🚨 카메라를 열 수 없습니다. 인덱스를 확인해 주세요.")
        sys.exit(1)

    # set 호출이 silently fail할 수 있으므로 실제 적용값 검증 print.
    # 학습 분포는 640x480 @ 30fps 가정 → 다르면 도메인 갭 발생.
    for name, cam in [("cam_top  ", cam_top), ("cam_wrist", cam_wrist)]:
        w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cam.get(cv2.CAP_PROP_FPS)
        buf = cam.get(cv2.CAP_PROP_BUFFERSIZE)
        print(f"📷 {name} actual: {w:.0f}x{h:.0f} @ {fps:.0f}fps, buffersize={buf:.0f}")

    # 에피소드 시작 전에 서버 내부 상태(prev_images, prev_state) 초기화.
    # 동일 서버에 재접속하는 경우 이전 run의 잔재가 첫 추론을 오염시키지 않도록.
    if hasattr(agent, "reset"):
        agent.reset()

    print(f"🚀 실물 로봇 제어 시작 | 태스크: '{instruction}'")
    time.sleep(1) # 카메라 및 통신 안정화 대기

    # ------------------------------------------------------------------
    # 5. 메인 제어 루프
    # ------------------------------------------------------------------
    try:
        for i in range(1000):
            loop_start_time = time.time()
            
            # (1) [수정됨] 실제 카메라 이미지 획득
            ret_top, frame_top = cam_top.read()
            ret_wrist, frame_wrist = cam_wrist.read()
            
            if not ret_top or not ret_wrist:
                print("⚠️ 카메라 프레임 누락")
                continue

            img_primary = Image.fromarray(cv2.cvtColor(frame_top, cv2.COLOR_BGR2RGB))
            img_wrist = Image.fromarray(cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB))

            # (2) 하드웨어 서버에서 실제 관절 상태 동기화
            current_state = env.get_state()
            if current_state is None:
                continue # 통신 타임아웃 시 다음 스텝으로
                
            q_deg = np.rad2deg(current_state['q']).tolist()

            # (3) Octo 추론
            raw_action = agent.predict(
                img_primary,
                instruction,
                wrist_image=img_wrist,
                state=q_deg,
                # state=None if PRETRAINED_MODE else q_deg,
            )
            # hardcoding debugging
            # raw_action = np.array([0.0, -0.01, 0.01, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            # (4) IK 계산 및 실제 로봇 모터로 전송
            q_target, gripper_target = controller.get_joint_targets(raw_action, current_state)
            env.step(q_target, gripper_target) # ZMQ를 통해 모터 제어 명령 전송

            # (5) 터미널 로그 출력 및 카메라 시각화
            latency = (time.time() - loop_start_time) * 1000
            print(f"[Step {i:03d}] ⏱️ {latency:5.1f}ms | Pos: {np.round(raw_action[:3], 3)} | Grip: {raw_action[6]:.2f}")

            vis_top_bgr = cv2.cvtColor(np.array(img_primary), cv2.COLOR_RGB2BGR)
            vis_wrist_bgr = cv2.cvtColor(np.array(img_wrist), cv2.COLOR_RGB2BGR)
            combined_bgr = np.hstack((vis_top_bgr, vis_wrist_bgr))
            
            cv2.imshow("Real Robot Control (Top | Wrist)", combined_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 10Hz 제어 주기 조절
            elapsed = time.time() - loop_start_time
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)

    except KeyboardInterrupt:
        print("\n🛑 키보드 입력으로 실험 강제 종료.")
    except Exception as e:
        print(f"\n🚨 에러 발생: {e}")
    finally:
        # [수정됨] 카메라 자원 해제
        if hasattr(cam_top, 'release'): cam_top.release()
        if hasattr(cam_wrist, 'release'): cam_wrist.release()
        
        env.disconnect()
        cv2.destroyAllWindows()
        print("🧹 리소스 정리 완료.")

if __name__ == "__main__":
    main()