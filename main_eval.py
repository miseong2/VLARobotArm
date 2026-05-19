import os
import sys
import time
import numpy as np
import cv2
from PIL import Image

# gRPC proto 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
proto_dir = os.path.join(current_dir, "proto")
if proto_dir not in sys.path:
    sys.path.append(proto_dir)

# ===================================================================
# CONFIG
# ===================================================================
USE_REMOTE_AGENT = True   # True: Octo (gRPC), False: OpenVLA (Local)
USE_REAL_CAMERA  = False  # True: 실제 카메라, False: Genesis 렌더 이미지

# 사전학습(pretrained) 모델 테스트 여부
#   True  → action_scaling=1.0, state 미전달 (Bridge 스케일 그대로 사용)
#   False → action_scaling=0.005, state 전달 (파인튜닝 후 사용)
PRETRAINED_MODE = False

CAM_TOP_INDEX   = 0       # top 카메라 인덱스   → Octo primary
CAM_WRIST_INDEX = 2       # wrist 카메라 인덱스 → Octo wrist

DEFAULT_URDF = "/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
URDF_PATH = os.getenv("ROBOT_URDF", DEFAULT_URDF)

INSTRUCTIONS = [
    "move above the orange cube",  # main_real2와 일치
    "put inside the container",    # put_inside 태스크
]
CURRENT_TASK_IDX = 0

# main_real2와 일치: action chunking 사용. Octo 학습 시 action_horizon=4.
ACTION_DIM    = 7
ACTION_CHUNK  = 4

TARGET_LATENCY = 0.0   # 논문용 인위적 지연(초). 일반 테스트 시 0.0
# ===================================================================


def open_cameras():
    """LeRobot OpenCVCamera로 top/wrist 카메라를 열고 반환합니다."""
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cfg_top   = OpenCVCameraConfig(index_or_path=CAM_TOP_INDEX,   fps=30, width=640, height=480)
    cfg_wrist = OpenCVCameraConfig(index_or_path=CAM_WRIST_INDEX, fps=30, width=640, height=480)

    cam_top   = OpenCVCamera(cfg_top)
    cam_wrist = OpenCVCamera(cfg_wrist)
    cam_top.connect()
    cam_wrist.connect()
    return cam_top, cam_wrist


def show_dual_cam(frame_top, frame_wrist, raw_action, step):
    """top/wrist 카메라 피드와 추론 결과를 한 창에 표시합니다."""
    vis_top   = cv2.cvtColor(frame_top,   cv2.COLOR_RGB2BGR)
    vis_wrist = cv2.cvtColor(frame_wrist, cv2.COLOR_RGB2BGR)
    combined  = np.hstack((vis_top, vis_wrist))

    overlays = [
        f"Step {step:03d}",
        f"Pos dxyz: {np.round(raw_action[:3], 3)}",
        f"Rot dxyz: {np.round(raw_action[3:6], 3)}",
        f"Gripper:  {raw_action[6]:.3f}",
    ]
    for i, text in enumerate(overlays):
        cv2.putText(combined, text, (10, 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    cv2.imshow("Real Cam [Top | Wrist]", combined)
    return cv2.waitKey(1) & 0xFF == ord('q')


def main():
    instruction = INSTRUCTIONS[CURRENT_TASK_IDX]

    # ------------------------------------------------------------------
    # 1. 에이전트 초기화 (VLA 추론)
    # ------------------------------------------------------------------
    if USE_REMOTE_AGENT:
        from agents.remote_agent import RemoteAgent
        agent = RemoteAgent(target_ip="localhost", target_port=50051)
    else:
        from agents.openvla_agent import OpenVLAAgent
        agent = OpenVLAAgent()

    # ------------------------------------------------------------------
    # 2. Genesis 환경 초기화 (IK 계산 + 시뮬레이션 실행)
    # ------------------------------------------------------------------
    from envs.genesis_env import GenesisEnv
    env = GenesisEnv(urdf_path=URDF_PATH)

    # ------------------------------------------------------------------
    # 3. IK 제어기
    # ------------------------------------------------------------------
    from controllers.ik_ctrl import IKController
    # 서버에서 데이터셋 stat을 이용해 이미 unnormalize를 수행하므로, 
    # IK 컨트롤러 단에서의 인위적인 action_scaling은 제거하거나 1.0으로 두어야 합니다.
    action_scaling = 1.0
    
    controller = IKController(
        robot=env.robot,
        ee_link=env.ee_link,
        action_scaling=action_scaling,
        smoothing_alpha=0.9,   # main_real2와 일치 (1.0 → 0.9)
    )

    from utils.debugger import SystemDebugger
    debugger = SystemDebugger()

    # 에피소드 시작 전에 서버 내부 상태(prev_images, prev_state) 초기화.
    # 동일 서버에 재접속하는 경우 이전 run의 잔재가 첫 추론을 오염시키지 않도록.
    if hasattr(agent, "reset"):
        agent.reset()

    # ------------------------------------------------------------------
    # 4. 실제 카메라 초기화
    # ------------------------------------------------------------------
    cam_top, cam_wrist = None, None
    if USE_REAL_CAMERA:
        try:
            cam_top, cam_wrist = open_cameras()
            print(f"📷 카메라 연결 완료: top(idx={CAM_TOP_INDEX}), wrist(idx={CAM_WRIST_INDEX})")
        except Exception as e:
            print(f"⚠️ 카메라 연결 실패: {e}  →  Genesis 렌더 이미지로 대체합f니다.")
            cam_top, cam_wrist = None, None

    using_real_cam = USE_REAL_CAMERA and cam_top is not None and cam_wrist is not None
    print(f"🚀 실험 시작 | 카메라: {'실제' if using_real_cam else 'Genesis 렌더'} | 태스크: '{instruction}'")

    # ------------------------------------------------------------------
    # 5. 메인 루프
    # ------------------------------------------------------------------
    # main_real2와 일치: 추론 1회당 (K, action_dim) chunk를 받아 K cycle 동안 순차 소비.
    # Sim은 단일 thread이므로 chunk 소진 시점에 재추론 (real2의 async replace-on-arrival과
    # 다르지만, 모델이 학습 시 의도한 action_horizon=K 사용 패턴은 그대로 살림)
    chunk = None
    chunk_idx = 0

    time.sleep(1)
    try:
        for i in range(1000):
            # 루프시간 기록
            loop_start_time = time.time()

            # (1) 이미지 획득 (Genesis 렌더링)
            obs_dict = env.get_obs()
            img_primary = obs_dict['image_primary']
            img_wrist = obs_dict['image_wrist']

            # (2) 현재 관절 상태 (Radian -> Degree 변환)
            current_state = env.get_state()
            q_deg = np.rad2deg(current_state['q']).tolist()


            #dummy_raw_state = [24.307, -103.428, 96.131, 80.351, -105.098, 10.489]

            # (3) Octo 추론 — chunk 소진 시점에만 재추론
            need_predict = chunk is None or chunk_idx >= len(chunk)
            if need_predict:
                chunk = agent.predict(
                    img_primary,
                    instruction,
                    wrist_image=img_wrist,
                    state=q_deg,
                    # state=None if PRETRAINED_MODE else q_deg,
                )
                # remote_agent는 (K, action_dim) 반환. 1D fallback도 안전 처리.
                if chunk.ndim == 1:
                    chunk = chunk.reshape(1, -1)
                chunk_idx = 0

            raw_action = chunk[chunk_idx]
            chunk_idx += 1

            # 강제로 X축 앞으로만 1cm씩 이동하도록 셋팅
            # raw_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1])

            # (4) IK 및 실행
            q_target, gripper_target = controller.get_joint_targets(raw_action, current_state)
            env.step(q_target, gripper_target, sub_steps=10)

            # (5) 터미널 로그 출력 및 카메라 입력 시각화
            latency = (time.time() - loop_start_time) * 1000
            print(
                f"[Step {i:03d}] ⏱️ {latency:5.1f}ms | "
                f"chunk={chunk_idx-1}/{len(chunk)-1} {'🔄new' if need_predict else '     '} | "
                f"Pos: {np.round(raw_action[:3], 3)} | Grip: {raw_action[6]:.2f}"
            )

            # 입력 이미지 OpenCV 윈도우 창으로 시각화 (Dual View)
            render_top_bgr = cv2.cvtColor(np.array(img_primary), cv2.COLOR_RGB2BGR)
            render_wrist_bgr = cv2.cvtColor(np.array(img_wrist), cv2.COLOR_RGB2BGR)
            combined_bgr = np.hstack((render_top_bgr, render_wrist_bgr))

            cv2.imshow("Input View (Top | Wrist)", combined_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 10Hz 제어 주기 조절
            elapsed = time.time() - loop_start_time
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)

    except KeyboardInterrupt:
        print("\n실험 종료")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
