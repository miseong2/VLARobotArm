import os
import sys
import time
import threading
import numpy as np
import cv2
from PIL import Image

# -------------------------------------------------------------------
# 클라이언트 / 컨트롤러 임포트 (main_real.py와 동일)
# -------------------------------------------------------------------
from envs.real_env_client import RealRobotEnvClient
from controllers.ik_ctrl import IKController

current_dir = os.path.dirname(os.path.abspath(__file__))
proto_dir = os.path.join(current_dir, "proto")
if proto_dir not in sys.path:
    sys.path.append(proto_dir)

# ===================================================================
# CONFIG
# ===================================================================
USE_REMOTE_AGENT = True
USE_REAL_CAMERA  = True
PRETRAINED_MODE  = False

CAM_TOP_INDEX    = 0
CAM_WRIST_INDEX  = 2

DEFAULT_URDF = "/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
URDF_PATH = os.getenv("ROBOT_URDF", DEFAULT_URDF)

INSTRUCTIONS = [
    "go to the pen",
]
CURRENT_TASK_IDX = 0

# 멀티스레드 제어 설정
CONTROL_HZ      = 10.0
CONTROL_PERIOD  = 1.0 / CONTROL_HZ      # 0.1s
ACTION_DIM      = 7
SHOW_VIZ        = True                  # 카메라 시각화 on/off
MAX_STEPS       = 2000                  # 메인 control loop 최대 step
FIRST_CHUNK_TIMEOUT = 15.0              # 추론 thread가 첫 chunk 던질 때까지 대기

# Hz 모니터링: 실제 사이클 간격이 목표(10Hz=100ms)에 맞는지 검증용
HZ_WINDOW_N     = 20                    # rolling 평균에 쓸 최근 사이클 수
HZ_REPORT_EVERY = 20                    # 매 N step마다 control Hz 요약 출력
LOG_DETAIL_TIMING = True                # per-step에 state/ik/step 구간별 ms 표시
# ===================================================================


# ===================================================================
# 동시성 자원
# ===================================================================
class ChunkBuffer:
    """추론 thread가 publish, 메인 thread가 step 단위로 consume.

    설계 메모:
    - chunk shape = (K, ACTION_DIM). Octo 학습 설정상 K=4.
    - 추론 thread가 새 chunk를 던지면 기존 chunk를 통째로 교체 (replace-on-arrival).
      → 가장 신선한 시각 정보가 항상 chunk[0]에 반영됨.
    - chunk 소진 후 메인 thread는 zero action(=정지)을 받아 안전 fallback.
      (학습 분포 밖 step을 누적해 ghost target이 발산하는 것보다 정지가 안전)
    """

    def __init__(self, action_dim=7):
        self.lock = threading.Lock()
        self.action_dim = action_dim
        self.chunk = None       # ndarray (K, action_dim) or None
        self.idx = 0
        self.first_chunk_event = threading.Event()

    def replace(self, new_chunk):
        with self.lock:
            self.chunk = new_chunk
            self.idx = 0
        self.first_chunk_event.set()

    def pop(self):
        """다음 action 1 step을 반환. (action, stale_flag)"""
        with self.lock:
            if self.chunk is None or self.idx >= len(self.chunk):
                return np.zeros(self.action_dim, dtype=np.float32), True
            a = self.chunk[self.idx].copy()
            self.idx += 1
            return a, False

    def wait_first(self, timeout=10.0):
        return self.first_chunk_event.wait(timeout)


class SharedState:
    """메인 thread → 추론 thread로 최신 관절 각도(deg) 전달."""

    def __init__(self):
        self.lock = threading.Lock()
        self.q_deg = None
        self.ready_event = threading.Event()

    def set(self, q_deg):
        with self.lock:
            self.q_deg = list(q_deg)
        self.ready_event.set()

    def get(self):
        with self.lock:
            return None if self.q_deg is None else list(self.q_deg)


class SharedFrame:
    """추론 thread → 메인 thread로 시각화용 BGR frame 전달.
    cv2.imshow/waitKey는 메인 thread에서만 호출하므로 frame 자체를 넘긴다."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None

    def set(self, frame_bgr):
        with self.lock:
            self.frame = frame_bgr

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


# ===================================================================
# 추론 thread
# ===================================================================
def inference_worker(agent, instruction, cam_top, cam_wrist,
                     shared_state, shared_frame, buffer, stop_event):
    """추론 thread는 카메라(read)와 agent(gRPC)를 owning한다.

    이유:
    - cv2.VideoCapture는 thread-safe하지 않으므로 한 thread에서만 read.
    - 추론 직전에 카메라를 read하면 prev↔current frame gap이 추론 cycle(≈167ms)에
      맞춰져 학습 분포(100ms)와 약간 어긋나지만, 1단계에서는 수용. (정밀화는 별도
      camera-grab thread + ring buffer로 100ms stride 맞추는 식으로 2단계에서 진행)
    """
    # 메인 thread가 초기 state를 채울 때까지 대기
    if not shared_state.ready_event.wait(timeout=5.0):
        print("⚠️ [Infer] 초기 state 대기 실패. 종료.")
        return

    n_infer = 0
    while not stop_event.is_set():
        t0 = time.time()

        ret_top, frame_top = cam_top.read()
        ret_wrist, frame_wrist = cam_wrist.read()
        if not ret_top or not ret_wrist:
            time.sleep(0.005)
            continue

        img_primary = Image.fromarray(cv2.cvtColor(frame_top, cv2.COLOR_BGR2RGB))
        img_wrist   = Image.fromarray(cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB))

        q_deg = shared_state.get()
        if q_deg is None:
            continue

        try:
            chunk = agent.predict(
                img_primary,
                instruction,
                wrist_image=img_wrist,
                state=q_deg,
            )
        except Exception as e:
            print(f"⚠️ [Infer] predict 실패: {e}")
            continue

        # remote_agent.py에서 (K, action_dim)으로 reshape해서 반환해야 함.
        # 안전망: 1D로 들어오면 단일 action으로 처리해 chunking 비활성 시에도 동작.
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)

        buffer.replace(chunk)

        if SHOW_VIZ:
            # cv2.imshow는 메인 thread에서만 호출 → frame만 넘긴다
            combined = np.hstack((frame_top, frame_wrist))
            shared_frame.set(combined)

        n_infer += 1
        latency_ms = (time.time() - t0) * 1000
        if n_infer == 1 or n_infer % 5 == 0:
            print(f"[Infer #{n_infer:03d}] ⏱️ {latency_ms:5.1f}ms | chunk_len={len(chunk)}")


# ===================================================================
# 메인 thread (제어 루프)
# ===================================================================
def main():
    instruction = INSTRUCTIONS[CURRENT_TASK_IDX]

    # 1. 에이전트
    if USE_REMOTE_AGENT:
        from agents.remote_agent import RemoteAgent
        agent = RemoteAgent(target_ip="localhost", target_port=50051)
    else:
        from agents.openvla_agent import OpenVLAAgent
        agent = OpenVLAAgent()

    # 2. 하드웨어 클라이언트 (ZMQ REQ socket — 메인 thread만 사용)
    print("🔌 하드웨어 서버(ZMQ)에 연결을 시도합니다...")
    env = RealRobotEnvClient(target_ip="localhost", target_port=5555, urdf_path=URDF_PATH)

    # 3. IK 제어기
    controller = IKController(
        robot=env.shadow_robot,
        ee_link=env.ee_link,
        action_scaling=1.0,
        smoothing_alpha=0.9,
    )

    # 4. 카메라 초기화 (추론 thread가 read만 담당)
    print(f"📷 카메라 연결 중: top({CAM_TOP_INDEX}), wrist({CAM_WRIST_INDEX})")
    cam_top = cv2.VideoCapture(CAM_TOP_INDEX)
    cam_top.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam_top.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam_top.set(cv2.CAP_PROP_FPS, 30)
    cam_top.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cam_wrist = cv2.VideoCapture(CAM_WRIST_INDEX)
    cam_wrist.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam_wrist.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam_wrist.set(cv2.CAP_PROP_FPS, 30)
    cam_wrist.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cam_top.isOpened() or not cam_wrist.isOpened():
        print("🚨 카메라를 열 수 없습니다. 인덱스를 확인해 주세요.")
        sys.exit(1)

    for name, cam in [("cam_top  ", cam_top), ("cam_wrist", cam_wrist)]:
        w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cam.get(cv2.CAP_PROP_FPS)
        buf = cam.get(cv2.CAP_PROP_BUFFERSIZE)
        print(f"📷 {name} actual: {w:.0f}x{h:.0f} @ {fps:.0f}fps, buffersize={buf:.0f}")

    # 서버 내부 prev_images/prev_state 초기화 (에피소드 경계)
    if hasattr(agent, "reset"):
        agent.reset()

    # 5. 공유 자원
    buffer       = ChunkBuffer(action_dim=ACTION_DIM)
    shared_state = SharedState()
    shared_frame = SharedFrame()
    stop_event   = threading.Event()

    # 6. 메인 thread가 초기 state를 한 번 채워서 추론 thread가 즉시 시작 가능하도록
    init_state = env.get_state()
    if init_state is None:
        print("🚨 초기 state read 실패. 종료.")
        sys.exit(1)
    shared_state.set(np.rad2deg(init_state['q']).tolist())

    # 7. 추론 thread 시작
    infer_thread = threading.Thread(
        target=inference_worker,
        args=(agent, instruction, cam_top, cam_wrist,
              shared_state, shared_frame, buffer, stop_event),
        daemon=True,
        name="InferenceThread",
    )
    infer_thread.start()

    print(f"🚀 실물 로봇 제어 시작 | 태스크: '{instruction}'")
    print(f"⏳ 첫 chunk 도착 대기 (max {FIRST_CHUNK_TIMEOUT}s)...")
    if not buffer.wait_first(timeout=FIRST_CHUNK_TIMEOUT):
        print("🚨 첫 chunk 도착 timeout. 종료.")
        stop_event.set()
        infer_thread.join(timeout=2.0)
        sys.exit(1)
    print("✅ 첫 chunk 도착. 10Hz control loop 시작.")

    # 8. 메인 control loop (10Hz 고정 tick)
    next_tick = time.time()
    # Hz 검증용 상태
    prev_loop_start = None          # 직전 사이클 시작 시각 → 사이클 간격(dt) 계산
    cycle_dt_window = []            # 최근 N 사이클의 dt(ms) — rolling 평균용
    drift_count = 0                 # sleep_for < 0 (한 사이클 따라잡기 실패) 누적
    # 구간별 bottleneck 추적 (rolling 평균용)
    state_ms_window = []            # env.get_state() 소요 (ZMQ + motor read)
    ik_ms_window    = []            # controller.get_joint_targets() 소요 (genesis IK)
    step_ms_window  = []            # env.step() 소요 (ZMQ + motor write)
    other_ms_window = []            # 나머지 (buffer.pop, print, shared_*.set 등)
    try:
        for i in range(MAX_STEPS):
            loop_start = time.time()

            # [Hz] 사이클 간격(=실제 제어 주기) 측정. work_time이 아니라 loop_start↔loop_start 간격.
            if prev_loop_start is not None:
                cycle_dt_ms = (loop_start - prev_loop_start) * 1000.0
                cycle_dt_window.append(cycle_dt_ms)
                if len(cycle_dt_window) > HZ_WINDOW_N:
                    cycle_dt_window.pop(0)
            else:
                cycle_dt_ms = float("nan")
            prev_loop_start = loop_start

            # (1) 실제 state 동기화 → IK용 + 추론 thread에 publish
            t_a = time.time()
            current_state = env.get_state()
            t_b = time.time()
            state_ms = (t_b - t_a) * 1000.0

            if current_state is None:
                # ZMQ timeout: 다음 tick까지 정렬만 맞추고 skip
                next_tick = max(next_tick + CONTROL_PERIOD, time.time())
                time.sleep(max(0.0, next_tick - time.time()))
                continue

            shared_state.set(np.rad2deg(current_state['q']).tolist())

            # (2) chunk buffer에서 다음 step의 action pop
            raw_action, stale = buffer.pop()

            # (3) IK 계산 + 모터 명령 (구간별 측정)
            t_c = time.time()
            q_target, gripper_target = controller.get_joint_targets(raw_action, current_state)
            t_d = time.time()
            ik_ms = (t_d - t_c) * 1000.0

            env.step(q_target, gripper_target)
            t_e = time.time()
            step_ms = (t_e - t_d) * 1000.0

            # (4) 로그 — work_ms(이번 cycle의 일한 시간) + cycle_dt_ms(직전→이번 사이클 간격)
            work_ms = (time.time() - loop_start) * 1000
            # other = work에서 측정된 3구간(state/ik/step)을 뺀 나머지 (buffer.pop, shared_state.set, print 등)
            other_ms = max(0.0, work_ms - state_ms - ik_ms - step_ms)
            # rolling window 갱신
            for win, val in (
                (state_ms_window, state_ms),
                (ik_ms_window, ik_ms),
                (step_ms_window, step_ms),
                (other_ms_window, other_ms),
            ):
                win.append(val)
                if len(win) > HZ_WINDOW_N:
                    win.pop(0)

            stale_mark = "⚠️stale" if stale else "      "
            if LOG_DETAIL_TIMING:
                print(
                    f"[Ctrl {i:03d}] work={work_ms:5.1f}ms dt={cycle_dt_ms:6.1f}ms {stale_mark} | "
                    f"state={state_ms:5.1f} ik={ik_ms:5.1f} step={step_ms:5.1f} other={other_ms:4.1f} | "
                    f"a[:3]={np.round(raw_action[:3], 3).tolist()} grip={raw_action[6]:+.2f}"
                )
            else:
                print(
                    f"[Ctrl {i:03d}] work={work_ms:5.1f}ms dt={cycle_dt_ms:6.1f}ms {stale_mark} | "
                    f"a[:3]={np.round(raw_action[:3], 3).tolist()} grip={raw_action[6]:+.2f}"
                )

            # [Hz] 매 HZ_REPORT_EVERY step마다 rolling 평균 Hz + 구간별 평균 요약
            if i > 0 and i % HZ_REPORT_EVERY == 0 and cycle_dt_window:
                mean_dt = float(np.mean(cycle_dt_window))
                max_dt  = float(np.max(cycle_dt_window))
                actual_hz = 1000.0 / mean_dt if mean_dt > 0 else 0.0
                mean_state = float(np.mean(state_ms_window)) if state_ms_window else 0.0
                mean_ik    = float(np.mean(ik_ms_window))    if ik_ms_window    else 0.0
                mean_step  = float(np.mean(step_ms_window))  if step_ms_window  else 0.0
                mean_other = float(np.mean(other_ms_window)) if other_ms_window else 0.0
                print(
                    f"📊 [HzReport @ step {i}] "
                    f"avg_dt={mean_dt:5.1f}ms ({actual_hz:4.2f}Hz, target {CONTROL_HZ:.1f}Hz) "
                    f"max_dt={max_dt:5.1f}ms drift={drift_count}/{i} | "
                    f"breakdown: state={mean_state:5.1f} ik={mean_ik:5.1f} step={mean_step:5.1f} other={mean_other:4.1f}"
                )

            # (5) 시각화 (메인 thread)
            if SHOW_VIZ:
                frame = shared_frame.get()
                if frame is not None:
                    cv2.imshow("Real Robot Control (Top | Wrist)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            # (6) 10Hz 고정 tick — drift 방지
            next_tick += CONTROL_PERIOD
            sleep_for = next_tick - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # control loop가 한 cycle을 못 따라잡음 → tick 재정렬 + 카운트
                drift_count += 1
                next_tick = time.time()

    except KeyboardInterrupt:
        print("\n🛑 키보드 입력으로 실험 강제 종료.")
    except Exception as e:
        print(f"\n🚨 에러 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # [Hz] 종합 요약 — 전체 사이클의 평균/최대 dt + 구간별 평균
        if cycle_dt_window:
            mean_dt = float(np.mean(cycle_dt_window))
            max_dt  = float(np.max(cycle_dt_window))
            actual_hz = 1000.0 / mean_dt if mean_dt > 0 else 0.0
            mean_state = float(np.mean(state_ms_window)) if state_ms_window else 0.0
            mean_ik    = float(np.mean(ik_ms_window))    if ik_ms_window    else 0.0
            mean_step  = float(np.mean(step_ms_window))  if step_ms_window  else 0.0
            mean_other = float(np.mean(other_ms_window)) if other_ms_window else 0.0
            print(
                f"📊 [HzReport FINAL] avg_dt={mean_dt:5.1f}ms ({actual_hz:4.2f}Hz, "
                f"target {CONTROL_HZ:.1f}Hz) max_dt={max_dt:5.1f}ms drift={drift_count}"
            )
            print(
                f"📊 [Breakdown FINAL] state={mean_state:5.1f}ms ik={mean_ik:5.1f}ms "
                f"step={mean_step:5.1f}ms other={mean_other:4.1f}ms"
            )

        stop_event.set()
        infer_thread.join(timeout=2.0)

        # [홈 복귀] 추론 thread 정지 + ZMQ socket 정리 전에 호출.
        # send_go_home은 fresh socket을 만들어 쓰므로 기존 self.socket이
        # Ctrl+C로 broken state가 됐어도 무관하게 작동한다.
        # env.disconnect() 전에 호출되어야 context.term() 전에 fresh socket 사용 가능.
        try:
            print("🏠 홈자세 복귀 명령 전송 중...")
            ok = env.send_go_home(timeout_ms=8000)
            print(f"🏠 홈 복귀 결과: {'성공' if ok else '실패'}")
        except Exception as e:
            print(f"⚠️ 홈 복귀 명령 전송 실패: {type(e).__name__}: {e}")

        if hasattr(cam_top, 'release'):
            cam_top.release()
        if hasattr(cam_wrist, 'release'):
            cam_wrist.release()
        env.disconnect()
        cv2.destroyAllWindows()
        print("🧹 리소스 정리 완료.")


if __name__ == "__main__":
    main()
