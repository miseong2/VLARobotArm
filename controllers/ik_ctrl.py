import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from .base_ctrl import BaseController

class IKController(BaseController):
    def __init__(self, robot, ee_link, action_scaling=0.05, cam_pos=None, lookat=None, smoothing_alpha=0.3,
                 resync_drift_cm=5.0, resync_rot_deg=30.0, min_z=0.06):
        self.robot = robot
        self.ee_link = ee_link
        self.action_scaling = action_scaling
        self.smoothing_alpha = smoothing_alpha
        self.last_q_target = None

        self.R_cam_to_base = np.eye(3)

        # ----------------------------------------------------------------
        # [Ghost Re-sync] 임계값
        # ghost와 real 간 갭이 이 값을 넘으면 ghost를 real로 강제 동기화한다.
        # 학습 분포 기준:
        #   - pos delta max ≈ 2.5cm/step  → 5cm는 정상 추적 시 절대 안 넘는 값
        #   - rot delta max ≈ 15°/step    → 30°도 마찬가지
        # 너무 빡빡하면(예: 2cm) 정상 동작 중에도 발동해서 jerky해지고,
        # 너무 느슨하면(예: 15cm) 발산이 깊어진 뒤에야 발동해서 회복 손해.
        # ----------------------------------------------------------------
        self.resync_drift_cm = resync_drift_cm
        self.resync_rot_deg = resync_rot_deg

        # ----------------------------------------------------------------
        # [Z Floor] 그리퍼 바닥 긁힘 방지 하한
        # 그리퍼 끝(fingertip)이 테이블에 끌리지 않도록 ghost.z의 하한을 둔다.
        # gripper_link 기준점은 fingertip보다 위에 있어 gripper_link.z = min_z일 때
        # fingertip은 그보다 아래에 위치. 너무 빡빡하면 큐브 grasp 불가, 너무 느슨하면
        # 긁힘. 0.06m가 fingertip이 큐브 중앙 높이에 오는 균형점.
        # 긁히면 0.07로↑, grasp 실패하면 0.05로↓ 튜닝.
        # ----------------------------------------------------------------
        self.min_z = min_z

        print(f"🔧 IK Controller: 회전 보정 미적용 (Identity 모드) | Smoothing Alpha: {smoothing_alpha}"
              f" | Re-sync 임계값: drift>{resync_drift_cm}cm or rot>{resync_rot_deg}°"
              f" | Z floor: {min_z}m")

        # [추가] 고스트 타겟 변수 초기화
        self.reset_ghost()

    def reset_ghost(self):
        """에피소드가 시작될 때 고스트 타겟을 초기화합니다."""
        self.ghost_pos = None
        self.ghost_rot_mat = None
        self.ghost_qpos = None
        self.ghost_gripper = None

    def get_joint_targets(self, raw_action, current_state):
        """
        raw_action: OpenVLA 출력 [7]
        current_state: {'pos': [3], 'quat': [4], 'rot_mat': [3,3], 'q': [7]}
        """
        # ==========================================
        # [고스트 타겟 1] 최초 1회 실행 시 현재 물리 상태를 고스트로 복사
        # ==========================================
        if self.ghost_pos is None:
            self.ghost_pos = current_state['pos'].copy()
            self.ghost_rot_mat = current_state['rot_mat'].copy()
            self.ghost_qpos = current_state['q'].copy()
            #self.ghost_gripper = current_state['q'][5]  # 제네시스 SO-100 기준 6번째가 그리퍼
            self.ghost_gripper = 1.2  # rad 단위, [-5.38, -3.09] 중간값
            print(
                f"👻 Ghost Target 초기화 | "
                f"ghost_pos={np.round(self.ghost_pos, 4).tolist()} "
                f"real_pos={np.round(current_state['pos'], 4).tolist()}"
            )
        else:
            # ============================================================
            # [Ghost Re-sync] 발산 감지 시 ghost를 real로 강제 동기화
            # ------------------------------------------------------------
            # 배경: SO-101은 5-DOF arm으로 6-DOF EE pose를 표현하므로 IK가
            #       over-constrained. 회전 누적이 어느 임계점을 넘으면 IK가
            #       다른 해 branch로 점프(IK flipping)하면서 real EE가 ghost
            #       에서 멀어진다. 한 번 벌어지면 ghost는 누적만 하고 real
            #       state와 동기화되는 메커니즘이 없어 영구 발산.
            #
            # 동작: 이전 스텝 ghost와 현재 real 사이 갭(위치 OR 회전)이
            #       임계값을 넘으면 ghost를 real로 덮어써서 좌표계 리셋.
            #       이렇게 하면 다음 스텝부터 모델 delta가 다시 "올바른
            #       기준점(real)"에서 누적되므로 추론+제어 파이프라인이
            #       의미 있게 작동한다.
            #
            # 주의: 위치/회전/관절 세 ghost를 동시에 동기화해야 IK가 다음
            #       스텝에 또 flipping하지 않는다. last_q_target도 리셋해
            #       smoothing이 발산 시점 q_target으로 끌고 가지 않도록 함.
            #       gripper는 누적값 유지 (위치/회전 발산과 무관).
            # ============================================================
            pre_drift_cm = np.linalg.norm(self.ghost_pos - current_state['pos']) * 100
            R_diff_pre = self.ghost_rot_mat @ current_state['rot_mat'].T
            pre_rot_deg = np.rad2deg(np.linalg.norm(R.from_matrix(R_diff_pre).as_rotvec()))

            if pre_drift_cm > self.resync_drift_cm or pre_rot_deg > self.resync_rot_deg:
                print(
                    f"⚠️  Ghost Re-sync 발동 | "
                    f"drift={pre_drift_cm:.2f}cm (>{self.resync_drift_cm}cm) "
                    f"rot_diff={pre_rot_deg:.1f}° (>{self.resync_rot_deg}°)"
                )
                # 좌표계 리셋: 위치 / 회전 / 관절 모두 real로 덮어쓰기
                self.ghost_pos = current_state['pos'].copy()
                self.ghost_rot_mat = current_state['rot_mat'].copy()
                self.ghost_qpos = current_state['q'].copy()
                # 평활화 baseline도 리셋: 망가진 q_target이 EMA로 살아남지 않도록
                self.last_q_target = current_state['q'].copy()

        # ==========================================
        # [고스트 타겟 2] 실제 상태가 아닌 '고스트 타겟'에 델타 누적
        # ==========================================
        # 1. 포지션 변환 및 누적
        delta_pos_cam = raw_action[:3].astype(np.float32) * self.action_scaling
        delta_pos_cam = np.clip(delta_pos_cam, -0.03, 0.03)  # 안전 Clamping
        self.ghost_pos = self.ghost_pos + (self.R_cam_to_base @ delta_pos_cam)

        # [Z Floor] ghost.z를 self.min_z 이상으로 강제. 모델이 음수 z delta를
        # 누적해 ghost가 바닥 아래로 내려가는 것을 막음. real EE는 IK가 ghost.z를
        # 따라가는 만큼만 내려가므로, 여기서 ghost.z를 막으면 real도 안전 높이 유지.
        if self.ghost_pos[2] < self.min_z:
            self.ghost_pos[2] = self.min_z

        # [DEBUG] ghost vs real EE pose drift
        real_pos = current_state['pos']
        drift_cm = np.linalg.norm(self.ghost_pos - real_pos) * 100
        print(
            f"📏 drift={drift_cm:5.2f}cm | "
            f"ghost={np.round(self.ghost_pos, 3).tolist()} "
            f"real={np.round(real_pos, 3).tolist()}"
        )

        # 2. 회전 변환 및 누적
        # 학습 데이터 규약: R_delta = R_next @ R_curr.T  (base-frame 회전)
        # → 복원 공식: R_next = R_delta @ R_curr  (LEFT multiplication)
        delta_rotvec = raw_action[3:6].astype(np.float32)
        delta_rot_mat = R.from_rotvec(delta_rotvec).as_matrix()
        self.ghost_rot_mat = delta_rot_mat @ self.ghost_rot_mat

        # [DEBUG] ghost vs real rotation comparison
        R_diff = self.ghost_rot_mat @ current_state['rot_mat'].T
        rot_diff_deg = np.rad2deg(np.linalg.norm(R.from_matrix(R_diff).as_rotvec()))
        ghost_rotvec_deg = np.rad2deg(R.from_matrix(self.ghost_rot_mat).as_rotvec())
        real_rotvec_deg = np.rad2deg(R.from_matrix(current_state['rot_mat']).as_rotvec())
        print(
            f"🔄 rot_diff={rot_diff_deg:5.1f}° | "
            f"ghost_rv={np.round(ghost_rotvec_deg, 1).tolist()}° "
            f"real_rv={np.round(real_rotvec_deg, 1).tolist()}°"
        )

        # Scipy: [x, y, z, w] -> Genesis: [w, x, y, z]
        target_quat_xyzw = R.from_matrix(self.ghost_rot_mat).as_quat()
        target_quat_wxyz = np.array([target_quat_xyzw[3], target_quat_xyzw[0], target_quat_xyzw[1], target_quat_xyzw[2]])

        # ---------------------------------------------------------
        # 3. IK 계산 (실제 위치가 아닌 고스트 변수 주입)
        # ---------------------------------------------------------
        # [TIMING] genesis IK 솔버 자체 시간
        _t_ik_start = time.time()
        q_target = self.robot.inverse_kinematics(
            link=self.ee_link,
            pos=self.ghost_pos,            # [수정] current_state['pos'] -> ghost_pos
            quat=target_quat_wxyz,
            init_qpos=self.ghost_qpos,     # [수정] IK Flipping 방지를 위해 방금 전 계산한 고스트 관절 주입
            respect_joint_limit=True,
        )
        _t_ik_solver_end = time.time()

        # 다음 스텝을 위해 이번에 푼 순수 수학적 관절 각도를 고스트에 저장
        # [TIMING] genesis는 GPU tensor 반환 → .cpu()는 GPU 동기화 wait. 추론 thread가
        # GPU를 점유 중이면 여기서 block 시간이 폭증할 수 있다 (자원 경쟁 증거).
        self.ghost_qpos = q_target.detach().cpu().numpy()
        _t_cpu_sync_end = time.time()

        _ik_solver_ms = (_t_ik_solver_end - _t_ik_start) * 1000.0
        _cpu_sync_ms  = (_t_cpu_sync_end  - _t_ik_solver_end) * 1000.0
        print(f"⏱️ [IK] solver={_ik_solver_ms:5.1f}ms cpu_sync={_cpu_sync_ms:5.1f}ms")

        # ---------------------------------------------------------
        # 4. 관절 궤적 평활화 (Joint Smoothing & EMA)
        # (이 부분은 모터 제어용이므로 그대로 유지)
        # ---------------------------------------------------------
        q_target_np = self.ghost_qpos.copy()
        if self.last_q_target is None:
            self.last_q_target = q_target_np
        else:
            q_target_np = self.smoothing_alpha * q_target_np + (1 - self.smoothing_alpha) * self.last_q_target
            self.last_q_target = q_target_np
            q_target = torch.from_numpy(q_target_np).to(q_target.device) if hasattr(q_target, 'device') else q_target_np
        
        # ---------------------------------------------------------
        # 5. 그리퍼 누적 제어 (고스트 그리퍼에 누적)
        # ---------------------------------------------------------
        gripper_delta = float(raw_action[6])
        self.ghost_gripper = self.ghost_gripper + gripper_delta

        self.ghost_gripper = np.clip(self.ghost_gripper, 0.0, 96.0)

        # [DEBUG]
        if abs(gripper_delta) > 0.01: # 너무 작은 변화는 출력 생략
            print(f"🔧 Ghost Gripper: {self.ghost_gripper:.4f} rad (Delta: {gripper_delta:.2f} deg)")

        # [수정] 업데이트된 고스트 값을 반환
        return q_target, self.ghost_gripper