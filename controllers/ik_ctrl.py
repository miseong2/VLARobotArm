import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from .base_ctrl import BaseController

class IKController(BaseController):
    def __init__(self, robot, ee_link, action_scaling=0.05, cam_pos=None, lookat=None, smoothing_alpha=0.3):
        self.robot = robot
        self.ee_link = ee_link
        self.action_scaling = action_scaling
        self.smoothing_alpha = smoothing_alpha 
        self.last_q_target = None 
        
        self.R_cam_to_base = np.eye(3) 
        print(f"🔧 IK Controller: 회전 보정 미적용 (Identity 모드) | Smoothing Alpha: {smoothing_alpha}")
        
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
            self.ghost_gripper = -4.0  # rad 단위, [-5.38, -3.09] 중간값
            print("👻 Ghost Target 초기화 완료 (Home State 기준)")

        # ==========================================
        # [고스트 타겟 2] 실제 상태가 아닌 '고스트 타겟'에 델타 누적
        # ==========================================
        # 1. 포지션 변환 및 누적
        delta_pos_cam = raw_action[:3].astype(np.float32) * self.action_scaling
        delta_pos_cam = np.clip(delta_pos_cam, -0.03, 0.03)  # 안전 Clamping
        self.ghost_pos = self.ghost_pos + (self.R_cam_to_base @ delta_pos_cam)

        # 2. 회전 변환 및 누적
        # 학습 데이터 규약: R_delta = R_next @ R_curr.T  (base-frame 회전)
        # → 복원 공식: R_next = R_delta @ R_curr  (LEFT multiplication)
        delta_rotvec = raw_action[3:6].astype(np.float32)
        delta_rot_mat = R.from_rotvec(delta_rotvec).as_matrix()
        self.ghost_rot_mat = delta_rot_mat @ self.ghost_rot_mat
        
        # Scipy: [x, y, z, w] -> Genesis: [w, x, y, z]
        target_quat_xyzw = R.from_matrix(self.ghost_rot_mat).as_quat()
        target_quat_wxyz = np.array([target_quat_xyzw[3], target_quat_xyzw[0], target_quat_xyzw[1], target_quat_xyzw[2]])

        # ---------------------------------------------------------
        # 3. IK 계산 (실제 위치가 아닌 고스트 변수 주입)
        # ---------------------------------------------------------
        q_target = self.robot.inverse_kinematics(
            link=self.ee_link,
            pos=self.ghost_pos,            # [수정] current_state['pos'] -> ghost_pos
            quat=target_quat_wxyz,
            init_qpos=self.ghost_qpos,     # [수정] IK Flipping 방지를 위해 방금 전 계산한 고스트 관절 주입
            respect_joint_limit=True,
        )

        # 다음 스텝을 위해 이번에 푼 순수 수학적 관절 각도를 고스트에 저장
        self.ghost_qpos = q_target.detach().cpu().numpy()

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
        gripper_delta_deg = float(raw_action[6])
        delta_rad = np.deg2rad(gripper_delta_deg)

        # [수정] 실제 상태(current_state)가 아닌 고스트 상태(self.ghost_gripper)에 누적
        # 만약 초기화 시 ghost_gripper를 설정하지 않았다면 82번줄 근처 초기화 확인 필요
        self.ghost_gripper = self.ghost_gripper + delta_rad 

        # 물리적 한계 내로 클리핑 (하드웨어 그리퍼 범위: ~[-308°, -177°] = [-5.38, -3.09] rad)
        self.ghost_gripper = np.clip(self.ghost_gripper, -6.0, 0.0)

        # [DEBUG]
        if abs(gripper_delta_deg) > 0.01: # 너무 작은 변화는 출력 생략
            print(f"🔧 Ghost Gripper: {self.ghost_gripper:.4f} rad (Delta: {gripper_delta_deg:.2f} deg)")

        # [수정] 업데이트된 고스트 값을 반환
        return q_target, self.ghost_gripper