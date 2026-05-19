import genesis as gs
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

import os

class GenesisEnv:
    def __init__(self, urdf_path=None, show_viewer=True):
        if urdf_path is None:
            urdf_path = "/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
            
        # 1. 초기화 (GUI를 통한 개별 카메라 창 생성 방지: GUI=False)
        gs.init(seed=42, precision="32", logging_level="warning")

        # 2. 씬 구성 및 조명 최적화
        self.scene = gs.Scene(
            vis_options=gs.options.VisOptions(
                ambient_light=(0.9, 0.9, 0.9), 
                show_world_frame=True,
                world_frame_size=0.1,
                shadow=False, # 그림자 렌더링 제거
            ),
            show_viewer=show_viewer,
            # 가상 물리세계의 한 step당 시간 : 0.01s
            rigid_options=gs.options.RigidOptions(dt=0.01),
        )

        # 3. 바닥 색상 변경 (아이보리/베이지 계열)
        # 실제 환경과 유사하게 순백색에서 약간 노란기가 도는 색으로 변경
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
            surface=gs.surfaces.Rough(
                color=(0.93, 0.87, 0.75), # 더 짙고 확실한 베이지 색상으로 변경
                roughness=0.3, 
            )
        )
        
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=urdf_path, fixed=True)
        )

        # 4. 물체 및 방해물 배치
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04), 
                pos=(0.3, 0.0, 0.02),
            ),
            surface=gs.surfaces.Default(color=(1.0, 0.5, 0.0)) # 오렌지색으로 변경
        )

        # self.distractor = self.scene.add_entity(
        #     gs.morphs.Box(
        #         size=(0.08, 0.15, 0.05), 
        #         pos=(0.3, 0.2, 0.025),
        #     ),
        #     surface=gs.surfaces.Default(color=(0.9, 0.9, 0.9)) # 하얀 바구니 역할
        # )

        # 5. 듀얼 카메라 설정 (GUI=False로 설정하여 개별 창 생성을 막음)
        # Top View (Primary)
        self.cam_top = self.scene.add_camera(
            res=(256, 256), 
            pos=(-0.112, 0.109, 0.530), # 사용자가 튜닝한 최적의 위치
            lookat=(0.405, -0.010, 0.000), # 사용자가 튜닝한 최적의 초점
            fov=55,
            GUI=False # 메인 cv2 창에서만 보려고 하므로 False
        )
        
        # Wrist View (1인칭)
        self.cam_wrist = self.scene.add_camera(
            res=(256, 256),
            fov=50, # 65 -> 50으로 줄여 화면 확대
            GUI=False # 메별 창 생성 방지
        )
        
        self.apply_real_calibration()
        self.scene.build()

        self.ee_link = self.robot.get_link("gripper_link")
        self.arm_joints = [0, 1, 2, 3, 4]
        self.gripper_joint = [5]

        # 9. 초기 자세(Home Position) 설정
        # 데이터셋 수집 시와 동일한 시작 자세 (Degrees)
        home_angles_deg = [24.31, -103.43, 96.13, 80.35, -105.10, 10.49] 
        home_angles_rad = np.deg2rad(home_angles_deg)
        self.robot.set_dofs_position(home_angles_rad)

        # 8. 손목 카메라 링크 부착 및 위치 교정 (매우 중요)
        offset_T = np.eye(4)
        
        # 사용자가 튜닝한 오프셋 적용
        offset_T[:3, 3] = np.array([-0.150, -0.090, 0.080]) 
        
        # 사용자가 튜닝한 회전각 적용 (Roll:0, Pitch:71, Yaw:160)
        rot_mat = R.from_euler('xyz', [0.0, 71.0, 160.0], degrees=True).as_matrix()
        offset_T[:3, :3] = rot_mat

        wrist_link = self.robot.get_link("wrist_link")
        self.cam_wrist.attach(wrist_link, offset_T)

        self.ee_link = self.robot.get_link("gripper_link")
        self.arm_joints = [0, 1, 2, 3, 4]
        self.gripper_joint = [5]

    def apply_real_calibration(self):
        """실물 로봇의 캘리브레이션 데이터를 Genesis 가상 로봇에 주입합니다."""
        calib_data = {
            "shoulder_pan":  {"homing": -1843, "min": 675, "max": 3407},
            "shoulder_lift": {"homing": -980,  "min": 927, "max": 3343},
            "elbow_flex":    {"homing": 955,   "min": 1083, "max": 3303},
            "wrist_flex":    {"homing": -1477, "min": 398, "max": 2744},
            "wrist_roll":    {"homing": -949,  "min": 0, "max": 4095},
            "gripper":       {"homing": -2037, "min": 2026, "max": 3513}
        }
        
        joint_names = list(calib_data.keys())
        low_limits = []
        high_limits = []
        
        for name in joint_names:
            c = calib_data[name]
            # control_test_1.py의 공식 적용: (Raw - 2048 - Homing) * 360/4096
            deg_min = (c['min'] - 2048 - c['homing']) * (360.0 / 4096.0)
            deg_max = (c['max'] - 2048 - c['homing']) * (360.0 / 4096.0)
            
            # 부호 반전(-) 적용 (하드웨어-URDF 방향 일치용)
            real_min_deg = min(-deg_min, -deg_max)
            real_max_deg = max(-deg_min, -deg_max)
            
            low_limits.append(np.deg2rad(real_min_deg))
            high_limits.append(np.deg2rad(real_max_deg))

        # Genesis 로봇 DOF 한계 설정 (IK 계산 시 이 범위를 준수함)
        # [주의] 현재 Genesis 버전에서는 런타임에 DoF 한계를 수정하는 set_dofs_limit API가 지원되지 않을 수 있습니다.
        # 대신 hardware_remote_test.py의 메인 루프에서 SAFE_LIMITS를 통한 Clipping을 수행합니다.
        # low_limits_tensor = torch.tensor(low_limits, dtype=gs.tc_float)
        # high_limits_tensor = torch.tensor(high_limits, dtype=gs.tc_float)
        # try:
        #     self.robot.set_dofs_limit(low=low_limits_tensor, high=high_limits_tensor)
        # except Exception as e:
        #     print(f"⚠️ DoF 한계 주입 건너뜀: {e}")
            
        print(f"✅ Digital Twin: {len(joint_names)}개 관절의 실측 캘리브레이션 데이터 로드 완료 (제어 루프에서 필터링됨).")

    def get_obs(self):
        obs_dict = {}
        for cam_name, cam_obj in zip(['image_primary', 'image_wrist'], [self.cam_top, self.cam_wrist]):
            render_output = cam_obj.render(rgb=True)
            rgb = render_output[0] if isinstance(render_output, tuple) else render_output
            
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.detach().cpu().numpy()
                
            if rgb.dtype in [np.float32, np.float64]:
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                
            obs_dict[cam_name] = Image.fromarray(rgb).convert("RGB")
        return obs_dict

    def get_state(self):
        """로봇의 현재 상태(위치, 쿼터니언, 관절각)를 가져옵니다."""
        # [수정] 모든 텐서 추출에 detach().cpu().numpy() 적용
        pos = self.ee_link.get_pos().detach().cpu().numpy()
        
        # Genesis: [w, x, y, z] -> Scipy: [x, y, z, w]
        quat_wxyz = self.ee_link.get_quat().detach().cpu().numpy()
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        
        rot_mat = R.from_quat(quat_xyzw).as_matrix()
        q_current = self.robot.get_dofs_position().detach().cpu().numpy()
        
        return {'pos': pos, 'quat': quat_xyzw, 'rot_mat': rot_mat, 'q': q_current}

    def step(self, q_target, gripper_target, sub_steps=30):
        if hasattr(q_target, "cpu"):
            q_target = q_target.detach().cpu().numpy()

        # 암 관절: IK가 rad로 풀어주므로 Genesis(rad)에 그대로 전달.
        self.robot.control_dofs_position(q_target[self.arm_joints], self.arm_joints)

        # 그리퍼: IKController.ghost_gripper는 모터 단위(deg, clip 0~96)로 누적됨.
        # Genesis는 rad를 기대하므로 deg→rad 변환 후 전달.
        # (real 경로는 변환 없이 그대로 모터에 씀 — hardware_server가 deg를 받음)
        gripper_target_rad = np.deg2rad(float(gripper_target))
        self.robot.control_dofs_position(np.array([gripper_target_rad]), self.gripper_joint)

        for _ in range(sub_steps):
            self.scene.step()