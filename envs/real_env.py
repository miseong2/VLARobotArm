import zmq
import numpy as np
import torch
import genesis as gs
from scipy.spatial.transform import Rotation as R

class RealRobotEnvClient:
    def __init__(self, target_ip="localhost", target_port=5555, urdf_path=None):
        # 1. ZMQ 클라이언트 설정 (REQ 패턴)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{target_ip}:{target_port}")
        # 타임아웃 설정 (서버 응답 없을 시 무한 대기 방지)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000) 
        
        print(f"📡 [Client] Connected to Hardware Server at {target_ip}:{target_port}")

        # 2. Genesis Shadow Robot 초기화 (IK 계산용)
        if urdf_path is None:
            urdf_path = "/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

        gs.init(seed=0, precision="32", logging_level="warning")
        self.scene = gs.Scene(show_viewer=False) # 시뮬레이션 창은 띄우지 않음
        self.shadow_robot = self.scene.add_entity(
            gs.morphs.URDF(file=urdf_path, fixed=True)
        )
        self.scene.build()

        self.ee_link = self.shadow_robot.get_link("gripper_link")
        self.arm_joints = [0, 1, 2, 3, 4]
        self.gripper_joint = [5]

    def get_state(self):
        """서버에서 실제 하드웨어 각도를 가져와 Shadow 로봇을 동기화합니다."""
        try:
            # 서버에 상태 요청
            self.socket.send_json({"cmd": "get_state"})
            response = self.socket.recv_json()
            
            if response["status"] == "ok":
                q_real_deg = np.array(response["q_deg"])
                q_real_rad = np.deg2rad(q_real_deg)

                # Shadow 로봇 상태 업데이트 (IK 시작점 동기화)
                self.shadow_robot.set_dofs_position(q_real_rad)
                
                # 현재 End-effector의 기하학적 상태 추출
                pos = self.ee_link.get_pos().detach().cpu().numpy()
                quat_wxyz = self.ee_link.get_quat().detach().cpu().numpy()
                quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
                rot_mat = R.from_quat(quat_xyzw).as_matrix()
                
                return {'pos': pos, 'quat': quat_xyzw, 'rot_mat': rot_mat, 'q': q_real_rad}
        except zmq.Again:
            print("🚨 [Client] Hardware Server response timeout!")
            return None

    def step(self, q_target, gripper_target):
        """IK 목표값(Radian)을 Degree로 변환하여 서버에 전송합니다."""
        if hasattr(q_target, "detach"):
            q_target = q_target.detach().cpu().numpy()
            
        # 1. Radian -> Degree 변환
        q_target_deg = np.rad2deg(q_target).tolist()
        
        # 2. 그리퍼 매핑 (0.0~0.04m 범위를 하드웨어 각도 10~90도로 매핑 예시)
        gripper_val = float(np.interp(gripper_target, [0.0, 0.04], [10, 90]))

        try:
            # 3. 서버에 제어 명령 전송
            self.socket.send_json({
                "cmd": "step",
                "q_target_deg": q_target_deg,
                "gripper_val": gripper_val
            })
            # 완료 응답 대기
            self.socket.recv_json()
        except zmq.Again:
            print("🚨 [Client] Step command timeout!")

    def disconnect(self):
        """연결 종료"""
        self.socket.close()
        self.context.term()