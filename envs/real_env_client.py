import zmq
import numpy as np
import genesis as gs
from scipy.spatial.transform import Rotation as R

class RealRobotEnvClient:
    def __init__(self, target_ip="localhost", target_port=5555, urdf_path=None):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{target_ip}:{target_port}")
        self.socket.setsockopt(zmq.RCVTIMEO, 5000) 
        
        print(f"📡 [Client] Connected to Hardware Server at {target_ip}:{target_port}")

        if urdf_path is None:
            urdf_path = "/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

        gs.init(seed=0, precision="32", logging_level="warning")
        self.scene = gs.Scene(show_viewer=False) 
        self.shadow_robot = self.scene.add_entity(
            gs.morphs.URDF(file=urdf_path, fixed=True)
        )
        self.scene.build()
        self.ee_link = self.shadow_robot.get_link("gripper_link")

    def get_state(self):
        try:
            self.socket.send_json({"cmd": "get_state"})
            response = self.socket.recv_json()
            
            if response["status"] == "ok":
                q_real_deg = np.array(response["q_deg"])
                q_real_rad = np.deg2rad(q_real_deg)

                self.shadow_robot.set_dofs_position(q_real_rad)
                
                pos = self.ee_link.get_pos().detach().cpu().numpy()
                quat_wxyz = self.ee_link.get_quat().detach().cpu().numpy()
                quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
                rot_mat = R.from_quat(quat_xyzw).as_matrix()
                
                return {'pos': pos, 'quat': quat_xyzw, 'rot_mat': rot_mat, 'q': q_real_rad}
        except zmq.Again:
            print("🚨 [Client] Timeout!")
            return None

    def step(self, q_target, gripper_target):
        if hasattr(q_target, "detach"):
            q_target = q_target.detach().cpu().numpy()
            
        # [수정] 모든 관절을 Radian -> Degree로 직접 변환 (연속 제어 대응)
        q_target_deg = np.rad2deg(q_target).tolist()
        gripper_val = float(np.rad2deg(gripper_target))

        try:
            self.socket.send_json({
                "cmd": "step",
                "q_target_deg": q_target_deg,
                "gripper_val": gripper_val
            })
            self.socket.recv_json()
        except zmq.Again:
            print("🚨 [Client] Step Timeout!")

    def disconnect(self):
        self.socket.close()
        self.context.term()