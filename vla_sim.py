import torch
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from scipy.spatial.transform import Rotation as R
import time

# ---------------------------------------------------------
# 1. 🚨 OpenVLA 모델 최우선 로드 (CUDA 컨텍스트 충돌 원천 차단)
# Genesis(gs.init)가 PyTorch 기본 설정을 덮어쓰기 전에 반드시 먼저 로드해야 합니다.
# ---------------------------------------------------------
model_id = "openvla/openvla-7b"
print(f"[1/2] 📡 {model_id} 로드 중...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to("cuda")

# (이후에 genesis 임포트 및 초기화를 진행하여 안전성 확보)
import genesis as gs

# ---------------------------------------------------------
# 2. Genesis 엔진 초기화 및 씬 구축
# ---------------------------------------------------------
print("\n[2/2] Genesis 물리 엔진 초기화 중...")
gs.init(seed=0, precision="32", logging_level="info")

scene = gs.Scene(
    show_viewer=True,
    rigid_options=gs.options.RigidOptions(dt=0.01),
)

plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(
    gs.morphs.URDF(
        file="/home/aivlab/SO-ARM100/Simulation/SO101/so101_fixed.urdf", 
        fixed=True
    )
)

cube = scene.add_entity(
    gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.4, 0.0, 0.02)),
    surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
)

cam = scene.add_camera(
    res=(224, 224), pos=(-0.2, 0.0, 0.6), lookat=(0.4, 0.0, 0.0), fov=45
)
scene.build()

ee_link = robot.get_link("gripper_link")
arm_joints = [0, 1, 2, 3, 4, 5]
gripper_joint = [6]

# ---------------------------------------------------------
# 3. 제어 루프
# ---------------------------------------------------------
instruction = "pick up the red cube"
ACTION_SCALING = 0.05
R_cam_to_base = np.eye(3)


def run_sim():
    print("\n🚀 시뮬레이션 제어 루프 시작!")
    for i in range(1000):
        # 3.1 렌더링 및 텐서 안전 추출
        render_output = cam.render(rgb=True)
        rgb = render_output[0] if isinstance(render_output, tuple) else render_output

        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        if rgb.dtype in [np.float32, np.float64]:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

        image_pil = Image.fromarray(rgb).convert("RGB")

        # 3.2 VLA 추론
        inputs = processor(instruction, image_pil).to("cuda", dtype=torch.bfloat16)
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

        # 3.3 액션 타입 변환 (GPU Tensor -> CPU Numpy)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        delta_pos_cam = action[:3].astype(np.float32) * ACTION_SCALING
        delta_euler_local = action[3:6].astype(np.float32)
        gripper_raw = float(action[6])

        # 3.4 현재 상태 로드 (GPU Tensor -> CPU Numpy)
        current_pos = ee_link.get_pos().detach().cpu().numpy()
        current_quat = ee_link.get_quat().detach().cpu().numpy()
        current_rot = R.from_quat(current_quat).as_matrix()

        # 3.5 목표 포즈 연산
        target_pos = current_pos + (R_cam_to_base @ delta_pos_cam)
        delta_rot_mat = R.from_euler("xyz", delta_euler_local).as_matrix()
        target_rot_mat = current_rot @ delta_rot_mat
        target_quat = R.from_matrix(target_rot_mat).as_quat()

        # 3.6 IK 연산
        q_target = robot.inverse_kinematics(
            link=ee_link,
            pos=target_pos,
            quat=target_quat,
        )

        # 3.7 명령 하달 (Tensor 유지)
        robot.control_dofs_position(q_target[arm_joints], arm_joints)

        target_gripper = np.interp(gripper_raw, [-1, 1], [0.04, 0.0])
        robot.control_dofs_position(np.array([target_gripper]), gripper_joint)

        # 3.8 물리 서브스텝
        for _ in range(30):
            scene.step()

        if i % 5 == 0:
            print(
                f"Step {i:03d} | Pos Err: {np.linalg.norm(target_pos - current_pos):.4f} | Gripper: {target_gripper:.4f}"
            )


if __name__ == "__main__":
    run_sim()
