import genesis as gs
import numpy as np
import time
import tensorflow_datasets as tfds
import os
import torch
from scipy.spatial.transform import Rotation as R

def main():
    # 1. Genesis 초기화
    gs.init(backend=gs.cuda)

    # 2. 시뮬레이션 씬 생성
    scene = gs.Scene(
        show_viewer=True,
        # rigid_options=gs.options.RigidOptions(
        #     gravity=(0.0, 0.0, 0.0),  # 중력 비활성화 (테스트용)
        # ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -1.0, 1.0),
            camera_lookat=(0, 0, 0.2),
        )
    )

    # 3. 환경 및 로봇 구축
    plane = scene.add_entity(gs.morphs.Plane())
    
    # URDF 및 데이터셋 경로 설정
    urdf_path = '/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf'
    dataset_path = '/home/aivlab/kkb_capstone/datasets/tensorflow_datasets/pickup'
    
    print(f"URDF 로드 중: {urdf_path}")
    try:
        robot = scene.add_entity(gs.morphs.URDF(file=urdf_path, fixed=True))
    except Exception as e:
        print(f"URDF 로드 실패: {e}")
        return

    # 4. 씬 빌드
    scene.build()
    
    # 엔드 이펙터 링크 설정 
    ee_link_name = "gripper_link"
    try:
        ee_link = robot.get_link(ee_link_name)
    except Exception as e:
        print(f"오류: URDF에서 '{ee_link_name}' 링크를 찾을 수 없습니다.")
        return

    # 5. TFDS 데이터 로드
    try:
        builder = tfds.builder_from_directory(dataset_path)
        dataset = builder.as_dataset(split='train')
    except Exception as e:
        print(f"데이터셋 로드 실패: {e}")
        return
    
    print("데이터셋 로드 완료. 시뮬레이션을 시작합니다...")

    fps = 10
    frame_duration = 1.0 / fps

    # 6. 제어 재생 루프
    for episode_idx, episode in enumerate(dataset):
        print(f"========== 에피소드 {episode_idx} 재생 시작 ==========")
        steps = episode['steps']
        
        # --- [초기화] 에피소드 시작 시 Home State 설정 ---
        try:
            first_step = next(iter(steps))
            home_state_deg = first_step['observation']['state'].numpy()
            home_state_rad = np.deg2rad(home_state_deg)
            
            # 로봇을 홈 포지션으로 즉시 이동 (초기화)
            robot.set_qpos(home_state_rad)
            scene.step() # 물리 엔진 업데이트하여 EE 위치 갱신
            print(f"Home State 초기화 완료: {np.round(home_state_deg, 2)}")
            
            # ==========================================
            # [고스트 타겟 적용 1] 초기 수학적 위치(Ghost) 기록
            # ==========================================
            ghost_pos = ee_link.get_pos().cpu().numpy()
            
            curr_quat_wxyz = ee_link.get_quat().cpu().numpy()
            ghost_quat_xyzw = [curr_quat_wxyz[1], curr_quat_wxyz[2], curr_quat_wxyz[3], curr_quat_wxyz[0]]
            ghost_rot_mat = R.from_quat(ghost_quat_xyzw).as_matrix()
            
            # IK 다중 해(Flipping) 방지를 위해 초기 관절 각도 메모리 저장
            ghost_qpos = robot.get_qpos() 
            
            time.sleep(0.5) 
        except Exception as e:
            print(f"초기화 실패: {e}")
            break

        # --- [실행] 각 스텝별 델타 제어 ---
        for i, step in enumerate(steps):
            loop_start = time.time()
            
            # (기존 코드 삭제: ee_link.get_pos() 등 현실의 위치를 더 이상 보지 않습니다!)
            
            # 1. 데이터셋에서 Delta Action 추출
            action = step['action'].numpy()
            d_pos = action[:3]      
            d_euler = action[3:6]   
            gripper_cmd = action[6] 
            
            # ==========================================
            # [고스트 타겟 적용 2] 이론적 목표점 누적 연산
            # ==========================================
            ghost_pos = ghost_pos + d_pos
            
            d_rot_mat = R.from_euler('xyz', d_euler).as_matrix()
            ghost_rot_mat = ghost_rot_mat @ d_rot_mat
            
            # Genesis IK용 쿼터니언 변환 (xyzw -> wxyz)
            t_quat_xyzw = R.from_matrix(ghost_rot_mat).as_quat()
            ghost_quat_wxyz = np.array([t_quat_xyzw[3], t_quat_xyzw[0], t_quat_xyzw[1], t_quat_xyzw[2]])

            # 2. 역기구학(IK) 계산
            try:
                # ==========================================
                # [고스트 타겟 적용 3] 가상의 목표점과 이전 관절 각도 주입
                # ==========================================
                q_target_tensor = robot.inverse_kinematics(
                    link=ee_link,
                    pos=ghost_pos,          # 실제 위치(curr_pos)가 아닌 고스트 타겟 주입
                    quat=ghost_quat_wxyz,
                    init_qpos=ghost_qpos    # 실제 관절이 아닌 방금 전 계산한 고스트 관절 주입 (Flipping 방지)
                )
                
                # 다음 스텝 계산을 위해 이번에 푼 해(관절 각도)를 고스트 메모리에 저장
                ghost_qpos = q_target_tensor
                
                # CUDA Tensor를 NumPy로 변환
                q_target = q_target_tensor.cpu().numpy()
                q_target[-1] = gripper_cmd 
                
                # 3. 물리 엔진 제어 명령 전달 (PID 모터들이 알아서 고스트를 쫓아감)
                robot.control_dofs_position(q_target)
                
            except Exception as e:
                print(f"IK/제어 오류 (Step {i}): {e}")
                break
            
            # 시뮬레이션 진행
            scene.step()

            # 시간 동기화
            elapsed = time.time() - loop_start
            wait_time = max(0, frame_duration - elapsed)
            time.sleep(wait_time)
            
        print(f"에피소드 {episode_idx} 완료.")
        time.sleep(1.0) 

if __name__ == "__main__":
    main()