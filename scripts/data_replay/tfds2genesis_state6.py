import genesis as gs
import numpy as np
import time
import tensorflow_datasets as tfds
import os
import tensorflow as tf

def main():
    # 1. Genesis 초기화
    gs.init(backend=gs.cuda)

    # 2. 시뮬레이션 씬 생성
    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -1.0, 1.0),
            camera_lookat=(0, 0, 0.2),
        )
    )

    # 3. 환경 구축
    plane = scene.add_entity(gs.morphs.Plane())
    
    # URDF 경로 설정 (로컬 파일 사용)
    # 현재 파일 위치: capstone/src/tfds2genesis.py
    # URDF 위치: capstone/src/so101_new_calib.urdf
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, '/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf')
    
    print(f"URDF 로드 중: {urdf_path}")
    
    try:
        robot = scene.add_entity(
            gs.morphs.URDF(file=urdf_path, fixed=True)
        )
    except Exception as e:
        print(f"URDF 로드 실패: {e}")
        print("에셋(STL) 파일이 해당 경로에 있는지 확인해주세요.")
        return

    # 4. 씬 빌드
    scene.build()

    # 5. TFDS 데이터 로드
    # pickup 데이터셋 경로: capstone/tensorflow_datasets/pickup
    dataset_path = os.path.abspath(os.path.join(current_dir, '/home/aivlab/kkb_capstone/datasets/tensorflow_datasets/pickup'))
    
    print(f"TFDS 데이터셋 로드 중: {dataset_path}")
    
    try:
        # TFDS builder 로드
        builder = tfds.builder_from_directory(dataset_path)
        dataset = builder.as_dataset(split='train')
    except Exception as e:
        print(f"데이터셋 로드 실패: {e}")
        return

    print("데이터셋을 성공적으로 로드했습니다. 시각화를 시작합니다...")

    # 설정값 (데이터셋이 10fps로 전처리되었으므로 10으로 설정)
    fps = 10
    frame_duration = 1.0 / fps

    # 6. 재생 루프
    for episode_idx, episode in enumerate(dataset):
        print(f"에피소드 {episode_idx} 재생 시작...")
        
        # episode['steps']는 에피소드의 각 스텝을 포함하는 데이터셋입니다.
        steps = episode['steps']
        
        for i, step in enumerate(steps):
            loop_start = time.time()
            
            # 관찰값에서 state 추출 (6차원 관절값: [q1, q2, q3, q4, q5, gripper])
            # TFDS step에서 observation/state를 가져옵니다.
            state_deg = step['observation']['state'].numpy()
            
            # 단위 변환 (degree -> radian)
            # 이전 제어 코드(hardware2genesis.py) 기준에 따라 deg2rad 적용
            state_rad = np.deg2rad(state_deg)
            
            # Genesis 로봇의 qpos 설정
            # 로봇의 자유도(n_dofs)가 6인 경우 바로 주입 가능합니다.
            try:
                robot.set_qpos(state_rad)
            except Exception as e:
                if i == 0:
                    print(f"qpos 설정 오류: {e}")
                    print(f"로봇 DOFs: {robot.n_dofs}, 주입된 데이터 크기: {len(state_rad)}")
                break
            
            # 시뮬레이션 스텝 진행 (렌더링 업데이트)
            scene.step()

            # 실시간 동기화 딜레이
            elapsed = time.time() - loop_start
            wait_time = max(0, frame_duration - elapsed)
            time.sleep(wait_time)
            
        print(f"에피소드 {episode_idx} 재생 완료.")
        time.sleep(1.0) # 에피소드 간 대기 (확인을 위해 1초 대기)

    print("모든 에피소드 재생이 완료되었습니다.")

if __name__ == "__main__":
    main()
