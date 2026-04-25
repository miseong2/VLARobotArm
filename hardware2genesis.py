import genesis as gs
import pandas as pd
import numpy as np
import time

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
    robot = scene.add_entity(
        gs.morphs.URDF(file='/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf', fixed=True)
    )

    # 4. 씬 빌드
    scene.build()

    # 5. 데이터 로드
    try:
        df = pd.read_parquet("/home/aivlab/kkb_capstone/datasets/data/chunk-000/file-000.parquet")
        recorded_states = df['observation.state'].values
    except FileNotFoundError:
        print("에러: 'file-000.parquet' 파일을 찾을 수 없습니다.")
        return

    # 설정값 (30Hz)
    fps = 30
    frame_duration = 1.0 / fps

    print(f"총 {len(recorded_states)}프레임 재생 시작...")

    # 6. 재생 루프
    for i, state_deg in enumerate(recorded_states):
        loop_start = time.time()
        
        # 단위 변환 및 주입
        state_rad = np.deg2rad(state_deg)
        robot.set_qpos(state_rad)
        
        # 시뮬레이션 스텝 진행
        scene.step()

        # 실시간 동기화 딜레이
        elapsed = time.time() - loop_start
        wait_time = max(0, frame_duration - elapsed)
        time.sleep(wait_time)

    # 7. 종료 처리
    print("재생이 완료되었습니다. 3초 후 프로그램을 종료합니다.")
    
    # 마지막 프레임 자세로 3초간 대기 (중력 방지)
    last_state_rad = np.deg2rad(recorded_states[-1])
    end_hold_time = 3.0
    start_wait = time.time()
    
    while time.time() - start_wait < end_hold_time:
        robot.set_qpos(last_state_rad)
        scene.step()

    print("프로그램을 종료합니다.")
    # 루프를 빠져나오면 main()이 종료되면서 프로세스가 끝납니다.

if __name__ == "__main__":
    main()