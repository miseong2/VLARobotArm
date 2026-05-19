import genesis as gs
import numpy as np
import time

def main():
    # 1. Genesis 초기화 (GPU 백엔드 사용)
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
    
    # 'ㄱ'자 캘리브레이션이 반영된 URDF 로드
    robot = scene.add_entity(
        gs.morphs.URDF(file='/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf', fixed=True)
    )

    # 4. 씬 빌드
    scene.build()

    # 5. 목표 지점 및 방향 설정 (왼쪽 45도 상단)
    target_pos = np.array([0.15, 0.15, 0.25])
    target_quat = np.array([1, 0, 0, 0])

    print(f"목표 좌표 {target_pos}로의 역기구학(IK)을 계산합니다...")

    # 6. Genesis 내장 IK 솔버 호출
    q_target_tensor = robot.inverse_kinematics(
        link=robot.get_link('gripper_link'), 
        pos=target_pos,
        quat=target_quat,
    )
    
    # 🚨 수정 포인트 1: GPU 텐서를 CPU Numpy 배열로 변환
    q_target = q_target_tensor.cpu().numpy()

    # 7. 궤적 생성 (5 FPS, 10초 동안 50단계 이동)
    # 🚨 수정 포인트 2: 현재 위치도 GPU 텐서이므로 CPU로 가져옴
    q_start = robot.get_qpos().cpu().numpy() 
    
    num_steps = 50
    fps = 5
    dt = 1.0 / fps
    
    # 이제 둘 다 완벽한 Numpy 배열이므로 에러 없이 계산됩니다.
    trajectory = np.linspace(q_start, q_target, num_steps)

    print("궤적 재생 시작 (10초 소요)...")

    # 8. 재생 루프
    for i, q_step in enumerate(trajectory):
        start_time = time.time()
        
        # 로봇 관절 위치 업데이트 (Genesis는 Numpy 배열을 받으면 알아서 GPU로 다시 보냅니다)
        robot.set_qpos(q_step)
        scene.step()
        
        if (i + 1) % 5 == 0:
            print(f"진행율: {((i + 1) / num_steps) * 100:.0f}% 완료")
            
        elapsed = time.time() - start_time
        time.sleep(max(0, dt - elapsed))

    # 9. 도착 완료 후 유지 및 종료
    print("목표 지점에 도달했습니다. 3초 후 종료합니다.")
    
    end_hold_time = 3.0
    start_wait = time.time()
    
    while time.time() - start_wait < end_hold_time:
        robot.set_qpos(q_target) # Numpy 배열을 그대로 유지
        scene.step()

    print("프로그램을 종료합니다.")

if __name__ == "__main__":
    main()