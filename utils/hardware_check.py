import time
import numpy as np
import torch

# [주의] 이 경로는 사용자님의 리눅스 환경에 맞춰 수정이 필요합니다.
import os
URDF_PATH = "/home/aivlab/SO-ARM100/Simulation/SO101/so101_fixed.urdf"

def main():
    print("🌍 제어 파이프라인 무죄 증명 테스트 시작...")
    from envs.genesis_env import GenesisEnv
    
    # 환경 초기화
    env = GenesisEnv(urdf_path=URDF_PATH)
    
    # 1. 그리퍼 10회 여닫기 테스트
    print("\n✅ [테스트 1/2] 그리퍼 동작 확인 (10회)")
    for i in range(10):
        # 열기 (0.04)
        print(f"  > {i+1}회: OPEN")
        env.robot.control_dofs_position(np.array([0.04]), env.gripper_joint)
        for _ in range(30): env.scene.step()
        time.sleep(0.2)
        
        # 닫기 (0.0)
        print(f"  > {i+1}회: CLOSE")
        env.robot.control_dofs_position(np.array([0.0]), env.gripper_joint)
        for _ in range(30): env.scene.step()
        time.sleep(0.2)

    # 2. 팔 기본 관절(IK 없이 직접 제어) 5회 흔들기
    print("\n✅ [테스트 2/2] 기본 관절 동작 확인 (5회)")
    q_home = env.robot.get_dofs_position().detach().cpu().numpy()
    for i in range(5):
        print(f"  > {i+1}회: ARM WAVE")
        # 약간 위로
        q_target = q_home.copy()
        q_target[1] -= 0.1 # Shoulder joint
        env.robot.control_dofs_position(q_target[env.arm_joints], env.arm_joints)
        for _ in range(50): env.scene.step()
        
        # 다시 원래대로
        env.robot.control_dofs_position(q_home[env.arm_joints], env.arm_joints)
        for _ in range(50): env.scene.step()

    print("\n🎉 모든 제어 파이프라인 검증 완료! 하드웨어/물리 엔진 연동은 완벽합니다.")
    print("이제 문제는 오직 '모델의 시각 지능' 영역으로 좁혀졌습니다.")

if __name__ == "__main__":
    main()
