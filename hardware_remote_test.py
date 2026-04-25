import socket
import json
import numpy as np
import torch
import time
from envs.genesis_env import GenesisEnv # Genesis 기반 IK용
from controllers.ik_ctrl import IKController

def main():
    # 1. 도커(수신기) 주소 설정 (사용자 확인 IP)
    DOCKER_IP = '172.17.0.2' 
    DOCKER_PORT = 9999
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 2. 시뮬레이션 환경 (IK 계산용 Shadow Robot)
    print("🤖 [Sender] 가상 로봇(Shadow) 엔진 부팅 중...")
    env = GenesisEnv(show_viewer=True) 
    
    # [테스트] 실물 동기화 잠시 중단 (순수 시뮬레이션 테스트용)
    """
    print("📡 실물 로봇 상태 요청 중...")
    sock.settimeout(2.0) # 응답 대기 시간
    try:
        sock.sendto(b"GET_STATE", (DOCKER_IP, DOCKER_PORT))
        data, _ = sock.recvfrom(2048)
        real_q_deg = json.loads(data.decode())
        
        # Radians 변환 및 Genesis 세팅
        real_q_rad = np.deg2rad(real_q_deg)
        # Genesis 로봇에 실물 각도 주입
        env.robot.set_dofs_position(real_q_rad[:5], env.arm_joints)
        env.robot.set_dofs_position(np.array([real_q_rad[5]]), env.gripper_joint)
        
        # [디버그] 주입 직후 상태 확인
        initial_state = env.get_state()
        print(f"✅ 실물 동기화 완료: {np.round(real_q_deg, 1)} 도")
        print(f"📍 가상 로봇 EE 위치: {np.round(initial_state['pos'], 3)}")
        print(f"🧩 가상 로봇 EE 방향(Quat): {np.round(initial_state['quat'], 3)}")
    except Exception as e:
        print(f"⚠️ 상태 요청 실패: {e}. 기본 자세로 시작합니다.")
    finally:
        sock.settimeout(None)
    """
    print("🧪 [Pure Sim Mode] 실물 동기화 없이 기본 자세로 시작합니다.")

    controller = IKController(
        robot=env.robot, 
        ee_link=env.ee_link,
        action_scaling=0.005, 
        smoothing_alpha=0.5
    )

    # 🛡️ [안전장치] 실측 데이터 기반 관절 한계 계산 (Clipping Range)
    CALIB_DATA = {
        "shoulder_pan":  {"homing": -1843, "min": 675, "max": 3407},
        "shoulder_lift": {"homing": -980,  "min": 927, "max": 3343},
        "elbow_flex":    {"homing": 955,   "min": 1083, "max": 3303},
        "wrist_flex":    {"homing": -1477, "min": 398, "max": 2744},
        "wrist_roll":    {"homing": -949,  "min": 0, "max": 4095},
        "gripper":       {"homing": -2037, "min": 2026, "max": 3513}
    }
    JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    SAFE_LIMITS = []
    for name in JOINT_NAMES:
        c = CALIB_DATA[name]
        d_min = (c['min'] - 2048 - c['homing']) * (360.0 / 4096.0)
        d_max = (c['max'] - 2048 - c['homing']) * (360.0 / 4096.0)
        SAFE_LIMITS.append((min(-d_min, -d_max), max(-d_min, -d_max)))

    print("\n" + "="*50)
    print("🚀 실물 로봇 원격 IK 정밀 제어 모드 (Safety Enabled)")
    print(f"대상 IP: {DOCKER_IP} | 포트: {DOCKER_PORT}")
    print("명령어: 1(앞), 2(뒤), 3(좌), 4(우), 5(위), 6(아래), 0(종료)")
    print("="*50)

    try:
        while True:
            cmd = input("입력 (0~6): ")
            if cmd == '0': break
            
            # (1) 가상 액션 생성
            raw_action = np.zeros(7)
            if cmd == '1': raw_action[0] = 1.0 # 전진
            elif cmd == '2': raw_action[0] = -1.0 # 후진
            elif cmd == '3': raw_action[1] = 1.0 # 왼쪽
            elif cmd == '4': raw_action[1] = -1.0 # 오른쪽
            elif cmd == '5': raw_action[2] = 1.0 # 위
            elif cmd == '6': raw_action[2] = -1.0 # 아래
            else: continue

            # (2) 현재 가상 로봇 상태 획득
            current_state = env.get_state()

            # (3) IK 계산 (Radias 단위)
            q_target, gripper_target = controller.get_joint_targets(raw_action, current_state)

            # (4) 하드웨어용 데이터 변환 (Radian -> Degree)
            # ⭐ [수정] IK 결과에서 팔 관절 5개만 추출 (슬라이싱 적용)
            q_target_raw = q_target.detach().cpu().numpy()
            q_target_deg = np.rad2deg(q_target_raw[:5]).tolist()
            
            # 🛡️ [안전장치 1] 관절 개수 검증 (SO-100은 팔 5축)
            if len(q_target_deg) != 5:
                print(f"❌ 오류: 관절 데이터 슬라이싱 실패. (현재 {len(q_target_deg)}개)")
                continue

            # 🛡️ [안전장치 2] 하드웨어 스펙 기반 Clipping (물리적 보호)
            mapped_q_deg = [
                float(np.clip(q_target_deg[i], SAFE_LIMITS[i][0], SAFE_LIMITS[i][1]))
                for i in range(5)
            ]

            # 🛡️ [안전장치 3] 급격한 관절 이동 차단 (Delta Check)
            # 현재 가상 로봇의 각도와 목표 각도 사이의 차이가 너무 크면 위험으로 간주합니다.
            q_current_deg = np.rad2deg(current_state['q'][:5]) 
            diff = np.abs(np.array(mapped_q_deg) - q_current_deg)
            
            MAX_DELTA_DEG = 15.0 # 한 번에 최대 15도까지만 허용 (안전 제일)
            if np.any(diff > MAX_DELTA_DEG):
                max_diff_idx = np.argmax(diff)
                print(f"⚠️ 위험 감지: {max_diff_idx}번 관절의 변화량이 {diff[max_diff_idx]:.1f}도입니다. 명령을 무시합니다.")
                continue

            # (5) 그리퍼 매핑: 0~0.04m -> 5~95도
            gripper_val = float(np.interp(float(gripper_target), [0.0, 0.04], [5.0, 95.0]))
            
            # 최종 페이로드 구성 (6개 데이터)
            payload = mapped_q_deg + [gripper_val]

            # 🛡️ [안전장치 3] 전송 전 유효성 검사 (NaN/Inf 차단)
            if not np.isfinite(payload).all():
                print("❌ 오류: IK 계산 결과가 유효하지 않습니다(NaN/Inf). 범위를 벗어났을 수 있습니다!")
                continue

            # (6) 도커 브릿지로 UDP 전송
            sock.sendto(json.dumps(payload).encode(), (DOCKER_IP, DOCKER_PORT))
            print(f"📤 전송 완료 (Deg): {np.round(payload, 1)}")

            # (7) 가상 로봇 시각화 업데이트
            env.step(q_target, gripper_target)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n🛑 원격 제어를 중단합니다.")
    finally:
        sock.close()
        print("🔌 연결 종료.")

if __name__ == "__main__":
    main()
