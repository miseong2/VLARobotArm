from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

def main():
    print("🔓 [Release] 모든 모터의 토크를 해제합니다. (Port: /dev/ttyACM0)")
    
    # 1. 로봇 설정 및 연결 (Port 0)
    config = SOFollowerRobotConfig(port="/dev/ttyACM0", id="my_follower", use_degrees=True)
    robot = SOFollower(config)
    
    try:
        robot.connect()
        
        # ⭐ [가장 확실한 방법] 라이브러리 내장 함수 사용
        # FeetechMotorsBus.disable_torque()는 등록된 모든 모터를 순회하며
        # Torque_Enable과 Lock 레지스터를 자동으로 해제합니다.
        print("▶ 라이브러리 내장 함수(disable_torque) 실행 중...")
        robot.bus.disable_torque()
        
        print("\n🎉 모든 모터 토크 OFF 완료! 이제 손으로 움직일 수 있습니다.")
        print("💡 참고: 전원을 아예 빼도 기어 마찰력 때문에 뻑뻑할 수 있습니다.")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        print("💡 팁: 만약 연결 실패가 뜬다면 로봇이 /dev/ttyACM1에 있는지 확인하세요.")
    finally:
        # disconnect()를 호출하되, 토크가 다시 걸리지 않도록 주의합니다.
        # LeRobot은 보통 disconnect 시점에 토크를 끄는 옵션이 켜져 있습니다.
        robot.disconnect()
        print("🔌 연결 해제됨.")

if __name__ == "__main__":
    main()
