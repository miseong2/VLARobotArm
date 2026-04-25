import time
import numpy as np
import json  # json 라이브러리 추가
from scservo_sdk import PortHandler, PacketHandler 

class SO101Hardware:
    def __init__(self, device_name='/dev/ttyACM1', baudrate=1000000):
        self.port_handler = PortHandler(device_name)
        self.packet_handler = PacketHandler(1.0)
        
        # ==========================================
        # [핵심 수정] 하드코딩 삭제하고 json 파일 동적 로드
        # ==========================================
        try:
            # my_follower.json 파일이 같은 폴더에 있다고 가정
            with open('/home/aivlab/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json', 'r') as f:
                follower_data = json.load(f)
            
            self.calib_data = {
                1: {"homing": follower_data["shoulder_pan"]["homing_offset"], "name": "shoulder_pan"},
                2: {"homing": follower_data["shoulder_lift"]["homing_offset"], "name": "shoulder_lift"},
                3: {"homing": follower_data["elbow_flex"]["homing_offset"], "name": "elbow_flex"},
                4: {"homing": follower_data["wrist_flex"]["homing_offset"], "name": "wrist_flex"},
                5: {"homing": follower_data["wrist_roll"]["homing_offset"], "name": "wrist_roll"},
                6: {"homing": follower_data["gripper"]["homing_offset"], "name": "gripper"}
            }
            print("✅ 캘리브레이션 데이터(JSON) 로드 완료!")
        except Exception as e:
            print(f"❌ JSON 로드 실패: {e}")
            return
        # ==========================================
        
        self.servo_ids = [1, 2, 3, 4, 5, 6]

        if not self.port_handler.openPort():
            print("❌ 포트를 열 수 없습니다.")
        if not self.port_handler.setBaudRate(baudrate):
            print("❌ 보드레이트 설정 실패.")

    def read_raw_positions(self):
        """모터로부터 0~4095 사이의 RAW 위치값을 읽어옵니다."""
        raw_positions = []
        for s_id in self.servo_ids:
            # 0x38은 STS3215의 현재 위치(Present Position) 주소입니다.
            pos, result, error = self.packet_handler.read2ByteTxRx(self.port_handler, s_id, 0x38)
            if result != 0:
                print(f"⚠️ ID {s_id} 읽기 실패: {self.packet_handler.getTxRxResult(result)}")
                raw_positions.append(2048) # 실패 시 중간값으로 대체
            else:
                raw_positions.append(pos)
        return raw_positions

    def get_q_deg(self):
        """RAW 값을 Degree 단위로 변환하여 리스트로 반환합니다."""
        raw_pos = self.read_raw_positions()
        q_deg = []
        
        for i, raw in enumerate(raw_pos):
            s_id = self.servo_ids[i]
            homing = self.calib_data[s_id]["homing"]
            
            # 🚨 genesis_env.py의 공식 적용: (Raw - 2048 - Homing) * 360/4096
            deg = (raw - 2048 - homing) * (360.0 / 4096.0)
            
            # URDF와 방향을 맞추기 위한 부호 반전 (-)
            q_deg.append(-deg)
            
        return q_deg

    def close(self):
        self.port_handler.closePort()

# 테스트 실행
if __name__ == "__main__":
    hw = SO101Hardware(device_name='/dev/ttyACM1')
    try:
        while True:
            current_q = hw.get_q_deg()
            print(f"📍 현재 관절각(deg): {np.round(current_q, 2)}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        hw.close()