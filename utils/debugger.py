import numpy as np
import cv2

class SystemDebugger:
    def __init__(self, window_name="OpenVLA Debugger"):
        self.window_name = window_name
        self.action_history = []
        
    def analyze_action(self, raw_action):
        """VLA 출력값의 수치적 이상 여부 판단"""
        # 1. NaN 또는 Inf 체크
        if np.any(np.isnan(raw_action)):
            print("🚨 [CRITICAL] VLA Output contains NaN!")
            return False
            
        # 2. 값의 범위 체크 (보통 -1 ~ 1 사이여야 함)
        if np.any(np.abs(raw_action) > 1.5): # 약간의 여유를 둠
            print(f"⚠️ [WARNING] VLA Output out of expected range: {raw_action}")
            
        # 3. 정적 상태 체크 (모델이 굳었는지 확인)
        self.action_history.append(raw_action)
        if len(self.action_history) > 20:
            self.action_history.pop(0)
            std = np.std(self.action_history, axis=0)
            if np.all(std < 1e-4):
                print("⚠️ [WARNING] Action Mode Collapse detected (Output is frozen)")
                
        return True

    def draw_debug_info(self, image, raw_action, instruction, current_step):
        """이미지 위에 디버깅 정보 오버레이"""
        # PIL Image를 numpy(BGR)로 변환
        debug_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w, _ = debug_img.shape
        
        # 1. 텍스트 정보 표시
        y_offset = 20
        cv2.putText(debug_img, f"Step: {current_step}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_img, f"Task: {instruction}", (10, y_offset + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # 2. 액션값 시각화 (막대 그래프 형태)
        for i, val in enumerate(raw_action):
            color = (0, 0, 255) if i < 3 else (255, 0, 0) # Pos: Red, Rot: Blue
            if i == 6: color = (0, 255, 255) # Gripper: Yellow
            
            bar_w = int(val * 50) # 스케일링
            cv2.rectangle(debug_img, (w//2, 150 + i*10), (w//2 + bar_w, 155 + i*10), color, -1)
            
        # 3. 방향 화살표 (Robot Base Frame 기준 시각화)
        # X: Forward (Up in image), Y: Left (Left in image)
        center = (w//2, h//2)
        arrow_scale = 100
        # X_base는 위(-Y_img), Y_base는 왼쪽(-X_img)
        end_point = (int(center[0] - raw_action[1] * arrow_scale), 
                     int(center[1] - raw_action[0] * arrow_scale))
        cv2.arrowedLine(debug_img, center, end_point, (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, debug_img)
        cv2.waitKey(1)
        
        return debug_img
