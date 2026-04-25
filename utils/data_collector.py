import cv2
import os
import json
import time
import threading
from datetime import datetime
from queue import Queue

class DataCollector:
    def __init__(self, base_path="data", top_cam_id=0, wrist_cam_id=1):
        """
        base_path: 데이터가 저장될 루트 폴더
        top_cam_id: 탑뷰 카메라 (OpenVLA/Octo Primary)
        wrist_cam_id: 손목 카메라 (Octo Wrist)
        """
        self.base_path = base_path
        self.is_recording = False
        self.step_count = 0
        self.episode_path = ""
        self.save_queue = Queue()
        
        # 1. 카메라 초기화 (고화질 설정)
        print(f"📷 Initializing Cameras... (Top: {top_cam_id}, Wrist: {wrist_cam_id})")
        self.cap_top = cv2.VideoCapture(top_cam_id)
        self.cap_wrist = cv2.VideoCapture(wrist_cam_id)
        
        for cap in [self.cap_top, self.cap_wrist]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # 포커스 고정

        # 2. 백그라운드 저장 스레드 시작 (I/O 병목 방지)
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        
        self.metadata = []

    def start_episode(self, instruction):
        """새 에피소드 시작 (폴더 생성)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_path = os.path.join(self.base_path, f"ep_{timestamp}")
        
        # 폴더 생성
        os.makedirs(os.path.join(self.episode_path, "top"), exist_ok=True)
        os.makedirs(os.path.join(self.episode_path, "wrist"), exist_ok=True)
        
        self.instruction = instruction
        self.step_count = 0
        self.metadata = []
        self.is_recording = True
        print(f"🎬 [START] {self.episode_path} | Task: {instruction}")

    def collect_step(self, action, state):
        """한 스텝의 이미지와 데이터를 큐에 삽입 (Dual-Grab 동기화)"""
        if not self.is_recording: return

        ts = time.time()
        step_id = f"{self.step_count:06d}"

        # ⭐ Dual-Grab: 노출 시간을 최대한 일치시킴
        self.cap_top.grab()
        self.cap_wrist.grab()

        # 이미지 데이터 가져오기 (메모리로 복사)
        ret_t, img_top = self.cap_top.retrieve()
        ret_w, img_wrist = self.cap_wrist.retrieve()

        if not ret_t or not ret_w:
            print(f"⚠️ Step {step_id}: Camera frame drop!")
            return

        # 저장 큐에 데이터 투척 (루프 지연 방지)
        self.save_queue.put({
            'path_top': os.path.join(self.episode_path, "top", f"{step_id}.jpg"),
            'path_wrist': os.path.join(self.episode_path, "wrist", f"{step_id}.jpg"),
            'img_top': img_top.copy(),
            'img_wrist': img_wrist.copy()
        })

        # 메타데이터 기록
        self.metadata.append({
            "step_id": step_id,
            "timestamp": ts,
            "instruction": self.instruction,
            "action": action.tolist() if hasattr(action, 'tolist') else action,
            "state": state.tolist() if hasattr(state, 'tolist') else state,
            "rel_path_top": f"top/{step_id}.jpg",
            "rel_path_wrist": f"wrist/{step_id}.jpg"
        })
        
        self.step_count += 1

    def stop_episode(self):
        """에피소드 종료 및 메타데이터 저장"""
        if not self.is_recording: return
        
        self.is_recording = False
        # 메타데이터 저장
        meta_path = os.path.join(self.episode_path, "metadata.json")
        with open(meta_path, "w", encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=4, ensure_ascii=False)
            
        print(f"✅ [SAVED] Total Steps: {self.step_count} | Path: {self.episode_path}")

    def _save_worker(self):
        """백그라운드에서 이미지를 디스크에 기록 (JPEG Quality 95)"""
        while True:
            data = self.save_queue.get()
            if data is None: break
            
            # 고화질 저장을 위해 압축률 최소화
            cv2.imwrite(data['path_top'], data['img_top'], [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(data['path_wrist'], data['img_wrist'], [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            self.save_queue.task_done()

    def __del__(self):
        if hasattr(self, 'cap_top'): self.cap_top.release()
        if hasattr(self, 'cap_wrist'): self.cap_wrist.release()
        print("🔌 Cameras released.")
