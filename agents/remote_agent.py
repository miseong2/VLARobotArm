import grpc
import numpy as np
import cv2
from PIL import Image
from .base_agent import BaseAgent
import proto.vla_pb2 as vla_pb2
import proto.vla_pb2_grpc as vla_pb2_grpc

class RemoteAgent(BaseAgent):
    def __init__(self, target_ip="localhost", target_port=50051):
        """gRPC 서버에 연결하는 원격 에이전트"""
        self.channel = grpc.insecure_channel(f"{target_ip}:{target_port}")
        self.stub = vla_pb2_grpc.VLAServiceStub(self.channel)
        print(f"📡 Remote Agent connected to {target_ip}:{target_port}")

    def predict(self, image, instruction, wrist_image=None, state=None):
        """
        image: PIL Image (Primary)
        instruction: str
        wrist_image: PIL Image (Optional, Wrist)
        state: list of floats (Optional, Robot Joint Positions)
        """
        # 1. 이미지 압축
        def encode_image(pil_img):
            img_np = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            return buffer.tobytes()

        # 2. gRPC 요청 구성 (top/wrist 이름 명시적 사용)
        image_list = [
            vla_pb2.Image(data=encode_image(image), camera_name="top")
        ]
        
        if wrist_image:
            image_list.append(vla_pb2.Image(data=encode_image(wrist_image), camera_name="wrist"))

        request = vla_pb2.PredictRequest(
            images=image_list,
            instruction=instruction,
            state=state if state is not None else []
        )

        # 3. RPC 호출
        try:
            response = self.stub.Predict(request)
            return np.array(response.actions, dtype=np.float32)
        except grpc.RpcError as e:
            print(f"🚨 gRPC Error: {e.code()} - {e.details()}")
            return np.zeros(7, dtype=np.float32) # 에러 시 정지 명령 반환

    def __del__(self):
        if hasattr(self, 'channel'):
            self.channel.close()
