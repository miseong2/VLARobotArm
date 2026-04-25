import os
import sys
import time

# [Protocol C] JAX VRAM 독점 방지 및 TensorFlow GPU 비활성화
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import grpc
import numpy as np
import cv2
import tensorflow as tf
import copy

# TensorFlow GPU 차단
tf.config.set_visible_devices([], 'GPU')

import jax
import jax.tree_util as jtu
from concurrent import futures
from octo.model.octo_model import OctoModel

current_dir = os.path.dirname(os.path.abspath(__file__))
proto_path = os.path.join(current_dir, "proto")
if proto_path not in sys.path:
    sys.path.append(proto_path)

import vla_pb2 as vla_pb2
import vla_pb2_grpc as vla_pb2_grpc

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OctoServer")

class OctoService(vla_pb2_grpc.VLAServiceServicer):
    def __init__(self, checkpoint_path=None, step=None):
        if checkpoint_path is None:
            checkpoint_path = "hf://rail-berkeley/octo-small-1.5"
            logger.warning(f"⚠️ 경고: 기본 사전 학습 모델({checkpoint_path})을 로드 중입니다.")
            self.model = OctoModel.load_pretrained(checkpoint_path)
        else:
            logger.info(f"🤖 Loading Octo model from {checkpoint_path} (Step: {step})...")
            self.model = OctoModel.load_pretrained(checkpoint_path, step=step)
            
        self.dataset_name = list(self.model.dataset_statistics.keys())[0]
        self.rng = jax.random.PRNGKey(int(time.time()))

        # 수정 없이 원본 통계값 그대로 사용
        self.action_stats = self.model.dataset_statistics[self.dataset_name]["action"]

        logger.info(f"📊 Dataset statistics loaded for: '{self.dataset_name}'")

        logger.info("✅ Octo model loaded. Starting JAX warmup...")
        self._warmup()
        logger.info("🔥 Octo model warmup complete and ready to serve.")

    def _warmup(self):
        """서버 시작 시 모델의 example_batch 뼈대를 복제하여 안전하게 웜업합니다."""
        dummy_obs = jtu.tree_map(
            lambda x: np.zeros((1,) + x.shape[1:], dtype=x.dtype), 
            self.model.example_batch['observation']
        )
        dummy_obs["timestep_pad_mask"] = np.array([[True, True]], dtype=bool)
        
        dummy_task = self.model.create_tasks(texts=["warm up"])
        self.model.sample_actions(
            dummy_obs, 
            dummy_task, 
            # 조작해둔 통계값을 적용
            unnormalization_statistics=self.action_stats,
            rng=jax.random.PRNGKey(0)
        )

    def Predict(self, request, context):
        start_time = time.time()
        try:
            # 1. 이미지 전처리 (Window=2 통일)
            images = {}
            for img_data in request.images:
                nparr = np.frombuffer(img_data.data, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    if img_data.camera_name == "wrist":
                        img_rgb = cv2.resize(img_rgb, (128, 128))
                    else:
                        img_rgb = cv2.resize(img_rgb, (256, 256))
                    images[img_data.camera_name] = np.stack([img_rgb, img_rgb], axis=0)[np.newaxis, ...]

            if not images:
                return vla_pb2.PredictResponse(actions=[0]*7, action_dim=7, chunk_size=0)

            # 2. Observation 구성
            observation = jtu.tree_map(
                lambda x: np.zeros((1,) + x.shape[1:], dtype=x.dtype), 
                self.model.example_batch['observation']
            )
            
            observation["image_primary"] = images.get("primary") or images.get("top")
            observation["image_wrist"] = images["wrist"]
            
            observation["timestep_pad_mask"] = np.array([[True, True]], dtype=bool)
            observation["pad_mask_dict"]["image_primary"] = np.array([[True, True]], dtype=bool)
            observation["pad_mask_dict"]["image_wrist"] = np.array([[True, True]], dtype=bool)
            observation["pad_mask_dict"]["timestep"] = np.array([[True, True]], dtype=bool)
            observation["timestep"] = np.array([[0, 1]], dtype=np.int32)

            if request.state and "proprio" in observation:
                state_np = np.array(request.state, dtype=np.float32)
                observation["proprio"] = np.stack([state_np, state_np], axis=0)[np.newaxis, ...]
                observation["pad_mask_dict"]["proprio"] = np.array([[True, True]], dtype=bool)

            # 3. 태스크 생성 및 추론
            task = self.model.create_tasks(texts=[request.instruction])
            
            self.rng, rng = jax.random.split(self.rng)
            actions = self.model.sample_actions(
                observation,
                task,
                # 여기서도 조작해둔 통계값을 사용합니다.
                unnormalization_statistics=self.action_stats,
                rng=rng
            )
            
            actions_np = np.array(actions)[0] 
            flat_actions = actions_np[0].tolist() 
            
            latency = (time.time() - start_time) * 1000
            logger.info(f"🚀 추론 완료 | Latency: {latency:.2f}ms | 명령: {request.instruction}")
            logger.info(f"Action stats mean: {self.action_stats['mean']}")
            logger.info(f"Action stats std:  {self.action_stats['std']}")

            return vla_pb2.PredictResponse(
                actions=flat_actions,
                action_dim=7,
                chunk_size=1
            )

        except Exception as e:
            logger.error(f"🚨 Inference Error: {e}")
            import traceback
            traceback.print_exc()
            return vla_pb2.PredictResponse(actions=[0]*7, action_dim=7, chunk_size=0)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # SELECT MODEL
    # ------------------------------------------------------------------
    # finetuned octo
    base_ckpt_path = "/home/aivlab/kkb_capstone/checkpoints4_full_only_pickup/octo_so101"
    target_step = 1999
    
    # pretraion octo
    # target_step =None
    # base_ckpt_path = "hf://rail-berkeley/octo-small-1.5"
    # -------------------------------------------------------------------

    vla_pb2_grpc.add_VLAServiceServicer_to_server(
        OctoService(checkpoint_path=base_ckpt_path, step=target_step), 
        server
    )
    
    server.add_insecure_port('[::]:50051')
    logger.info("🚀 Octo VLA Server started on port 50051")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()