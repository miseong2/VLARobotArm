import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from .base_agent import BaseAgent

class OpenVLAAgent(BaseAgent):
    def __init__(self, model_id="openvla/openvla-7b", device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        ).to(device)

    def predict(self, image, instruction):
        inputs = self.processor(instruction, image).to(self.device, dtype=torch.bfloat16)
        # unnorm_key="bridge_orig"는 데이터셋에 따라 달라질 수 있으므로 나중에 설정값으로 분리 가능
        action = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        return action
