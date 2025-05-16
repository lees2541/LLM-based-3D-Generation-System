from diffusers import ShapEStableDiffusionPipeline
import torch

# 모델 경로 지정
model_id = "cavargas10/TRELLIS-TextoImagen3D"

# 디바이스 설정 (가능하면 GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 로드
pipe = ShapEStableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)
