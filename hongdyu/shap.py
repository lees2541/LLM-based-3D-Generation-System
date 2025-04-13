import torch
import os
import imageio
import matplotlib.pyplot as plt

from test import name
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images

# gif_widget, display는 제거됨

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

# 샘플링 설정
batch_size = 1
guidance_scale = 15.0
prompt = name

latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
)

# 렌더링 설정
render_mode = 'nerf'
size = 64
cameras = create_pan_cameras(size, device)

# 출력 디렉토리
os.makedirs("renders", exist_ok=True)


# 이미지 시각화 및 저장 함수
def save_gif(images, filename="renders/render.gif", duration=0.1):
    imageio.mimsave(filename, images, duration=duration)
    print(f"✅ GIF saved to: {filename}")


def preview_images(images, max_preview=5):
    for i, img in enumerate(images[:max_preview]):
        plt.imshow(img)
        plt.title(f"Preview Frame {i + 1}")
        plt.axis("off")
        plt.show()


# 렌더링 및 결과 시각화
for i, latent in enumerate(latents):
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)

    # 일부 이미지 미리보기
    preview_images(images)

    # 전체 이미지 시퀀스를 GIF로 저장
    save_gif(images, filename=f"renders/render_{i}.gif")
