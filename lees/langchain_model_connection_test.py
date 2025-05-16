# 1. 라이브러리 임포트
import torch
from shap_e.models.download import load_model
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.diffusion.sample import sample_latents
from shap_e.util.notebooks import decode_latent_mesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로딩
model = load_model('text300M', device)
diffusion = diffusion_from_config('text300M')



# 3. 3D 생성 함수 정의
def generate_3d_obj_from_text(prompt: str, output_filename: str = "output.obj") -> str:
    batch = [dict(texts=[prompt])]

    latents = sample_latents(
        batch,
        diffusion,
        model,
        guidance_scale=15.0,
        model_kwargs_key_filter='texts',
        num_steps=64,
        device=device,
    )

    for latent in latents:
        mesh = decode_latent_mesh(model, latent).tri_mesh()
        with open(output_filename, 'wb') as f:
            mesh.write_obj(f)

    return f"3D 모델이 저장되었습니다: {output_filename}"


# 4. LangChain Runnable 생성
shape_e_runnable = RunnableLambda(lambda prompt: generate_3d_obj_from_text(prompt))

# 5. 사용 예시
result = shape_e_runnable.invoke("a futuristic flying car")
print(result)
