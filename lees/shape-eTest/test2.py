# 통합 파일: full_pipeline.py

import torch
import os
import imageio
import matplotlib.pyplot as plt
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images
from typing import List
from dotenv import load_dotenv

# .env 로드 및 LangChain 모델 설정
load_dotenv()
model = ChatOpenAI(model="gpt-4o",api_key="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")

# Pydantic 모델 정의
class SetCoord(BaseModel):
    name: str = Field(description="name of the object")
    X_coordinate: float
    Y_coordinate: float

class SetCoords(BaseModel):
    objects: List[SetCoord]

parser = PydanticOutputParser(pydantic_object=SetCoords)
format_instructions = parser.get_format_instructions()

# 프롬프트 정의
prompt = PromptTemplate(
    template=(
        "You are an interior designer.\n"
        "{format_instructions}\n"
        "User Query: {query}\n"
        "Please spread the furniture and doors evenly across the space, assigning unique names and coordinates."
    ),
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | model

def safe_model_invoke(query: str):
    output = chain.invoke({"query": query})
    try:
        return parser.invoke(output)
    except Exception as e:
        print(f"❌ Parsing Error: {e}")
        print("Raw Output:", output)
        return None

# Shape-E 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
model_3d = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

guidance_scale = 15.0
render_mode = 'nerf'
size = 64
cameras = create_pan_cameras(size, device)

# 결과 저장 폴더
os.makedirs("renders", exist_ok=True)

def save_gif(images, filename):
    imageio.mimsave(filename, images, duration=0.1)
    print(f"✅ Saved: {filename}")

# 🚀 전체 파이프라인 실행
query_text = "I want a classroom layout with all necessary furniture placed on a 100 * 100 grid."
parsed_output = safe_model_invoke(query_text)

if parsed_output:
    prompts = [obj.name for obj in parsed_output.objects]
    print("✅ GPU available:", torch.cuda.is_available())
    print("🖥️ Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
    print("🎯 Generating 3D for:", prompts)

    latents = sample_latents(
        batch_size=len(prompts),
        model=model_3d,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=prompts),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    for i, latent in enumerate(latents):
        images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        save_gif(images, filename=f"renders/{prompts[i].replace(' ', '_')}.gif")

else:
    print("❌ LangChain output failed, skipping 3D generation.")
