import torch
import os
import pandas as pd
import json
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh
from typing import List
from dotenv import load_dotenv

os.environ["PYTORCH3D_NAIVE_RASTERIZATION"] = "1"

# .env 로드 및 LangChain 모델 설정
load_dotenv()
model = ChatOpenAI(
    model="gpt-4o",

)

# Pydantic 모델 정의
class SetCoord(BaseModel):
    name: str = Field(description="name of the object")
    description: str = Field(description="a short detailed description of the object for 3D generation")
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
        "Please spread the furniture and doors evenly across the space, assigning unique names, coordinates, and a detailed description for each object. Your description must include outlook that is relevant to the place. You should mention the place in description. Don't include other objects, just a single object.Mention that this has simple features. You don't have to mention where it is in the place"
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
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
model_3d = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

guidance_scale = 15.0

# 결과 저장 폴더 (.obj)
output_dir = "objs"
os.makedirs(output_dir, exist_ok=True)

# 🚀 전체 파이프라인 실행
query_text = "I want a classroom layout with all necessary furniture placed on a 100 * 100 grid."
parsed_output = safe_model_invoke(query_text)

if parsed_output:
    objects_data = []

    print("✅ GPU available:", torch.cuda.is_available())
    print("🖥️ Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

    for obj in parsed_output.objects:
        print(f"🎯 Generating 3D for: {obj.name} — {obj.description}")
        objects_data.append({
            "name": obj.name,
            "description": obj.description,
            "coordinates": {
                "X_coordinate": obj.X_coordinate,
                "Y_coordinate": obj.Y_coordinate
            }
        })

        latents = sample_latents(
            batch_size=1,
            model=model_3d,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[obj.description]),
            progress=False,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        latent = latents[0]
        mesh = decode_latent_mesh(xm, latent).tri_mesh()
        filename = f"{output_dir}/{obj.name.replace(' ', '_')}.obj"
        with open(filename, 'w') as f:
            mesh.write_obj(f)
        print(f"✅ Saved .obj: {filename}")

    # Save CSV
    df = pd.DataFrame([{**d, **d.pop('coordinates')} for d in objects_data])
    df.to_csv("objects_coordinates.csv", index=False)
    print("✅ Saved coordinates to CSV")

    # Save JSON
    with open("objects_coordinates.json", "w") as json_file:
        json.dump(objects_data, json_file, indent=4)
    print("✅ Saved coordinates and descriptions to JSON")
else:
    print("❌ LangChain output failed, skipping 3D generation.")
