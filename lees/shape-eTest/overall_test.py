from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import List
from matplotlib import pyplot as plt
from langchain_openai import ChatOpenAI
import time
import re
import pandas as pd

import os




script_start = time.time()
##test
load_dotenv()

# 모델 정의
#model = OllamaLLM(model="llama3.2-vision")
#model = OllamaLLM(model="llama3.2")
model = ChatOpenAI(model="gpt-4o")



# pydantic 자료구조 정의
class SetCoord(BaseModel):
    name: str = Field(description="name of the object")
    description: str = Field(description="a short detailed description of the object for 3D generation")
    X_coordinate: float = Field(description="X coordinate of the object within given scale of a grid")
    Y_coordinate: float = Field(description="Y coordinate of the object within given scale of a grid")


class SetCoords(BaseModel):
    objects: List[SetCoord] = Field(description="List of objects with their names and coordinates")


# 출력 파서 정의
parser = PydanticOutputParser(pydantic_object=SetCoords)
format_instructions = parser.get_format_instructions()

# 프롬프트를 더욱 구체적으로 개선
prompt = PromptTemplate(
    template=(
        "You are an interior designer.\n"
        "{format_instructions}\n"
        "User Query: {query}\n"
        "Please spread the furniture and doors evenly across the space, coordinates, and a detailed description for each object. "
        "Please respond with multiple instances of common furniture (such as desks and chairs) if they typically occur "
        "multiple times in a given place environment. The maximum of the number of the common furniture is 20. "
        "Include various other given place items as well, making sure each object has unique coordinates within a given scale of a grid. "
        "Your description must include outlook that is relevant to the place and use. You should mention the place in description. "
        "Don't include other objects, just a single object. Mention that this has simple features. "
        "You don't have to mention where it is in the place."
    ),
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | model

def extract_room_scale(query_text):
    match = re.search(r'(\d+)\s*\*\s*(\d+)', query_text)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return width, height
    else:
        return 100, 100  # 기본값 (예외 처리)

def safe_model_invoke(query):
    output = chain.invoke({"query": query})

    try:
        # 정상적인 JSON 데이터 처리 시도
        parsed_output = parser.invoke(output)
        return parsed_output

    except Exception as e:
        # JSON 파싱 실패 시 처리
        print(f"모델 응답을 처리하는데 문제가 발생했습니다: {e}")
        print("모델 원본 응답:", output)
        return None


query_text = ("I have provided you a grid space with scale of 100 * 100 representing a formal inside of a school. Please set the objects of the room ")

parsed_output = safe_model_invoke(query_text)

# 정상적으로 데이터를 처리한 경우에만 시각화 진행
if parsed_output is not None:
    object_names = [obj.name for obj in parsed_output.objects]
    object_descriptions = [obj.description for obj in parsed_output.objects]
    x_coords = [obj.X_coordinate for obj in parsed_output.objects]
    y_coords = [obj.Y_coordinate for obj in parsed_output.objects]

    # 좌표 데이터프레임 생성




    df = pd.DataFrame({
        'name': object_names,
        'x': x_coords,
        'y': y_coords
    })

    os.makedirs("objs", exist_ok=True)
    output_image_path = "objs/object_placement_plot.png"

    # 그래프 생성 및 저장
    plt.figure(figsize=(10, 10))
    plt.scatter(df['x'], df['y'], s=100, color='blue')

    for i in range(len(df)):
        plt.text(df['x'][i] + 0.5, df['y'][i] + 0.5, df['name'][i],
                 fontsize=9, ha='left', va='bottom')

    plt.title("Object Placement from GPT Output")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)

    print(f"✅ 그래프 이미지 저장 완료: {output_image_path}")

import os
import sys
import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.notebooks import decode_latent_mesh

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)
shap_e_model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

from collections import OrderedDict
import re
print(object_names)
print(object_descriptions)
def normalize_name(name):
    return re.sub(r'\d+$', '', name)  # 이름 끝의 숫자 제거

# 일반화된 이름에 대한 설명 하나만 남기기
name_description_map = OrderedDict()
for name, description in zip(object_names, object_descriptions):
    norm_name = normalize_name(name)
    if norm_name not in name_description_map:
        name_description_map[norm_name] = description  # 첫 등장한 설명만 유지

# 결과 추출
normalized_names = list(name_description_map.keys())
normalized_descriptions = list(name_description_map.values())

print(normalized_names)
print(normalized_descriptions)

for i in range(len(normalized_descriptions)):
    name = normalized_names[i]  # 파일 이름용 (중복 없는 이름)
    prompt = normalized_descriptions[i]  # 텍스트 프롬프트로 사용

    print(f"Generating for: {name}")

    batch_size = 1
    guidance_scale = 15.0

    latents = sample_latents(
        batch_size=batch_size,
        model=shap_e_model,
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

    for ii, latent in enumerate(latents):
        t = decode_latent_mesh(xm, latents).tri_mesh()
        with open(f'objs/{name}.obj', 'w') as f:
            t.write_obj(f)



def parse_obj(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    vertices = []
    faces = []
    for line in lines:
        if line.startswith('v '):
            vertices.append(line.strip())
        elif line.startswith('f '):
            faces.append(line.strip())
    return vertices, faces

def translate_vertices(vertices, x_offset, y_offset):
    translated = []
    for v in vertices:
        parts = v.split()
        x, y, z = map(float, parts[1:4])
        x += x_offset
        y += y_offset
        translated.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    return translated

def shift_face_indices(faces, offset):
    shifted = []
    for face in faces:
        parts = face.split()
        new_indices = []
        for p in parts[1:]:
            idx_parts = p.split('/')
            idx_parts[0] = str(int(idx_parts[0]) + offset)
            new_indices.append('/'.join(idx_parts))
        shifted.append('f ' + ' '.join(new_indices))
    return shifted

def merge_objs(base_path, object_names, x_coords, y_coords, output_path="merged.obj"):
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for i, obj_name in enumerate(object_names):
        norm_name = normalize_name(obj_name)
        obj_path = os.path.join(base_path, f"{norm_name}.obj")
        if not os.path.isfile(obj_path):
            print(f"❌ Missing OBJ file: {obj_path}")
            continue

        vertices, faces = parse_obj(obj_path)
        translated = translate_vertices(vertices, x_coords[i], y_coords[i])
        shifted_faces = shift_face_indices(faces, vertex_offset)

        all_vertices.extend(translated)
        all_faces.extend(shifted_faces)
        vertex_offset += len(vertices)

    with open(output_path, 'w') as f:
        f.write('\n'.join(all_vertices + all_faces))
    print(f"✅ Merged .obj written to: {output_path}")

merge_objs(
    base_path='./objs',
    object_names=object_names,
    x_coords=x_coords,
    y_coords=y_coords,
    output_path='./objs/output.obj'
)

script_end = time.time()
print(f"\n🕒 전체 실행 시간: {script_end - script_start:.2f}초")