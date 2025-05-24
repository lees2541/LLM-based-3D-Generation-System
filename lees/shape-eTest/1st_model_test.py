from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import os

# 환경변수에서 OPENAI 키 로딩
load_dotenv()

# LLM 모델 정의
model = ChatOpenAI(model="gpt-4o")

# Pydantic 모델
class SetCoord(BaseModel):
    name: str = Field(description="name of the object")
    description: str = Field(description="a short detailed description of the object for 3D generation")
    X_coordinate: float = Field(description="X coordinate of the object within given scale of a grid")
    Y_coordinate: float = Field(description="Y coordinate of the object within given scale of a grid")

class SetCoords(BaseModel):
    objects: List[SetCoord] = Field(description="List of objects with their names and coordinates")

# 파서 설정
parser = PydanticOutputParser(pydantic_object=SetCoords)
format_instructions = parser.get_format_instructions()

# 프롬프트 정의 (요청한 버전)
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

# 체인 구성
chain = prompt | model

# 안전하게 GPT 호출
def safe_model_invoke(query):
    output = chain.invoke({"query": query})
    try:
        parsed_output = parser.invoke(output)
        return parsed_output
    except Exception as e:
        print(f"⚠️ 모델 응답 처리 오류: {e}")
        print("🔎 모델 원본 응답:", output)
        return None

# 쿼리 입력
query_text = "I have provided you a grid space with scale of 100 * 100 representing a formal inside of a school. Please set the objects of the room"
parsed_output = safe_model_invoke(query_text)

# 결과 처리
if parsed_output is not None:
    object_names = [obj.name for obj in parsed_output.objects]
    object_descriptions = [obj.description for obj in parsed_output.objects]
    x_coords = [obj.X_coordinate for obj in parsed_output.objects]
    y_coords = [obj.Y_coordinate for obj in parsed_output.objects]

    # 표 출력
    df = pd.DataFrame({
        "Name": object_names,
        "Description": object_descriptions,
        "X Coordinate": x_coords,
        "Y Coordinate": y_coords
    })

    print("📋 GPT로부터 받은 오브젝트 표:")
    print(df.to_string(index=False))

    # 그래프 시각화 및 저장
    plt.figure(figsize=(12, 12))
    plt.scatter(x_coords, y_coords, color='blue', s=100)

    for i, name in enumerate(object_names):
        plt.text(x_coords[i] + 1, y_coords[i] + 1, name, fontsize=8)

    plt.title("🏫 Object Placement in School (100x100 Grid)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    # 이미지 저장
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, "school_map_plot.png")
    plt.savefig(image_path, dpi=300)
    print(f"🖼️ 그래프 이미지 저장 완료: {image_path}")

    # 화면에 표시
    plt.show()
else:
    print("❌ GPT 응답에서 유효한 좌표 데이터를 가져오지 못했습니다.")
