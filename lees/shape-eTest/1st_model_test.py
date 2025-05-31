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
model = ChatOpenAI(model="gpt-4o",api_key="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")

# Pydantic 모델
class SetCoord(BaseModel):
    name: str = Field(description="name of the object")
    description: str = Field(description="Your description must include outlook that is relevant to the place. You should mention the place in description. Don't include other objects, just a single object.Mention that this has simple features. You don't have to mention where it is in the place")
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
        "You are a professional interior designer.\n"
        "You design realistic furniture layouts for various types of rooms based on their intended function (e.g., classroom, office, bedroom, etc).\n"
        "{format_instructions}\n"
        "User Query: {query}\n"
        "\n"
        "Each room is represented as a 2D grid (for example, 30 * 30).\n"
        "If scale of the room is not specified, choose the most formal size in meters.\n"
        "Your task is to place appropriate furniture in the space, considering the room’s function, layout logic, and human usability.\n"
        "\n"
        "Design constraints:\n"
        "- Do not place objects using mathematical patterns such as diagonal lines (x = y), perfect vertical or horizontal symmetry, or arithmetic sequences.\n"
        "- Avoid placing all objects in one corner or clustered area.\n"
        "- Avoid using completely random coordinates.\n"
        "- Spread objects in reasonable position"
        "\n"
        "Spatial logic examples:\n"
        "- Desks in classrooms are typically arranged in rows or clusters.\n"
        "- Beds in bedrooms are placed against walls.\n"
        "- Whiteboards are usually placed on one main wall, opposite to student seating.\n"
        "- Doors should be located at logical entrances, usually along the edges or corners.\n"
        "\n"
        "🪑 Furniture placement rules:\n"
        "- Include multiple instances of common objects (e.g., desks, chairs), up to 20 per type.\n"
        "- Each object must have:\n"
        "  • A unique name\n"
        "  • Distinct X and Y coordinates within the grid\n"
        "  • A short, place-relevant description (mentioning the room type and the object’s appearance)\n"
        "- Doors must be included and positioned near walls or entrances.\n"
        "\n"
        "Use spatial logic instead of randomness or symmetry. The layout should look intentional and usable.\n"
        "\n"
        "After generating the layout, review and revise it if the objects are in appropriate place.\n"
        "Return the final layout strictly in JSON format as instructed above."
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
        print("🔎 모델 원본 응답:", output)
        parsed_output = parser.invoke(output)
        return parsed_output
    except Exception as e:
        print(f"⚠️ 모델 응답 처리 오류: {e}")
        print("🔎 모델 원본 응답:", output)
        return None

# 쿼리 입력
query_text = " I want you to create inside of cathedral. Please set the objects of the room"
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
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    # 이미지 저장
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, "school_map_plot.png")
    plt.savefig(image_path, dpi=300)
    print(f"🖼️ 그래프 이미지 저장 완료: {image_path}")

    # JSON 저장
    json_output_path = os.path.join(output_dir, "table_based_output.json")
    df.to_json(json_output_path, orient="records", force_ascii=False, indent=2)
    print(f"📄 표 기반 JSON 저장 완료: {json_output_path}")

    # 화면에 표시
    plt.show()

else:
    print("❌ GPT 응답에서 유효한 좌표 데이터를 가져오지 못했습니다.")
