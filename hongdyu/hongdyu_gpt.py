from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
#from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import List
from matplotlib import pyplot as plt
from langchain_openai import ChatOpenAI
##test
load_dotenv()

# 모델 정의
#model = OllamaLLM(model="llama3.2-vision")
#model = OllamaLLM(model="llama3.2")
model = ChatOpenAI(model="gpt-4o",api_key="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")

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
        "Please spread the furniture and doors evenly across the space, coordinates, and a detailed description for each object. Your description must include outlook that is relevant to the place. You should mention the place in description. Don't include other objects, just a single object.Mention that this has simple features. You don't have to mention where it is in the place"
        "The furnitures can be more than one and it's okay to overlap"
    ),
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | model



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


query_text = ("I have provided you a grid space with scale of 30 * 30 representing a formal inside of a school. Please set the objects of the room ")

parsed_output = safe_model_invoke(query_text)

# 정상적으로 데이터를 처리한 경우에만 시각화 진행
if parsed_output is not None:
    object_names = [obj.name for obj in parsed_output.objects]
    object_descriptions = [obj.description for obj in parsed_output.objects]
    x_coords = [obj.X_coordinate for obj in parsed_output.objects]
    y_coords = [obj.Y_coordinate for obj in parsed_output.objects]

    # 시각화
    plt.figure(figsize=(30, 30))
    plt.scatter(x_coords, y_coords, marker='o', color='blue')

    # 각 포인트에 객체 이름 표시
    for i, (name, description) in enumerate(zip(object_names, object_descriptions)):
        text = f"{name}\n{description}"  # 이름과 설명을 줄바꿈으로 연결
        plt.text(x_coords[i] + 0.5, y_coords[i] + 0.5, text, fontsize=9)

    plt.title("Cathedral Objects on Grid")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle="--", alpha=0.5)
.
    plt.xlim(0, 30)
    plt.ylim(0, 30)

    plt.show()
else:
    print("객체 데이터를 가져오는데 실패하여 시각화를 생략했습니다.")
