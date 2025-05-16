from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
#from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import List
from matplotlib import pyplot as plt
from langchain_openai import ChatOpenAI
import son
##test
load_dotenv()

# 모델 정의
#model = OllamaLLM(model="llama3.2-vision")
#model = OllamaLLM(model="llama3.2")
model = ChatOpenAI(model="gpt-4o",api_key="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")

# pydantic 자료구조 정의
class SetCoord(BaseModel):
    name: str = Field(description="name of the object")
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
        "You are a interior designer\n"
        "{format_instructions}\n"
        "User Query: {query}\nPlease set the furniture evenly spread placed in the given scale of 2D grid."
        "Doors should be included.\n"
        "Please respond with multiple instances of common furniture (such as desks and chairs) if they typically occur "
        "several times in a given place environment. Include various other given place items as well, making sure each "
        "object has unique coordinates within a given scale of a grid. Provide the response strictly in JSON format."
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


query_text = ("I have provided you a grid space with scale of 100 * 100 representing a formal inside of a classroom. Please set the objects of the room ")

parsed_output = safe_model_invoke(query_text)

# 정상적으로 데이터를 처리한 경우에만 시각화 진행
if parsed_output is not None:
    object_names = [obj.name for obj in parsed_output.objects]
    x_coords = [obj.X_coordinate for obj in parsed_output.objects]
    y_coords = [obj.Y_coordinate for obj in parsed_output.objects]

    # 시각화
    plt.figure(figsize=(10, 10))
    plt.scatter(x_coords, y_coords, marker='o', color='blue')

    # 각 포인트에 객체 이름 표시
    for i, name in enumerate(object_names):
        plt.text(x_coords[i] + 0.5, y_coords[i] + 0.5, name, fontsize=9)

    plt.title("Cathedral Objects on Grid")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.xlim(0, 100)
    plt.ylim(0, 60)

    plt.show()
else:
    print("객체 데이터를 가져오는데 실패하여 시각화를 생략했습니다.")
