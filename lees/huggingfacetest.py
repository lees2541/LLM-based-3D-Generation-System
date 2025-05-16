
from dotenv import load_dotenv

load_dotenv()
import os

# ./cache/ 경로에 다운로드 받도록 설정
os.environ["TRANSFORMERS_CACHE"] = "./cache/"
os.environ["HF_HOME"] = "./cache/"

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


# HuggingFace 모델을 다운로드 받습니다.
hf = HuggingFacePipeline.from_model_id(
    model_id="JeffreyXiang/TRELLIS-text-xlarge",  # 사용할 모델의 ID를 지정합니다.
    task="glb-generation",  # 수행할 작업을 지정합니다. 여기서는 텍스트 생성입니다.
    # 파이프라인에 전달할 추가 인자를 설정합니다. 여기서는 생성할 최대 토큰 수를 10으로 제한합니다.
    pipeline_kwargs={},
)

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "cavargas10/TRELLIS-TextoImagen3D"  # 사용할 모델의 ID를 지정합니다.
tokenizer = AutoTokenizer.from_pretrained(
    model_id
)  # 지정된 모델의 토크나이저를 로드합니다.
model = AutoModelForCausalLM.from_pretrained(model_id)  # 지정된 모델을 로드합니다.
# 텍스트 생성 파이프라인을 생성하고, 최대 생성할 새로운 토큰 수를 10으로 설정합니다.
pipe = pipeline("glb-generation", model=model,
                tokenizer=tokenizer, max_new_tokens=512)
# HuggingFacePipeline 객체를 생성하고, 생성된 파이프라인을 전달합니다.
hf = HuggingFacePipeline(pipeline=pipe)
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = """Create the requested object in glb file format. 
#Question: 
{question}

#Answer: """  # 질문과 답변 형식을 정의하는 템플릿
prompt = PromptTemplate.from_template(template)  # 템플릿을 사용하여 프롬프트 객체 생성

# 프롬프트와 언어 모델을 연결하여 체인 생성
chain = prompt | hf | StrOutputParser()

question = "create a classroom chair"  # 질문 정의

print(
    chain.invoke({"question": question})
)  # 체인을 호출하여 질문에 대한 답변 생성 및 출력