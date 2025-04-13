from langchain_huggingface.llms import HuggingFacePipeline
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["TRANSFORMERS_CACHE"] = "./cache/"
os.environ["HF_HOME"] = "./cache/"

hf = HuggingFacePipeline.from_model_id(
    model_id = "openai/shape-e",
    task = "text-generation",
    pipeline_kwargs={"max_new_tokens": 50}
)

response = hf("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k")