from huggingface_hub import login

login(token="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")

from transformers import AutoTokenizer, AutoModel

# 모델 로딩
tokenizer = AutoTokenizer.from_pretrained("JeffreyXiang/TRELLIS-text-xlarge")
model = AutoModel.from_pretrained("JeffreyXiang/TRELLIS-text-xlarge")

# 입력 텍스트
prompt = "a wooden chair with four legs"

# 텍스트 → latent
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model(**inputs)
latent = outputs.last_hidden_state  # 또는 필요한 latent representation
# Hugging Face에서 발급받은 토큰 입력
