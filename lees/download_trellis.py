from trellis.pipeline import TrellisPipeline

# 파이프라인 초기화
pipeline = TrellisPipeline(model_path="path_to_model_weights")

# 텍스트 프롬프트를 통한 3D 자산 생성
output = pipeline.generate_from_text("A futuristic flying car")

# 결과 저장 또는 시각화
output.save("output_directory")
