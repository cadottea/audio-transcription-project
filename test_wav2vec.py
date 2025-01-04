from transformers import pipeline

# Specify the model
model_name = "./wav2vec2-base-superb-er"
#model_name = "superb/wav2vec2-base-superb-er"

# Load the emotion classifier model using the pipeline
emotion_classifier = pipeline("audio-classification", model=model_name, device=-1)

# Test the model (this just ensures it loads)
print(f"Model {model_name} loaded successfully.")