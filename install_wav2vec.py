from transformers import AutoModelForAudioClassification

model_name = "superb/wav2vec2-base-superb-er"

# Download the model
model = AutoModelForAudioClassification.from_pretrained(model_name)

# Get model size (in MB) for reference
model_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # Approx size in MB
print(f"Model size: {model_size} MB")