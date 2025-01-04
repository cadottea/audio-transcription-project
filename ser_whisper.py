from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch
import numpy as np

# Define the model ID (this is the Hugging Face model name)
model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"

# Load the model and feature extractor from Hugging Face
model = AutoModelForAudioClassification.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)

# Get the id-to-label mapping for emotions
id2label = model.config.id2label

# Preprocess the audio file
def preprocess_audio(audio_path, feature_extractor, max_duration=30.0):
    # Load the audio using librosa
    audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    
    # Ensure the audio length does not exceed max_duration
    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

    # Use the feature extractor to preprocess the audio
    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs

# Predict emotion from the audio file
def predict_emotion(audio_path, model, feature_extractor, id2label, max_duration=30.0):
    # Preprocess the audio
    inputs = preprocess_audio(audio_path, feature_extractor, max_duration)
    
    # Move the model and inputs to the appropriate device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Perform emotion prediction with the model
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_id]
    
    return predicted_label

# Example usage
audio_path = "test_audio/V20230515-124213.WAV"  # Path to your audio file in the 'test_audio' directory

# Run emotion prediction
predicted_emotion = predict_emotion(audio_path, model, feature_extractor, id2label)
print(f"Predicted Emotion: {predicted_emotion}")