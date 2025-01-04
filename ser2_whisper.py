import torch
import librosa
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# Define the model and feature extractor
model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
model = AutoModelForAudioClassification.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = model.config.id2label

# Function to preprocess audio
def preprocess_audio(audio_path, feature_extractor, max_duration=30.0):
    audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    
    # Max length for the input (in terms of samples)
    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))
    
    # Extract features using the feature extractor
    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    # Check the length of input features and pad to the required length (3000)
    input_length = inputs['input_features'].shape[-1]
    if input_length < 3000:
        padding = 3000 - input_length
        inputs['input_features'] = torch.nn.functional.pad(inputs['input_features'], (0, padding))

    return inputs

# Predict emotion
def predict_emotion(audio_path, model, feature_extractor, id2label, max_duration=30.0):
    inputs = preprocess_audio(audio_path, feature_extractor, max_duration)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_id]
    
    return predicted_label

# Audio file path and output JSON file path
audio_path = "/Users/thor/whisper.cpp/whisper_arg/test_audio/V20230515-124213.WAV"  # Fixed path to the .wav file
output_json_path = "/Users/thor/Desktop/emotion_output.json"  # Path to save JSON on the desktop

# Process the audio in chunks and append emotions
duration = 2  # seconds
intervals = []
for start_time in range(0, int(librosa.get_duration(path=audio_path)), duration):
    end_time = start_time + duration
    emotion = predict_emotion(audio_path, model, feature_extractor, id2label, max_duration=duration)
    print(f"Predicted Emotion: {emotion}")
    intervals.append({
        "start_time": start_time,
        "end_time": end_time,
        "emotion": emotion
    })

# Write to the JSON file
with open(output_json_path, 'w') as json_file:
    json.dump(intervals, json_file, indent=4)

print(f"Emotion labels have been appended to the JSON file at {output_json_path}")