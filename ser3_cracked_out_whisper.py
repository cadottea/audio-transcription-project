import torch
import librosa
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor
import json

# Define the model and feature extractor for emotion recognition
model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
model = AutoModelForAudioClassification.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = model.config.id2label

# Define the model and processor for transcription (Whisper or similar STT model)
stt_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
stt_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

# Function to preprocess audio for emotion recognition
def preprocess_audio(audio_path, feature_extractor, max_duration=30.0):
    audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    
    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))
    
    # Extract features for emotion recognition
    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    input_length = inputs['input_features'].shape[-1]
    if input_length < 3000:
        padding = 3000 - input_length
        inputs['input_features'] = torch.nn.functional.pad(inputs['input_features'], (0, padding))

    return inputs

# Function to predict emotion from audio
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

# Function to transcribe audio to text (using Whisper or any other STT model)
def transcribe_audio(audio_path, stt_model, stt_processor, start_time, duration):
    # Load audio segment based on start_time and duration
    audio_array, _ = librosa.load(audio_path, sr=16000, offset=start_time, duration=duration)  # Load only the specific segment
    input_values = stt_processor(audio_array, return_tensors="pt").input_values
    with torch.no_grad():
        logits = stt_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = stt_processor.decode(predicted_ids[0])
    return transcription

# Audio file path and output JSON file path
audio_path = "/Users/thor/whisper.cpp/whisper_arg/test_audio/V20230515-124213.WAV"  # Fixed path to the .wav file
output_json_path = "/Users/thor/Desktop/emotion_output.json"  # Path to save JSON on the desktop

# Process the audio in chunks and append emotions with transcriptions
duration = 2  # seconds
intervals = []
for start_time in range(0, int(librosa.get_duration(path=audio_path)), duration):
    end_time = start_time + duration
    emotion = predict_emotion(audio_path, model, feature_extractor, id2label, max_duration=duration)
    
    # Transcribe the specific audio segment
    transcription = transcribe_audio(audio_path, stt_model, stt_processor, start_time, duration)
    print(f"Predicted Emotion: {emotion}, Transcription: {transcription}")
    
    intervals.append({
        "start_time": start_time,
        "end_time": end_time,
        "emotion": emotion,
        "transcription": transcription  # Add the transcription text to the JSON output
    })

# Write to the JSON file
with open(output_json_path, 'w') as json_file:
    json.dump(intervals, json_file, indent=4)

print(f"Emotion labels and transcriptions have been appended to the JSON file at {output_json_path}")