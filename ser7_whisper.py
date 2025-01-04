import os
import subprocess
import sys
import json
import torch
import librosa
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from split import split_audio  # Import the split function
from model_selection import select_model  # Import model selection

# Define the model and feature extractor for emotion recognition
emotion_model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
emotion_model = AutoModelForAudioClassification.from_pretrained(emotion_model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(emotion_model_id, do_normalize=True)
id2label = emotion_model.config.id2label

# Function to preprocess audio for emotion recognition
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

# Function to run Whisper for transcription (same as batch.py)
def run_whisper(audio_file, output_dir, model):
    if not os.path.isfile(audio_file):
        print(f"Error: Audio file '{audio_file}' does not exist.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    audio_basename = os.path.splitext(os.path.basename(audio_file))[0]
    txt_output_file = os.path.join(output_dir, f"{audio_basename}.txt")

    command = [
        "../build/bin/main", "-m", model, "-f", audio_file, "--output-txt",
        "--split-on-word", "--max-len", "50", "--word-thold", "0.5", "--print-colors"
    ]

    try:
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        colored_output = result.stdout

        with open(txt_output_file, 'w') as file:
            file.write(colored_output)
        print(f"Transcription saved to: {txt_output_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error: Whisper.cpp failed with return code {e.returncode}")
        sys.exit(1)

    return txt_output_file

# Process audio in chunks, transcribe, and analyze emotion
def process_audio_chunks(audio_file, model, output_dir):
    duration = 2  # duration of each chunk in seconds
    intervals = []

    # Transcribe and process each chunk
    for start_time in range(0, int(librosa.get_duration(path=audio_file)), duration):
        end_time = start_time + duration

        # Run Whisper for transcription (if necessary for this chunk)
        transcription_file = run_whisper(audio_file, output_dir, model)
        
        # Predict emotion for the current audio chunk
        emotion = predict_emotion(audio_file, emotion_model, feature_extractor, id2label, max_duration=duration)
        print(f"Predicted Emotion: {emotion}")

        intervals.append({
            "start_time": start_time,
            "end_time": end_time,
            "emotion": emotion,
            "transcription": transcription_file
        })

    return intervals

def main():
    input_audio_path = "/Users/thor/whisper.cpp/whisper_arg/test_audio/V20230515-124213.WAV"  # Path to input audio file
    output_dir = "/Users/thor/Desktop/output"  # Folder to save outputs

    # Select model using the previous method
    model = select_model()

    # Split the input audio into 5-second chunks and save to 'temp_split'
    split_audio(input_audio_path, '/Users/thor/Desktop/temp_split', segment_duration=5)

    # Process the split audio files in 'temp_split'
    intervals = []
    temp_split_dir = '/Users/thor/Desktop/temp_split'
    for file_name in os.listdir(temp_split_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(temp_split_dir, file_name)
            intervals.extend(process_audio_chunks(file_path, model, output_dir))

    # Save results to JSON
    output_json_path = os.path.join(output_dir, "emotion_output.json")
    with open(output_json_path, 'w') as json_file:
        json.dump(intervals, json_file, indent=4)

    print(f"Emotion labels and transcriptions have been appended to the JSON file at {output_json_path}")

    # Clean up temporary split files
    for file_name in os.listdir(temp_split_dir):
        file_path = os.path.join(temp_split_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print(f"Temporary files in {temp_split_dir} have been cleaned up.")

if __name__ == "__main__":
    main()