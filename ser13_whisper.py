import os
import shutil
import subprocess
import sys
from split3 import split_audio  # Import the split function from split3.py
import librosa
import torch
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from model_selection import select_model

# Define the model and feature extractor for emotion recognition
emotion_model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
emotion_model = AutoModelForAudioClassification.from_pretrained(emotion_model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(emotion_model_id, do_normalize=True)
id2label = emotion_model.config.id2label

# Function to preprocess audio for emotion recognition
def preprocess_audio(audio_path, feature_extractor, max_duration=30.0):
    audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))
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

# Function to run Whisper for transcription
def run_whisper_with_emotion(audio_file, output_file_path, model):
    if not os.path.isfile(audio_file):
        print(f"Error: Audio file '{audio_file}' does not exist.")
        sys.exit(1)

    command = [
        "../build/bin/main", "-m", model, "-f", audio_file, "--output-txt",
        "--split-on-word", "--max-len", "50", "--word-thold", "0.5",
    ]

    try:
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        transcription_output = result.stdout.splitlines()

        # Output transcription with emotion labels
        with open(output_file_path, 'a') as file:  # Append to the single file for this .wav
            for line in transcription_output:
                if '-->' in line:
                    timestamp, text = line.split(']', 1)
                    audio_chunk_file = audio_file  # Reference the current audio chunk
                    emotion = predict_emotion(audio_chunk_file, emotion_model, feature_extractor, id2label)
                    formatted_line = f"{timestamp}] {text.strip()} [{emotion}]"
                    print(formatted_line)  # Output to terminal
                    file.write(formatted_line + '\n')
                else:
                    print(line)  # Output non-timestamped lines to terminal
                    file.write(line + '\n')
        print(f"Transcription with emotions saved to: {output_file_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error: Whisper.cpp failed with return code {e.returncode}")
        sys.exit(1)

# Process audio in chunks, transcribe, and analyze emotion
def process_audio_chunks(audio_file, model, transcribe_dir, output_dir):
    split_audio_dir = os.path.join(transcribe_dir, "temp_split")
    if os.path.exists(split_audio_dir):
        print(f"Deleting existing temp directory: {split_audio_dir}")
        shutil.rmtree(split_audio_dir)
    split_audio(audio_file, split_audio_dir)

    audio_basename = os.path.splitext(os.path.basename(audio_file))[0]
    output_folder = os.path.join(output_dir, audio_basename)
    os.makedirs(output_folder, exist_ok=True)

    output_file_path = os.path.join(output_folder, f"{audio_basename}.txt")

    for split_file in sorted(os.listdir(split_audio_dir)):
        split_file_path = os.path.join(split_audio_dir, split_file)
        if os.path.isfile(split_file_path):
            run_whisper_with_emotion(split_file_path, output_file_path, model)

    return output_folder

def main():
    desktop = os.path.expanduser("~/Desktop")
    transcribe_dir = os.path.join(desktop, "transcribe")
    output_dir = os.path.join(desktop, "output")
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(transcribe_dir)
    audio_files = [os.path.join(transcribe_dir, f) for f in files if f.lower().endswith(".wav")]

    if not audio_files:
        print("No .wav files found in the 'transcribe' directory.")
        exit(1)

    model = select_model()

    for audio_file in audio_files:
        process_audio_chunks(audio_file, model, transcribe_dir, output_dir)

if __name__ == "__main__":
    main()