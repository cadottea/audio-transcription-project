from transformers import pipeline

if __name__ == "__main__":
    # Force CPU usage with `device=-1`
    audio_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=-1)

    # Test with your audio file
    audio_path = "test_audio/V20230515-124213.WAV"
    audio_result = audio_classifier(audio_path)

    # Print the result
    print(audio_result)