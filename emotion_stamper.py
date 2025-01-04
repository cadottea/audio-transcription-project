import json
from transformers import pipeline

# Load the pre-trained Wav2Vec2 model for emotion classification
emotion_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=-1)

def add_emotion_to_transcription(json_file):
    """
    Adds emotion stamps to the .json file by analyzing each transcription line.
    
    Args:
        json_file (str): The path to the .json file to be processed.
    """
    # Load the JSON file
    with open(json_file, 'r') as f:
        transcription_data = json.load(f)

    # Process each line of the transcription and add the emotion
    for entry in transcription_data:
        text = entry["text"]

        # Use the emotion classifier to get predictions for the text
        result = emotion_classifier(text)
        
        # Find the emotion with the highest score (this assumes the model returns a sorted list of emotions)
        if result:
            # Get the emotion with the highest score
            emotion = result[0]['label']
        else:
            emotion = "neutral"  # Default to neutral if no emotion detected

        # Add the emotion to the current entry
        entry["emotion"] = emotion
    
    # Save the updated transcription data with emotion stamps back to the JSON file
    with open(json_file, 'w') as f:
        json.dump(transcription_data, f, indent=4)
    
    print(f"Emotion stamps successfully added to: {json_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python emotion_stamper.py <path_to_json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    # Add emotion stamps to the provided JSON file
    add_emotion_to_transcription(json_file)