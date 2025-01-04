import os
import json

def test_file_writing(audio_file, output_dir):
    """
    Test the process of writing .txt and .json files to the output directory.
    Logs detailed information to investigate potential issues.

    Args:
        audio_file (str): Name of the audio file being processed.
        output_dir (str): Directory where files will be saved.
    """
    print("=== Starting File Writing Test ===")
    print(f"Input audio file: {audio_file}")
    print(f"Output directory: {output_dir}")

    # Ensure output directory exists
    print(f"Attempting to create directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directory exists: {os.path.exists(output_dir)}")

    # Simulate transcription data
    transcription_data = [
        {"text": "Hello world", "start": 0.0, "end": 2.0},
        {"text": "This is a test", "start": 2.0, "end": 4.0},
    ]
    print(f"Simulated transcription data: {json.dumps(transcription_data, indent=4)}")

    # Define file paths
    transcription_txt_file = os.path.join(output_dir, os.path.basename(audio_file).replace(".WAV", ".txt"))
    transcription_json_file = os.path.join(output_dir, os.path.basename(audio_file).replace(".WAV", ".json"))

    try:
        # Write .txt file
        with open(transcription_txt_file, "w") as txt_file:
            for line in transcription_data:
                txt_file.write(f"[{line['start']:.3f} --> {line['end']:.3f}] {line['text']}\n")
        print(f".txt file successfully created at: {transcription_txt_file}")

        # Write .json file
        with open(transcription_json_file, "w") as json_file:
            json.dump(transcription_data, json_file, indent=4)
        print(f".json file successfully created at: {transcription_json_file}")

        # Verify files exist
        print("\n=== Verifying File Existence ===")
        if os.path.exists(transcription_txt_file):
            print(f"Verified: .txt file exists at {transcription_txt_file}")
        else:
            print(f"ERROR: .txt file is missing at {transcription_txt_file}")

        if os.path.exists(transcription_json_file):
            print(f"Verified: .json file exists at {transcription_json_file}")
        else:
            print(f"ERROR: .json file is missing at {transcription_json_file}")

        # Log file contents for debugging
        print("\n=== File Content Debugging ===")
        with open(transcription_txt_file, "r") as txt_file:
            txt_content = txt_file.read()
            print(f"Content of .txt file:\n{txt_content}")

        with open(transcription_json_file, "r") as json_file:
            json_content = json.load(json_file)
            print(f"Content of .json file:\n{json.dumps(json_content, indent=4)}")

    except Exception as e:
        print(f"ERROR during file writing or validation: {e}")

# Test the function
if __name__ == "__main__":
    desktop = os.path.expanduser("~/Desktop")
    output_dir = os.path.join(desktop, "test_output")  # Test directory for outputs

    test_audio_file = "V20230515-155530.WAV"
    test_file_writing(test_audio_file, output_dir)