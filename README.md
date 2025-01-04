# Whisper Python Wrapper

This project is a Python wrapper around the Whisper transcription model, utilizing the high-performance capabilities of whisper.cpp. The wrapper is designed to offer efficient transcription processing with support for batch processing, emotion tagging, and future enhancements like speaker identification. Specifically, this wrapper is tailored for leveraging GPU acceleration on Apple M1 chips, providing a more efficient transcription experience by utilizing the available hardware resources.

## Features
- **Batch Processing**: Process multiple audio files (batch2.py) in one go, reducing time spent on individual transcriptions.
- **HTML Coloring**: Takes ANSI terminal-output color from Whisper.cpp and converts it into permanent HTML colors - for assessing whisper transcription confidence
- **Keyword labeling**: Available but not activated in `batch2.py` with `keyword_analyzer.py`.
- **Emotion Tagging**: Adds emotion labels (using ser13_whisper.py) to the transcriptions (work-in-progress).
- **GPU Support on Apple M1**: Leverages whisper.cpp for GPU acceleration, optimizing transcription on Apple M1 hardware.
- **Future Features**: Plans to add speaker identification to distinguish between speakers in multi-speaker environments.

## Prerequisites
- Python 3.6 or later installed on your machine.
- whisper.cpp installed and configured to run on your system for GPU acceleration.
- A system with GPU support, especially Apple M1 hardware, for optimal performance.

## Installation
1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/whisper-python-wrapper.git
    ```
2. Navigate into the project directory:
    ```bash
    cd whisper-python-wrapper
    ```
3. Install the required Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Set up whisper.cpp: Follow the installation instructions in the whisper.cpp repository to enable GPU acceleration.

## Usage
1. After installation, you can process audio files in batch using the following script.
2. Run the batch processing:
    ```bash
    python batch_processor.py
    ```
    This will process all `.wav` files in the specified directory, transcribe them with Whisper, and save the output to text files. Emotion tags will be added (with `ser13_whisper.py`), and the transcription will be saved in JSON format as well.
3. **Directory Structure**:
   - `transcribe/`: Folder containing .wav audio files for transcription.
   - `output/`: Folder where transcription results will be saved.

## Roadmap
- **Emotion Tagging**: Work-in-progress feature that adds emotion labels to transcriptions.
- **Speaker Identification**: Future feature to tag different speakers in multi-speaker audio files.

## Conclusion

This Python wrapper for Whisper offers an efficient way to handle transcription tasks, using the power of the Whisper model and whisper.cpp for GPU-accelerated transcription. With batch processing, emotion tagging, and planned speaker identification, it aims to provide an easy-to-use tool for high-quality transcription services.