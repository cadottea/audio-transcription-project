import ffmpeg
import os

def split_audio(input_file, transcribe_dir, segment_duration=5):
    # Get the duration of the audio file
    probe = ffmpeg.probe(input_file, v="error", select_streams="a", show_entries="stream=duration")
    duration = float(probe['streams'][0]['duration'])

    # Calculate the number of parts based on the segment duration (e.g., 5 seconds)
    num_parts = int(duration // segment_duration)

    # Create the temp_split folder in the transcribe directory
    temp_split_dir = os.path.join(transcribe_dir, "temp_split")
    os.makedirs(temp_split_dir, exist_ok=True)

    # Split the audio file into parts
    for i in range(num_parts):
        start_time = i * segment_duration
        output_file = os.path.join(temp_split_dir, f'output_{i+1:03d}.wav')

        # Use ffmpeg to split the audio
        ffmpeg.input(input_file, ss=start_time, t=segment_duration).output(output_file).run()

    print(f"Audio has been split into {num_parts} parts in '{temp_split_dir}'.")

# Example usage
def main():
    # Get the first .wav file from transcribe folder
    transcribe_dir = os.path.expanduser('~/Desktop/transcribe')  # Path to your transcribe folder
    files = os.listdir(transcribe_dir)
    audio_files = [os.path.join(transcribe_dir, f) for f in files if f.lower().endswith(".wav")]

    if not audio_files:
        print("No .wav files found in the 'transcribe' directory.")
        return

    input_file = audio_files[0]  # Only process the first .wav file
    split_audio(input_file, transcribe_dir)

if __name__ == "__main__":
    main()