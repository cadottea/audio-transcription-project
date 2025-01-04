import ffmpeg
import os

def split_audio(input_file, output_dir, segment_duration):
    """
    Splits the input audio file into smaller chunks.

    :param input_file: Path to the input audio file to split
    :param output_dir: Directory to save the split audio files
    :param segment_duration: Duration (in seconds) for each split chunk
    """
    # Get the duration of the audio file
    probe = ffmpeg.probe(input_file, v="error", select_streams="a", show_entries="stream=duration")
    duration = float(probe['streams'][0]['duration'])

    # Calculate the number of parts based on the segment duration (e.g., 5 seconds)
    num_parts = int(duration // segment_duration)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split the audio file into parts
    for i in range(num_parts):
        start_time = i * segment_duration
        output_file = os.path.join(output_dir, f'output_{i+1:03d}.wav')
        
        # Use ffmpeg to split the audio
        ffmpeg.input(input_file, ss=start_time, t=segment_duration).output(output_file).run()

    print(f"Audio has been split into {num_parts} parts in '{output_dir}'.")

# The script will be called and used by ser_whisper.py
if __name__ == "__main__":
    pass  # The file will be used in ser_whisper.py so no need for direct execution