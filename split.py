import ffmpeg
import os

def split_audio(input_file, output_dir, num_parts=5):
    # Get the duration of the audio file
    probe = ffmpeg.probe(input_file, v="error", select_streams="a", show_entries="stream=duration")
    duration = float(probe['streams'][0]['duration'])

    # Calculate the segment duration
    segment_duration = duration / num_parts

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split the audio file into parts
    for i in range(num_parts):
        start_time = i * segment_duration
        output_file = os.path.join(output_dir, f'output_{i+1:03d}.wav')
        
        # Use ffmpeg to split the audio
        ffmpeg.input(input_file, ss=start_time, t=segment_duration).output(output_file).run()

    print(f"Audio has been split into {num_parts} parts in '{output_dir}'.")

# Example usage
input_file = os.path.expanduser('~/Desktop/R20241205-080549.WAV')  # Path to your original .wav file on Desktop
output_dir = os.path.expanduser('~/Desktop/transcribe')  # Directory where split audio files will be saved on Desktop
split_audio(input_file, output_dir)