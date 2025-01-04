import ffmpeg
import os
import sys

def split_audio(input_file, output_dir, segment_duration=5):
    # Check if the input file exists
    if not os.path.isfile(input_file):
        print(f"Error: Audio file '{input_file}' does not exist.")
        sys.exit(1)

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

def main():
    if len(sys.argv) < 2:
        print("Usage: python split2.py <path_to_audio_file>")
        sys.exit(1)

    input_file = sys.argv[1]  # Take the input file from the command line argument
    transcribe_dir = os.path.expanduser('~/Desktop/transcribe')  # Directory for transcribing files
    output_dir = os.path.join(transcribe_dir, "temp_split")  # Output directory for temp split files

    # Call the function to split the audio
    split_audio(input_file, output_dir)

if __name__ == "__main__":
    main()