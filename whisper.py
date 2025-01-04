import os
import subprocess
import sys
from color_utils import convert_ansi_to_html

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

        print("\nColored transcription in terminal:")
        print(colored_output)

        with open(txt_output_file, 'w') as file:
            file.write(colored_output)
        print(f"Transcription saved to: {txt_output_file}")

        html_content = convert_ansi_to_html(colored_output)
        html_content_wrapped = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Whisper Transcription with Colors</title></head>
        <body style="background-color: black; color: white;">
            <pre>{html_content}</pre>
        </body>
        </html>
        """
        html_output_file = txt_output_file.replace('.txt', '_colored.html')
        with open(html_output_file, 'w', encoding='utf-8') as file:
            file.write(html_content_wrapped)
        print(f"HTML file saved to: {html_output_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error: Whisper.cpp failed with return code {e.returncode}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)