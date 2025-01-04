import os
import time
import datetime

# Define keywords for analysis
KEYWORDS = ["hot", "confused", "don't understand", "unclear", "misunderstood", "what do you mean"]

# Paths
OUTPUT_DIR = os.path.expanduser("~/Desktop/output")  # Output directory used by ser whisper
FLAGGED_FILE = os.path.join(OUTPUT_DIR, "flagged_keywords.txt")  # Log file for flagged keywords

# Function to analyze text for keywords
def analyze_keywords(text):
    """Check for predefined keywords in the text."""
    return [keyword for keyword in KEYWORDS if keyword in text.lower()]

# Main program loop to monitor the output directory
def monitor_transcriptions():
    """Continuously monitor the output directory for new transcription files."""
    print(f"Monitoring transcriptions in {OUTPUT_DIR}...")

    # Ensure the flagged keywords log file exists
    if not os.path.exists(FLAGGED_FILE):
        with open(FLAGGED_FILE, 'w') as file:
            file.write("Flagged Keywords Log\n")
            file.write("=====================\n")

    processed_files = set()  # Keep track of processed files

    while True:
        # Get all .txt files in the output directory
        transcription_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".txt")]

        for file_name in transcription_files:
            file_path = os.path.join(OUTPUT_DIR, file_name)

            # Skip files that have already been processed
            if file_path in processed_files:
                continue

            # Read the transcription file
            with open(file_path, 'r') as f:
                text = f.read()

            # Analyze the text for keywords
            flagged_keywords = analyze_keywords(text)
            if flagged_keywords:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(FLAGGED_FILE, 'a') as log_file:
                    log_file.write(f"[{timestamp}] File: {file_name} | Keywords: {', '.join(flagged_keywords)}\n")
                    log_file.write(f"Text: {text}\n\n")
                print(f"Flagged keywords in {file_name}: {flagged_keywords}")

            # Mark the file as processed
            processed_files.add(file_path)

        # Wait for a short interval before checking again
        time.sleep(5)

if __name__ == "__main__":
    monitor_transcriptions()