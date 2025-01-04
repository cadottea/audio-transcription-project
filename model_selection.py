import os
import sys

def select_model():
    model_dir = "../models"  # Assuming models is in the parent directory
    models = [f for f in os.listdir(model_dir) if f.startswith("ggml") and f.endswith(".bin")]

    if not models:
        print("No models found in the 'models/' directory.")
        sys.exit(1)

    print("Select a Whisper model:")
    for i, model in enumerate(models, 1):
        print(f"{i}) {model}")

    try:
        choice = int(input("Enter your choice: "))
        if 1 <= choice <= len(models):
            return os.path.join(model_dir, models[choice - 1])
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)
    except ValueError:
        print("Invalid input. Please enter a number.")
        sys.exit(1)