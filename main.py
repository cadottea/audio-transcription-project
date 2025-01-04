from whisper import run_whisper
from model_selection import select_model
from arg_parser import parse_arguments

def main():
    args = parse_arguments()

    model_path = args.model if args.model else select_model()
    run_whisper(args.file, args.output_dir, model_path)

if __name__ == "__main__":
    main()