import argparse
from src.utils import capture_frames, extract_faces

def main():
    parser = argparse.ArgumentParser(
        description="Data Creation Script: Capture webcam frames and extract faces."
    )
    parser.add_argument(
        '--capture',
        action='store_true',
        help="Capture frames from the webcam and save them to the frames folder."
    )
    parser.add_argument(
        '--extract',
        action='store_true',
        help="Extract faces from captured frames and save them to the faces folder."
    )
    args = parser.parse_args()

    if args.capture:
        print("Starting frame capture...")
        capture_frames()
    if args.extract:
        print("Starting face extraction...")
        extract_faces()
    if not (args.capture or args.extract):
        print("Please specify --capture and/or --extract.")
    
if __name__ == '__main__':
    main()
