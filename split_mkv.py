import subprocess
import json
import sys
import os
import shutil

def check_ffmpeg_installed():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("Error: ffmpeg and ffprobe must be installed and accessible in the system PATH.")
        sys.exit(1)

def get_video_duration(input_file):
    """Get the duration of the video in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        input_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    duration = float(data["format"]["duration"])
    print(f'Duration: {duration} seconds')
    return duration

def split_video(input_file):
    """Split the input MKV file into three equal parts without re-encoding."""
    # Check if input file exists and output directory is writable
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)
    if not os.access(os.path.dirname(input_file) or ".", os.W_OK):
        print(f"Error: Output directory '{os.path.dirname(input_file) or '.'}' is not writable.")
        sys.exit(1)

    # Get the duration of the video
    try:
        duration = get_video_duration(input_file)
    except Exception as e:
        print(f"Error getting video duration: {e}")
        sys.exit(1)

    # Calculate the duration of each part
    part_duration = duration / 3

    # Base name for output files
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.dirname(input_file) or "."

    # Temporary files for split parts
    temp_files = []
    for i in range(3):  # Split into 3 parts
        start_time = i * part_duration
        temp_file = os.path.join(output_dir, f"{base_name}_temp_part{i+1}.mkv")
        temp_files.append(temp_file)

        # Construct the ffmpeg command for splitting
        cmd = [
            "ffmpeg", "-i", input_file,
            "-ss", str(start_time),
            "-t", str(part_duration),
            "-c", "copy",
            "-y",
            temp_file
        ]

        # Run the ffmpeg command
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Split part {i+1}: {temp_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error splitting part {i+1}: {e}")
            print(f"ffmpeg output: {e.stderr}")
            sys.exit(1)

    return temp_files

def convert_to_format(temp_files, output_format="mp4"):
    """Convert the temporary MKV files to the specified format."""
    output_files = []
    for temp_file in temp_files:
        base_name = os.path.splitext(temp_file)[0].replace("_temp", "")
        output_file = f"{base_name}.{output_format}"
        output_files.append(output_file)

        # Construct the ffmpeg command for conversion
        if output_format == "mp4":
            cmd = [
                "ffmpeg", "-i", temp_file,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-y",
                output_file
            ]
        elif output_format == "webm":
            cmd = [
                "ffmpeg", "-i", temp_file,
                "-c:v", "libvpx",
                "-c:a", "libvorbis",
                "-y",
                output_file
            ]

        # Run the ffmpeg command
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Converted to {output_file}")
            os.remove(temp_file)
            print(f"Removed temporary file {temp_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {temp_file}: {e}")
            print(f"ffmpeg output: {e.stderr}")
            sys.exit(1)

    return output_files

if __name__ == "__main__":
    check_ffmpeg_installed()

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python split_mkv.py <input_mkv_file> [output_format]")
        print("  output_format: 'mp4' (default) or 'webm'")
        sys.exit(1)

    input_file = sys.argv[1]
    output_format = sys.argv[2].lower() if len(sys.argv) == 3 else "mp4"

    if output_format not in ["mp4", "webm"]:
        print("Error: Output format must be 'mp4' or 'webm'.")
        sys.exit(1)

    print(f"Splitting '{input_file}' into three parts...")
    temp_files = split_video(input_file)
    print(f"Converting parts to '{output_format}'...")
    convert_to_format(temp_files, output_format)