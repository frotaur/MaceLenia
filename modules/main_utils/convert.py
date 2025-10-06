import subprocess
import json
import os
import sys

def speedup_video(input_file, output_file=None, speed_up_factor=1.0):
    # If output file not specified, create one with _speedup suffix and .mp4 extension
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_speedup_{speed_up_factor}x.mp4"
    else:
        # Ensure output has .mp4 extension
        base_name = os.path.splitext(output_file)[0]
        output_file = f"{base_name}.mp4"
    
    # Get the input video's fps using ffprobe
    probe_cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        input_file
    ]
    
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    video_info = json.loads(probe_result.stdout)
    
    # Find the video stream and get the fps
    video_stream = next((stream for stream in video_info['streams'] 
                         if stream['codec_type'] == 'video'), None)
    
    if video_stream and 'r_frame_rate' in video_stream:
        fps_str = video_stream['r_frame_rate']
        num, den = map(int, fps_str.split('/'))
        original_fps = num / den
    else:
        # Default if we can't determine the fps
        original_fps = 30
    
    # Calculate new fps (rounded to integer)
    new_fps = round(original_fps * speed_up_factor)
    
    # Handle audio speed-up (atempo only works between 0.5 and 2.0)
    if speed_up_factor <= 2.0:
        audio_filter = f'atempo={speed_up_factor}'
    else:
        # Chain multiple atempo filters for larger speed-up factors
        audio_filters = []
        remaining = speed_up_factor
        
        while remaining > 1.0:
            if remaining >= 2.0:
                audio_filters.append('atempo=2.0')
                remaining /= 2.0
            else:
                audio_filters.append(f'atempo={remaining}')
                remaining = 1.0
        
        audio_filter = ','.join(audio_filters)
    
    # Build FFmpeg command
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-c:v', 'libx264',       # H.264 codec
        '-preset', 'medium',     # Balance between speed and quality
        '-crf', '23',            # Quality factor
        '-vf', f'setpts={1/speed_up_factor}*PTS,fps={new_fps}',  # Speed adjustment and fps
        '-c:a', 'aac',           # Audio codec
        '-b:a', '128k',          # Audio bitrate
        '-af', audio_filter,     # Audio speed adjustment
        '-f', 'mp4',             # Force mp4 format
        output_file
    ]
    
    # Execute the command
    subprocess.run(cmd)
    
    return output_file

if __name__ == '__main__':
    # Common video file extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    
    # Get the speed up factor from command line if provided, otherwise use default 1.0
    speed_up_factor = 1.0
    if len(sys.argv) > 1:
        try:
            speed_up_factor = float(sys.argv[1])
        except ValueError:
            print(f"Invalid speed up factor: {sys.argv[1]}. Using default: 1.0")
    
    print(f"Processing videos with speed up factor: {speed_up_factor}...")
    
    # Create the 'converted' directory if it doesn't exist
    converted_dir = 'converted'
    if not os.path.exists(converted_dir):
        os.makedirs(converted_dir)
    
    # Get current directory
    current_dir = os.getcwd()
    
    # Process all video files in the current directory
    processed_count = 0
    
    for filename in os.listdir(current_dir):
        file_path = os.path.join(current_dir, filename)
        
        # Skip directories and non-video files
        if os.path.isdir(file_path):
            continue
        
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in video_extensions:
            continue
        
        # Create output path with the same filename in the 'converted' folder
        output_filename = os.path.splitext(filename)[0] + '.mp4'
        output_path = os.path.join(converted_dir, output_filename)
        
        print(f"Converting: {filename} -> {output_path}")
        
        try:
            speedup_video(file_path, output_path, speed_up_factor)
            processed_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"Conversion complete. Processed {processed_count} videos.")