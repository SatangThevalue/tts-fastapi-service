import os
import subprocess
import argparse
import json

def process_video(
    input_video: str,
    output_video: str,
    mute_original_audio: bool = False,
    bgm_file: str = None,
    bgm_volume: float = 0.3,
    crop_916: bool = False,
    drawtext_text: str = None,
    font_file: str = None,
    font_size: int = 48,
    font_color: str = "white"
):
    """
    Dynamically constructs and executes an FFmpeg command using complex filters.
    Highly optimized for single-pass processing without MoviePy.
    """
    
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "info"]
    
    # Inputs
    cmd.extend(["-i", input_video])
    input_count = 1
    
    if bgm_file and os.path.exists(bgm_file):
        cmd.extend(["-i", bgm_file])
        bgm_input_index = input_count
        input_count += 1
    else:
        bgm_input_index = -1

    filter_complex = []
    
    # --- Video Filters ---
    video_filters = []
    last_v_pad = "[0:v]"
    
    if crop_916:
        # Crop to 9:16 (e.g., for Shorts/Reels) by taking the center
        # iw = input width, ih = input height. Target width = ih * 9/16
        video_filters.append(f"{last_v_pad}crop='ih*9/16':'ih'[vcrop]")
        last_v_pad = "[vcrop]"
        
    if drawtext_text:
        # Thai font support needs a proper font file. If not provided, fallback to default.
        font_opt = f":fontfile={font_file}" if font_file and os.path.exists(font_file) else ""
        text_safe = drawtext_text.replace("'", r"\'").replace(":", r"\:")
        video_filters.append(
            f"{last_v_pad}drawtext=text='{text_safe}':fontcolor={font_color}:fontsize={font_size}{font_opt}:x=(w-text_w)/2:y=(h-text_h)/2[vtext]"
        )
        last_v_pad = "[vtext]"
        
    if video_filters:
        filter_complex.append(";".join(video_filters))
        final_v_pad = last_v_pad
    else:
        final_v_pad = "0:v" # No video filters, just use original

    # --- Audio Filters ---
    audio_filters = []
    last_a_pad = "[0:a]"
    final_a_pad = None
    
    if mute_original_audio and bgm_input_index == -1:
        # Just mute, no other audio
        pass # We will use -an or not map audio
    elif mute_original_audio and bgm_input_index != -1:
        # Only BGM
        audio_filters.append(f"[{bgm_input_index}:a]volume={bgm_volume}[a_bgm]")
        final_a_pad = "[a_bgm]"
    elif not mute_original_audio and bgm_input_index != -1:
        # Mix original and BGM
        audio_filters.append(f"[{bgm_input_index}:a]volume={bgm_volume}[a_bgm]")
        audio_filters.append(f"[0:a][a_bgm]amix=inputs=2:duration=first:dropout_transition=2[a_mix]")
        final_a_pad = "[a_mix]"
    else:
        # Original audio only
        final_a_pad = "0:a"
        
    if audio_filters:
        filter_complex.append(";".join(audio_filters))

    # --- Assemble Command ---
    if filter_complex:
        cmd.extend(["-filter_complex", ";".join(filter_complex)])
        
    # Map Video
    if final_v_pad != "0:v":
        cmd.extend(["-map", final_v_pad])
    else:
        cmd.extend(["-map", "0:v"])
        
    # Map Audio
    if mute_original_audio and bgm_input_index == -1:
        # No audio at all
        pass 
    else:
        if final_a_pad and final_a_pad != "0:a":
            cmd.extend(["-map", final_a_pad])
        else:
            # Fallback to map 0:a if it exists
            # We use a trick to only map audio if it exists using ? but let's assume it exists for now
            cmd.extend(["-map", "0:a?"])

    # Codecs
    cmd.extend([
        "-c:v", "libx264", 
        "-preset", "fast", 
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k"
    ])
    
    cmd.append(output_video)
    
    print(f"Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return {"status": "success", "output": output_video, "cmd": " ".join(cmd)}
    except subprocess.CalledProcessError as e:
        print("FFmpeg Error Output:", e.stderr)
        return {"status": "error", "error": e.stderr, "cmd": " ".join(cmd)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic FFmpeg Processor for n8n")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--mute", action="store_true", help="Mute original audio")
    parser.add_argument("--bgm", help="Path to background music file")
    parser.add_argument("--bgm-volume", type=float, default=0.3, help="BGM Volume (0.0 to 1.0)")
    parser.add_argument("--crop-916", action="store_true", help="Crop video to 9:16 vertical format")
    parser.add_argument("--text", help="Text to overlay in the center")
    
    args = parser.parse_args()
    
    res = process_video(
        input_video=args.input,
        output_video=args.output,
        mute_original_audio=args.mute,
        bgm_file=args.bgm,
        bgm_volume=args.bgm_volume,
        crop_916=args.crop_916,
        drawtext_text=args.text
    )
    
    print(json.dumps(res, indent=2))
