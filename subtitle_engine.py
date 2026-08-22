import os
from faster_whisper import WhisperModel
import datetime

# --- Whisper Model Configuration ---
# 'base' model requires only ~500MB RAM, fast enough on CPU
MODEL_SIZE = "base" 
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

def format_timestamp(seconds: float) -> str:
    """ Convert seconds (e.g. 1.234) to ASS format (H:MM:SS.cs) """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int(round((seconds - int(seconds)) * 100))
    if centisecs == 100:
        secs += 1
        centisecs = 0
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

def generate_ass_subtitle(audio_path: str, output_ass_path: str, style: str = "tiktok_yellow"):
    """
    Transcribes audio and generates a dynamic .ass subtitle file.
    Provides Karaoke-style word highlighting.
    """
    print(f"🎙️ [Faster-Whisper] Loading model '{MODEL_SIZE}' on {DEVICE}...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    
    print(f"🎧 [Faster-Whisper] Transcribing {audio_path}...")
    # enable word_timestamps to get the start/end time of EACH word
    segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    
    print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

    # Generate ASS File Content
    # We define two styles here: Default (the unhighlighted word) and Highlight
    ass_header = f"""[Script Info]
ScriptType: v4.00+
Collisions: Normal
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Sarabun,80,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,6,2,2,10,10,250,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ass_events = []
    
    for segment in segments:
        for word_info in segment.words:
            start_time = format_timestamp(word_info.start)
            end_time = format_timestamp(word_info.end)
            text = word_info.word.strip()
            
            # Simple TikTok style: word pops up, colored yellow
            # We use ASS override tags: {\c&H00FFFF&} for yellow in BGR hex format
            if style == "tiktok_yellow":
                styled_text = f"{{\\c&H00FFFF&}}{text}"
            else:
                styled_text = text
                
            # Alignment 2 is Bottom-Center.
            ass_line = f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{styled_text}\n"
            ass_events.append(ass_line)

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header)
        f.writelines(ass_events)

    print(f"✅ [Subtitle Engine] Generated dynamic subtitle: {output_ass_path}")
    return output_ass_path

if __name__ == "__main__":
    # Test script locally
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--style", default="tiktok_yellow")
    args = parser.parse_args()
    
    generate_ass_subtitle(args.audio, args.output, args.style)
