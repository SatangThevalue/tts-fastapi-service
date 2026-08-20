import os
import json
import argparse
from pydub import AudioSegment
from pydub.silence import split_on_silence

def prepare_dataset(input_audio_path, transcript_file, output_dir="dataset", min_silence_len=500, silence_thresh=-40):
    """
    อัตโนมัติหั่นไฟล์เสียงบรรยายยาวๆ ออกเป็นท่อนสั้นๆ ตามจังหวะหยุดหายใจ
    พร้อมสร้างโครงสร้าง Dataset สำหรับนำไป Fine-tune TTS
    """
    os.makedirs(output_dir, exist_ok=True)
    wavs_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    
    print(f"Loading audio: {input_audio_path}...")
    audio = AudioSegment.from_file(input_audio_path)
    
    print("Splitting audio on silence (this may take a while)...")
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=200 # เก็บความเงียบไว้ 200ms หัวท้ายให้ฟังสบาย
    )
    
    # In a real scenario, you'd align chunks with the transcript text using ASR (Whisper).
    # Here, we generate a dummy metadata.jsonl ready for manual or ASR filling.
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    
    print(f"Generated {len(chunks)} audio chunks. Saving to {wavs_dir}...")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            chunk_filename = f"chunk_{i:04d}.wav"
            chunk_path = os.path.join(wavs_dir, chunk_filename)
            chunk.export(chunk_path, format="wav")
            
            # Dummy JSONL entry
            entry = {
                "audio_filepath": f"wavs/{chunk_filename}",
                "text": "[PLACEHOLDER_FOR_ASR_TRANSCRIPT]",
                "speaker": "Speaker_Custom"
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\\n")
            
    print(f"\\n✅ Dataset prepared successfully at '{output_dir}/'!")
    print("Next Step: Use an ASR tool (like Whisper) to fill in the 'text' fields in metadata.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS Dataset Preparation Tool (Auto Audio Slicer)")
    parser.add_argument("--audio", type=str, required=True, help="Path to long input audio file")
    parser.add_argument("--out", type=str, default="dataset", help="Output directory")
    args = parser.parse_args()
    
    prepare_dataset(args.audio, None, args.out)
