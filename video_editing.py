import ffmpeg
import whisper
import json
import subprocess
import re
import os
import classify2
import librosa
import soundfile as sf
import srt
from scipy import signal

def vid_to_aud(input_video_path, output_audio_path="temp_audio.mp3"):
    """Extract audio from video using ffmpeg."""
    subprocess.run([
        "ffmpeg", "-i", input_video_path, "-q:a", "0", "-map", "a", output_audio_path, "-y"
    ])
    return output_audio_path

def process_video(input_vid_path):
    """Transcribe video and return speech data with timestamps."""
    audio_path = vid_to_aud(input_vid_path)
    whisper_model = whisper.load_model("small")
    transcription_result = whisper_model.transcribe(audio_path, word_timestamps=True)
    
    speech_data = [
        {"start": segment["start"], "end": segment["end"], "type": "speech", "text": segment["text"]}
        for segment in transcription_result["segments"]
    ]
    return speech_data

def detect_silence(audio_path):
    """Detect silence in audio using ffmpeg."""
    command = [
        "ffmpeg", "-i", audio_path, "-af",
        "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-"
    ]
    result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
    return result.stderr

def parse_silence_data(ffmpeg_output):
    """Parse silence data from ffmpeg output."""
    silence_data = []
    silence_start_times = re.findall(r"silence_start: (\d+\.\d+)", ffmpeg_output)
    silence_end_times = re.findall(r"silence_end: (\d+\.\d+)", ffmpeg_output)
    
    for start, end in zip(silence_start_times, silence_end_times):
        silence_data.append({"start": float(start), "end": float(end), "type": "silence"})
    return silence_data

def merge_and_save_json(speech_data, silence_data, output_json="output.json"):
    """Merge speech and silence data, save to JSON."""
    all_data = sorted(speech_data + silence_data, key=lambda x: x["start"])
    with open(output_json, "w") as f:
        json.dump(all_data, f, indent=4)
    print(f"✅ Merged data saved to {output_json}")

def classify_speech():
    """Classify speech segments as relevant/irrelevant."""
    classify2.classify()
    print("✅ Speech data classified successfully in the JSON file")

def trim_segments(input_vid_path, output_vid_path="output_trimmed.mp4"):
    """Trim silent and irrelevant parts from the video."""
    with open("output.json", "r") as f:
        data = json.load(f)
    
    # Keep only relevant segments
    keep_segments = [
        (entry["start"], entry["end"])
        for entry in data
        if entry.get("classification") == "Relevant"
    ]
    
    if not keep_segments:
        print("⚠️ No relevant segments found. Exiting...")
        return
    
    # Generate FFmpeg filter for trimming
    filter_complex = "".join([
        f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{idx}]; "
        f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{idx}]; "
        for idx, (start, end) in enumerate(keep_segments)
    ])
    
    concat_v = "".join([f"[v{idx}][a{idx}]" for idx in range(len(keep_segments))]) + f"concat=n={len(keep_segments)}:v=1:a=1 [v][a]"
    
    command = [
        "ffmpeg", "-i", input_vid_path, "-filter_complex",
        filter_complex + concat_v,
        "-map", "[v]", "-map", "[a]",
        output_vid_path, "-y"
    ]
    
    subprocess.run(command)
    print(f"✅ Trimmed video saved as {output_vid_path}")
    return output_vid_path

def apply_noise_reduction(audio_path, output_audio="denoised_audio.wav"):
    """Apply noise reduction to audio."""
    y, sr = librosa.load(audio_path, sr=None)
    filtered_audio = bandpass_filter(y, 300, 3400, sr)
    sf.write(output_audio, filtered_audio, sr)
    return output_audio

def bandpass_filter(audio_data, lowcut, highcut, sr, order=6):
    """Apply a bandpass filter to audio data."""
    nyquist = 0.5 * sr
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, audio_data)

def generate_captions_from_trimmed_video(trimmed_video_path, srt_file="captions.srt"):
    """Generate captions from the trimmed video's audio."""
    # Extract audio from the trimmed video
    trimmed_audio_path = vid_to_aud(trimmed_video_path, "trimmed_audio_for_captions.mp3")
    
    # Transcribe the trimmed audio
    whisper_model = whisper.load_model("small")
    transcription_result = whisper_model.transcribe(trimmed_audio_path, word_timestamps=True)
    
    # Generate subtitles
    subtitles = [
        srt.Subtitle(
            index=i + 1,
            start=srt.timedelta(seconds=seg['start']),
            end=srt.timedelta(seconds=seg['end']),
            content=seg['text']
        )
        for i, seg in enumerate(transcription_result["segments"])
    ]
    
    # Save subtitles to SRT file
    with open(srt_file, "w") as f:
        f.write(srt.compose(subtitles))
    
    print(f"✅ Captions generated from trimmed video and saved as: {srt_file}")
    return srt_file

def overlay_captions(video_path, srt_path, output_video="final_video_with_captions.mp4"):
    """Overlay captions on the video."""
    subprocess.run([
        "ffmpeg", "-i", video_path, "-vf", f"subtitles={srt_path}", output_video
    ], check=True)
    print(f"✅ Captions overlaid successfully, saved as: {output_video}")
    return output_video

def replace_audio(video_path, new_audio, output_video="final_output.mp4"):
    """Replace audio in the video with the denoised audio."""
    command = [
        "ffmpeg", "-i", video_path, "-i", new_audio,
        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", output_video
    ]
    subprocess.run(command, check=True)
    print(f"✅ Audio replaced successfully, saved as: {output_video}")
    return output_video

if __name__ == "__main__":
    input_vid_path = r"D:\Smartcut_using_SVM\test_video_files\asympt_tuto.mp4"
    
    print("🔄 Processing video...")
    speech_data = process_video(input_vid_path)
    
    print("🔎 Detecting silence...")
    ffmpeg_output = detect_silence("temp_audio.mp3")
    silence_data = parse_silence_data(ffmpeg_output)
    
    print("📂 Saving merged data...")
    merge_and_save_json(speech_data, silence_data)
    
    print("🧠 Running classification...")
    classify_speech()
    
    print("✂️ Trimming silence and irrelevant parts...")
    trimmed_video_path = trim_segments(input_vid_path)
    
    print("🎤 Extracting audio from trimmed video...")
    trimmed_audio_path = vid_to_aud(trimmed_video_path, "trimmed_audio.mp3")
    
    print("🔇 Applying noise reduction to audio...")
    denoised_audio_path = apply_noise_reduction(trimmed_audio_path)
    
    print("📝 Generating captions from trimmed video...")
    srt_file = generate_captions_from_trimmed_video(trimmed_video_path)
    
    print("🔊 Replacing audio with denoised version...")
    final_video_with_audio = replace_audio(trimmed_video_path, denoised_audio_path)
    
    print("🎥 Overlaying captions on video...")
    final_output = overlay_captions(final_video_with_audio, srt_file)
    
    print(f"✅ Final edited video saved as: {final_output}")