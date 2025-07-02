#!/usr/bin/env python3
import argparse
import wave
from pathlib import Path
from wyoming.audio import AudioStart, AudioStop, wav_to_chunks
from wyoming.asr import Transcribe, Transcript
from wyoming.client import AsyncClient

SAMPLES_PER_CHUNK = 1024

def transcribe_wav(uri: str, wav_path: Path) -> str:
    # Connect to the Wyoming ASR service
    connection = AsyncClient.from_uri(uri)
    
    try:
        # Start transcription session
        connection.write_event(
            Transcribe(name="nemo-parakeet-tdt-0.6b-v2").event()
        )
        
        # Open and stream the WAV file
        with wave.open(str(wav_path), "rb") as wav_file:
            # Send audio start event
            connection.write_event(
                AudioStart(
                    rate=wav_file.getframerate(),
                    width=wav_file.getsampwidth(),
                    channels=wav_file.getnchannels(),
                ).event()
            )
            
            # Stream audio chunks
            for chunk in wav_to_chunks(wav_file, SAMPLES_PER_CHUNK):
                connection.write_event(chunk.event())
            
            # Send audio stop event
            connection.write_event(AudioStop().event())
        
        # Wait for transcription result
        while True:
            event = connection.read_event()
            if event is None:
                break
                
            if Transcript.is_type(event.type):
                transcript = Transcript.from_event(event)
                return transcript.text
    
    finally:
        connection.close()
    
    raise RuntimeError("No transcription received")

def main():
    parser = argparse.ArgumentParser(description="Transcribe WAV file using Wyoming ASR service")
    parser.add_argument("wav_file", type=Path, help="Path to WAV file to transcribe")
    parser.add_argument("--host", default="tcp://localhost:10300", help="Wyoming ASR service host")

    args = parser.parse_args()
    
    if not args.wav_file.exists():
        print(f"Error: WAV file {args.wav_file} does not exist")
        return 1
        
    try:
        text = transcribe_wav(args.host, args.wav_file)
        print(f"Transcription: {text}")
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())