"""
main.py

This is the main entry point for the speech translation system.
Handles command-line arguments, initializes Whisper model,
and manages real-time audio processing.
"""

import argparse
import numpy as np
import speech_recognition as sr
import whisper
import torch
from queue import Queue
from time import sleep
from datetime import datetime, timedelta
from translation import translate_text, translator_spanish, translator_catalan
from audio_processing import speak_dual_translation
from text_processing import split_text_into_chunks


def process_audio_chunks(data_queue, audio_model, phrase_timeout):
    """
    Processes incoming audio, transcribes it using Whisper,
    translates to Spanish & Catalan, and plays it using TTS.

    Args:
        data_queue (Queue): The queue containing recorded audio data.
        audio_model (whisper.Whisper): The loaded Whisper model for STT.
        phrase_timeout (int): Timeout period for considering a phrase complete.
    """
    transcription = []  # Stores live transcription
    phrase_time = None  # Keeps track of when last phrase ended
    start_time = time.time()

    while True:
        try:
            now = datetime.utcnow()

            if not data_queue.empty():
                # Check if a phrase is complete
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now

                # Retrieve and process audio data
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()

                # Convert to NumPy array for Whisper processing
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Transcribe with Whisper
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                if not text:
                    continue  # Skip if no transcription was detected

                # Split long transcription into chunks for better translation
                text_chunks = split_text_into_chunks(text)

                for chunk in text_chunks:
                    translated_text_spanish = translate_text(chunk, translator_spanish, "eng_Latn", "spa_Latn")
                    translated_text_catalan = translate_text(chunk, translator_catalan, "eng_Latn", "cat_Latn")

                    transcription.append(f"Original: {chunk}")
                    transcription.append(f"Spanish: {translated_text_spanish}")
                    transcription.append(f"Catalan: {translated_text_catalan}")

                    # Output translated speech via Polly (to different audio sinks)
                    speak_dual_translation(translated_text_spanish, translated_text_catalan, sink_spanish="7", sink_catalan="3")

                # Display live transcription
                print("\n".join(transcription), end="\n\n", flush=True)

            else:
                sleep(0.25)  # Sleep briefly to prevent high CPU usage

        except KeyboardInterrupt:
            break  # Exit gracefully on keyboard interrupt

    total_time_taken = time.time() - start_time
    print(f"\n⏳ Total processing time: {total_time_taken:.2f} seconds")


def main():
    """
    Main function to initialize components and start audio processing.
    Handles command-line arguments for user-configurable options.
    """
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Real-time speech translation system")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--energy_threshold", type=int, default=1000, help="Microphone sensitivity for detecting speech")
    parser.add_argument("--record_timeout", type=int, default=2, help="Max duration (in seconds) of each recording segment")
    parser.add_argument("--phrase_timeout", type=int, default=3, help="Timeout duration to consider a phrase as completed")
    parser.add_argument("--microphone", type=str, default="Blue Snowball", help="Specify microphone name")

    args = parser.parse_args()

    # Queue for storing recorded audio data
    data_queue = Queue()

    # Initialize SpeechRecognition recognizer
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False  # Disable automatic threshold adjustment

    # Select microphone device (important for Linux users)
    source = None
    if 'linux' in platform:
        mic_name = args.microphone
        if not mic_name or mic_name.lower() == 'list':
            print("Available microphones:")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"{index}: {name}")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Load Whisper STT model
    model_size = args.model
    if model_size != "large":
        model_size += ".en"  # Use English-optimized models for smaller versions
    audio_model = whisper.load_model(model_size)

    # Adjust microphone sensitivity
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData):
        """
        Callback function for handling recorded audio segments.

        Args:
            _ (Any): Unused argument.
            audio (sr.AudioData): The recorded audio segment.
        """
        data_queue.put(audio.get_raw_data())

    # Start background recording
    recorder.listen_in_background(source, record_callback, phrase_time_limit=args.record_timeout)

    print("\n🚀 Model loaded. System is now running...\n")
    
    # Start processing recorded audio
    process_audio_chunks(data_queue, audio_model, args.phrase_timeout)


if __name__ == "__main__":
    main()
