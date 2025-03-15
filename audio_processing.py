"""
audio_processing.py

This file handles audio playback and text-to-speech synthesis using Amazon Polly.
"""

import os
import boto3
import pygame
import tempfile
import threading

def speak_text_with_polly(text: str, voice_id: str = "Lucia", output_format: str = "mp3", sink_name: str = None):
    """
    Synthesizes speech using Amazon Polly and plays it via Pygame.

    Args:
        text (str): The text to synthesize.
        voice_id (str): The Amazon Polly voice ID. Default is "Lucia".
        output_format (str): The output format (e.g., "mp3").
        sink_name (str, optional): The audio output sink for specific devices.
    """
    if not text.strip():
        return 

    polly_client = boto3.client("polly")

    try:
        response = polly_client.synthesize_speech(
            Text=text, OutputFormat=output_format, VoiceId=voice_id
        )

        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_audio:
            temp_audio.write(response["AudioStream"].read())
            temp_audio.flush()

            # Assign audio output device if specified
            if sink_name:
                os.environ["PULSE_SINK"] = sink_name

            # Play audio with Pygame
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096)
            pygame.mixer.music.load(temp_audio.name)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.delay(100)

            pygame.mixer.quit()

    except Exception as e:
        print(f"Error using Amazon Polly: {e}")

def speak_dual_translation(text_spanish: str, text_catalan: str, sink_spanish: str = "7", sink_catalan: str = "3"):
    """
    Outputs two translations simultaneously on separate output devices.

    Args:
        text_spanish (str): The Spanish translation.
        text_catalan (str): The Catalan translation.
        sink_spanish (str): The PulseAudio sink for Spanish audio.
        sink_catalan (str): The PulseAudio sink for Catalan audio.
    """
    thread1 = threading.Thread(target=speak_text_with_polly, args=(text_spanish, "Lucia", "mp3", sink_spanish))
    thread2 = threading.Thread(target=speak_text_with_polly, args=(text_catalan, "Lucia", "mp3", sink_catalan))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
