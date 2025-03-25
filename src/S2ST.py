import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import ctranslate2
import sentencepiece as spm
import time
import boto3
import threading
import tempfile
from pydub import AudioSegment
import sounddevice as sd
import soundfile as sf
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
import pygame  # Ensure pygame is imported for audio playback

# Set paths for your CTranslate2 model and SentencePiece model
CT_MODEL_PATH = "/home/atu/venv/Cossie/nllb-200-3.3B-ct2-fp16"  # Change to your model path
SP_MODEL_PATH = "/home/atu/venv/Cossie/flores200_sacrebleu_tokenizer_spm.model"  # Change to your SentencePiece model path

# Device setting
DEVICE = "cuda"  # Or use "cpu"

# Load SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL_PATH)

# Load CTranslate2 model
translator = ctranslate2.Translator(CT_MODEL_PATH, device=DEVICE)

def translate_text_dual(text, src_lang="eng_Latn", tgt_langs=["spa_Latn", "fra_Latn"]):
    """
    Translates a single text into multiple target languages simultaneously.
    
    Args:
        text (str): The input text in the source language.
        src_lang (str): The source language token (e.g., "eng_Latn").
        tgt_langs (list): A list of target language tokens (e.g., ["spa_Latn", "fra_Latn"]).
        
    Returns:
        list: Translated texts corresponding to each target language.
    """
    # Tokenize the source sentence once
    tokens = sp.encode_as_pieces(text)
    tokens = [src_lang] + tokens + ["</s>"]
    
    # Duplicate the tokenized sentence for each target language
    source_batch = [tokens for _ in tgt_langs]
    
    # Create a target prefix for each target language
    target_prefixes = [[lang] for lang in tgt_langs]
    
    # Perform batch translation
    translations = translator.translate_batch(
        source_batch,
        target_prefix=target_prefixes,
        beam_size=4
    )
    
    # Decode each translation and remove the target token if present
    translated_texts = []
    for i, translation in enumerate(translations):
        translated = sp.decode(translation.hypotheses[0])
        if translated.startswith(tgt_langs[i]):
            translated = translated[len(tgt_langs[i]):].strip()
        translated_texts.append(translated)
    
    return translated_texts

def split_text_into_chunks(text, chunk_length=100):
    """Split the transcription text into smaller chunks to ensure better processing."""
    sentences = text.split(". ")
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def speak_text_with_polly(text, voice_id="Lucia", output_format="mp3"):
    """Use Amazon Polly to synthesize `text` into speech and play via Pygame."""
    if not text.strip():
        return 

    polly_client = boto3.client("polly")

    try:
        response = polly_client.synthesize_speech(
            Text=text,
            OutputFormat=output_format,
            VoiceId=voice_id
        )

        # Save the speech to a temp file
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_audio:
            temp_audio.write(response["AudioStream"].read())
            temp_audio.flush()

            # Initialize pygame mixer with explicit audio parameters
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096)
            pygame.mixer.music.load(temp_audio.name)
            pygame.mixer.music.play()

            # Wait for audio to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.delay(100)

            pygame.mixer.quit()

    except Exception as e:
        print(f"Error using Amazon Polly: {e}")

def process_audio_chunks(data_queue, recorder, audio_model, phrase_timeout):
    """
    Continuously process incoming audio from `data_queue`, transcribe, and translate
    into both Spanish and French simultaneously.
    """
    transcription = []         # Live display transcript
    english_transcript = []    # Full English transcription
    spanish_transcript = []    # Full Spanish transcription
    french_transcript = []     # Full French transcription
    phrase_time = None

    # Start total timer
    start_time = time.time()

    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                # Check if a phrase is complete (based on phrase_timeout)
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now
                
                # Gather all queued audio data
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                # # Start timer for this chunk
                sentence_start_time = time.time()
                
                # Transcribe with Whisper
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()
                
                # Split the transcription into smaller chunks for translation
                text_chunks = split_text_into_chunks(text)

                # Process each chunk
                for chunk in text_chunks:
                    # Translate the chunk into both Spanish and French simultaneously
                    translations = translate_text_dual(chunk, src_lang="eng_Latn", tgt_langs=["spa_Latn", "fra_Latn"])
                    
                    # Append with language indicators
                    transcription.append(f"[EN] {chunk}")
                    transcription.append(f"[SPA] {translations[0]}")
                    transcription.append(f"[FRA] {translations[1]}")
                    
                    # Save full transcriptions
                    english_transcript.append(chunk)
                    spanish_transcript.append(translations[0])
                    french_transcript.append(translations[1])
                    
                    # Synthesize speech using Polly (example: only using Spanish here)
                    speak_text_with_polly(translations[0], voice_id="Lucia")
                    # To also speak French, uncomment the following line (ensure you have an appropriate French voice)
                    speak_text_with_polly(translations[1], voice_id="Lea")
                
                # End timer for this chunk
                sentence_end_time = time.time()
                sentence_time_taken = sentence_end_time - sentence_start_time
                print(f"\n⏱️ Sentence processed in {sentence_time_taken:.2f} seconds")

                # Clear screen and display live transcript with indicators
                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                print('', end='', flush=True)
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

    # End total timer
    total_time_taken = time.time() - start_time

    # Display full transcriptions at the end
    print("\n\n===== FULL TRANSCRIPTION COMPARISON =====\n")
    
    print("\n--- Full English Transcript ---\n")
    print(" ".join(english_transcript))
    
    print("\n--- Full Spanish Transcript ---\n")
    print(" ".join(spanish_transcript))
    
    print("\n--- Full French Transcript ---\n")
    print(" ".join(french_transcript))

    print(f"\n⏳ Total processing time: {total_time_taken:.2f} seconds")

def main():
    # Simulate command-line arguments
    args = argparse.Namespace(
        model="base",                # Simulate the --model argument
        non_english=False,           # Simulate --non_english argument
        energy_threshold=800,       # Simulate --energy_threshold argument
        record_timeout=3,            # Simulate --record_timeout argument
        phrase_timeout=2,            # Simulate --phrase_timeout argument
        default_microphone="Blue Snowball",  # Simulate --default_microphone argument
    )

    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    # Microphone setup (especially important for Linux users)
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices:")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Load / download Whisper model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    # Adjust for ambient noise
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread to capture audio data
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.\n")
    # Start processing the audio chunks (this will run until interrupted)
    process_audio_chunks(data_queue, recorder, audio_model, phrase_timeout)

if __name__ == "__main__":
    main()
