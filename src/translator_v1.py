import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import ctranslate2
import sentencepiece as spm

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

# Set paths for your CTranslate2 model and SentencePiece model
CT_MODEL_PATH = "/home/atu/venv/Cossie/nllb-200-3.3B-ct2-int8"  # model path
SP_MODEL_PATH = "/home/atu/venv/Cossie/flores200_sacrebleu_tokenizer_spm.model"  # SentencePiece model path

# Device setting
DEVICE = "cpu" 

# Load SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL_PATH)

# Load CTranslate2 model
translator = ctranslate2.Translator(CT_MODEL_PATH, device=DEVICE)

def translate_text(text, src_lang="eng_Latn", tgt_lang="spa_Latn"):
    """Translate the text using NLLB model"""
    source_sentences = [text]
    
    # Tokenize and prepare the source sentence
    source_sents_subworded = [sp.encode_as_pieces(sent) for sent in source_sentences]
    source_sents_subworded = [[src_lang] + sent + ["</s>"] for sent in source_sents_subworded]
    
    # Set the target prefix for each sentence
    target_prefix = [[tgt_lang]] * len(source_sents_subworded)
    
    # Translate the sentence
    translations = translator.translate_batch(
        source_sents_subworded,
        target_prefix=target_prefix,
        beam_size=4
    )
    
    # Decode the translated sentence
    translated_text = sp.decode(translations[0].hypotheses[0])
    
    return translated_text

def main():
    # Simulate the command-line arguments
    args = argparse.Namespace(
        model="tiny",  # Simulate the --model argument
        non_english=False,  # Simulate --non_english argument
        energy_threshold=1000,  # Simulate --energy_threshold argument
        record_timeout=2,  # Simulate --record_timeout argument
        phrase_timeout=3,  # Simulate --phrase_timeout argument
        default_microphone="Blue Snowball",  # Simulate --default_microphone argument
    )

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = []  # Start with an empty list

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue user
    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now
                
                # Combine audio data from queue
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Read the transcription.
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                # Translate the transcribed text
                translated_text = translate_text(text, src_lang="eng_Latn", tgt_lang="spa_Latn")  # Example: English to Spanish

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(f"Original (English): {text}")
                    transcription.append(f"Translated (Spanish): {translated_text}")
                else:
                    if len(transcription) < 2:  # Ensure we have at least two entries
                        transcription.append(f"Original (English): {text}")
                        transcription.append(f"Translated (Spanish): {translated_text}")
                    else:
                        transcription[-2] = f"Original (English): {text}"
                        transcription[-1] = f"Translated (Spanish): {translated_text}"

                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    print(line)
                print('', end='', flush=True)
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)




if __name__ == "__main__":
    main()
