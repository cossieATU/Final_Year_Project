import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import ctranslate2
import sentencepiece as spm
import time  # TIMER MODULE ADDED

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

# Set paths for your CTranslate2 model and SentencePiece model
CT_MODEL_PATH = "/home/atu/venv/Cossie/nllb-200-3.3B-ct2-fp16"  # model path
SP_MODEL_PATH = "/home/atu/venv/Cossie/flores200_sacrebleu_tokenizer_spm.model"  # SentencePiece model path

# Device setting
DEVICE = "cuda"  

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
    translated_text = translated_text.replace("spa_Latn", "").strip()
    
    return translated_text

def split_text_into_chunks(text, chunk_length=100):
    """ Split the transcription text into smaller chunks to ensure better processing. """
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

def process_audio_chunks(data_queue, recorder, audio_model, translator, sp, phrase_timeout):
    transcription = []  # Live display transcript
    english_transcript = []  # Stores full English transcription
    spanish_transcript = []  # Stores full Spanish transcription
    phrase_time = None

    # START TOTAL TIMER
    start_time = time.time()  # TIMER ADDED HERE

    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now
                
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # START TIMER FOR SENTENCE PROCESSING
                sentence_start_time = time.time()  # TIMER ADDED HERE
                
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()
                
                # Split the transcription into smaller chunks
                text_chunks = split_text_into_chunks(text)

                # Translate each chunk
                for chunk in text_chunks:
                    translated_text = translate_text(chunk, src_lang="eng_Latn", tgt_lang="spa_Latn")
                    
                    transcription.append(f"Original (English): {chunk}")
                    transcription.append(f"Translated (Spanish): {translated_text}")
                    
                    # Store full transcriptions for final display
                    english_transcript.append(chunk)
                    spanish_transcript.append(translated_text)

                # END TIMER FOR SENTENCE PROCESSING
                sentence_end_time = time.time()  # TIMER ADDED HERE
                sentence_time_taken = sentence_end_time - sentence_start_time
                print(f"\n⏱️ Sentence processed in {sentence_time_taken:.2f} seconds")  # PRINT SENTENCE TIME

                # Display live transcription updates
                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                print('', end='', flush=True)

            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

    # END TOTAL TIMER
    total_time_taken = time.time() - start_time  # TIMER ADDED HERE

    # **Display Full Transcriptions at the End**
    print("\n\n===== FULL TRANSCRIPTION COMPARISON =====\n")
    
    # Print full English transcription
    print("\n--- Full English Transcript ---\n")
    print(" ".join(english_transcript))  # Joins everything into a single paragraph
    
    # Print full Spanish transcription
    print("\n--- Full Spanish Transcript ---\n")
    print(" ".join(spanish_transcript))  # Joins everything into a single paragraph

    # PRINT TOTAL TIME TAKEN
    print(f"\n⏳ Total processing time: {total_time_taken:.2f} seconds")  # TIMER ADDED HERE

def main():
    # Simulate the command-line arguments
    args = argparse.Namespace(
        model="base",  # Simulate the --model argument
        non_english=False,  # Simulate --non_english argument
        energy_threshold=1000,  # Simulate --energy_threshold argument
        record_timeout=2,  # Simulate --record_timeout argument
        phrase_timeout=3,  # Simulate --phrase_timeout argument
        default_microphone="Blue Snowball",  # Simulate --default_microphone argument
    )

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    
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
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that grabs audio data.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.\n")

    # Start processing the audio chunks
    process_audio_chunks(data_queue, recorder, audio_model, translator, sp, phrase_timeout)

if __name__ == "__main__":
    main()

