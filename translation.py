"""
translation.py

This module handles text translation using CTranslate2 and SentencePiece.
"""

import sentencepiece as spm
import ctranslate2
from config import CT_MODEL_PATH, SP_MODEL_PATH, DEVICE

# Load SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL_PATH)

# Load CTranslate2 models
translator = ctranslate2.Translator(CT_MODEL_PATH, device=DEVICE)
translator_lock = threading.lock()

def translate_text(text, translator, src_lang, tgt_lang):
    """
    Translates text from the source language to the target language.

    Args:
        text (str): The input text to translate.
        translator (ctranslate2.Translator): The CTranslate2 translator instance.
        src_lang (str): The source language code.
        tgt_lang (str): The target language code.

    Returns:
        str: The translated text.
    """
    if not text.strip():
        return ""

    # Tokenize text using SentencePiece
    tokenized_text = sp.encode_as_pieces(text)
    source_input = [[src_lang] + tokenized_text + ["</s>"]]
    
    # Set the target prefix
    target_prefix = [[tgt_lang]] * len(source_input)

    # Perform translation
    translations = translator.translate_batch(
        source_input, target_prefix=target_prefix, beam_size=4
    )

    # Decode the translation output
    translated_text = sp.decode(translations[0].hypotheses[0]).strip()

    return translated_text
