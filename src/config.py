

"""
config.py 

This file stores global config variables for paths, device settings, and model config.

"""

import os 

# File paths to SentencePiece and CTranslate2 models
CT_MODEL_PATH = os.path.abspath("models/nllb-200-3.3B-ct2-fp16")  # Change to actual path
SP_MODEL_PATH = os.path.abspath("models/flores200_sacrebleu_tokenizer_spm.model")

# Device Settings
DEVICE = "cuda" # ensures program uses GPU for faster processing (Change to CPU if no GPU available)