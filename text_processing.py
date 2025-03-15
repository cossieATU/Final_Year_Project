"""
text_processing.py

This file handles text preprocessing, such as chunking text into manageable sizes.
"""

def split_text_into_chunks(text, chunk_length=100):
    """
    Splits text into smaller chunks for better processing.

    Args:
        text (str): The input text to split.
        chunk_length (int): The maximum number of words per chunk.

    Returns:
        list: A list of text chunks.
    """
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
