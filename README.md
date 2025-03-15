# :memo:Final_Year_Project
For my final year project I have set out to create a one-to-many speech-to-speech (S2S) translator. This translator will allow for one source language and the output of two unique target language. ( Current state of project: Incomplete)

# :black_nib:Introduction
This project is a real-time speech translation system that:

- Uses **Whisper** for speech-to-text (STT).
- Uses **CTranslate2 + SentencePiece** for machine translation (MT).
- Uses **Amazon Polly (via Boto3)** for text-to-speech (TTS).

The system is optimised for low latency, making it ideal for live translation scenarios.

## :dna:Features

- **Real-time speech transcription** using OpenAI's Whisper.
- **Fast Translation** using NLLB models in CTranslate2.
- **Text-to-Speech (TTS)** using Amazon Polly.
- Optimised for **Nvidia Jetson AGX Orin**.

### Features to come
- Simultaneous multi-language translation
- Multi-device audo output support

# :gear:Installation
1. Clone the repository
2. Set up Virtual Enviroment
3. Install Dependencies (requirements.txt)
4. Create .env file and add Amazon credentials (required to use Amazon Polly)

# :pray:Acknowledgments
This project would not be possible without the contributions of the following:
-🔗 [whisper_real_time](https://github.com/davabase/whisper_real_time)  
-🔗 [CTranslate2](https://github.com/OpenNMT/CTranslate2)
- **SentencePiece**
- **Amazon Polly**
