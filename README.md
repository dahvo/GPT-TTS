# Text-to-Speech Processing

This repository contains a set of Python scripts for processing text-to-speech (TTS) tasks, including splitting sentences, generating silence, and creating subtitles. 

This is meant for long form text for example converting an E-book to a realistic sounding TTS audio book with matching subtitles.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/dahvo/GPT-TTS.git
    cd GPT-TTS
    git clone https://github.com/yourusername/tts-processing.git
    cd tts-processing
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Synthesize Speech
1. Create a new file in the home directory named main.py
```python
from do_tts import text_to_tts()

with open("text_example.txt" 
```
Or whatever the file is called that holds the text you are converting to TTS. The text to voice conversion is set up so that any size of text can be converted, even a book works without a hitch
```python
, "r", encoding="utf-8") as file:
        input_text = file.read()

    speaker = "audiobook_lady"
    ```
Choose a speaker, either one from my examples or you may create your own speaker embedding using the provided files
```
```python
    output_name = "test_output"
    #Name of the output for the audio and srt file

    text_to_tts(input_text, speaker, output_name)
    ```