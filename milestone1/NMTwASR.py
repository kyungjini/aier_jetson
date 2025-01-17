import os
import wave
import argparse
from google.cloud import speech

import argostranslate.package
import argostranslate.translate

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service_key_team5.json"

client = speech.SpeechClient()


def generate_audio_stream(wav_file_path):
    with wave.open(wav_file_path, "rb") as wf:
        chunk_size = 4096
        while True:
            data = wf.readframes(chunk_size)
            if not data:
                break
            yield speech.StreamingRecognizeRequest(audio_content=data)


def install_korean_to_english_package():
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    ko_en_packages = [
        pkg
        for pkg in available_packages
        if pkg.from_code == "ko" and pkg.to_code == "en"
    ]

    if not ko_en_packages:
        print("No ko-en packages found")
        return

    package_to_install = ko_en_packages[0]
    download_path = package_to_install.download()
    argostranslate.package.install_from_path(download_path)
    print("ko-en package installed")


def translate_text(text):
    translated_text = argostranslate.translate.translate(text, "ko", "en")
    print(f"Original Text  : {text}")
    print(f"Translated Text: {translated_text}")
    return translated_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_file_path", type=str, default="./korean_convert.wav")
    args = parser.parse_args()

    wav_file_path = args.wav_file_path

    with wave.open(wav_file_path, "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError("Mono audio required")
        if wf.getsampwidth() != 2:
            raise ValueError("Sampling size 16bit required")
        if wf.getframerate() != 44100:
            raise ValueError("Sampling rate 44100Hz required")

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="ko-KR",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    install_korean_to_english_package()

    print("Loading WAV")
    try:
        responses = client.streaming_recognize(
            streaming_config, generate_audio_stream(wav_file_path)
        )
        for response in responses:
            if response.results and response.results[0].alternatives:
                transcript = response.results[0].alternatives[0].transcript
                print(f"korean text (ASR) : {transcript}")
                print(f"translated text   : {translate_text(transcript)}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
