import os
import pyaudio
from google.cloud import speech

import argostranslate.package
import argostranslate.translate

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service_key_team5.json"

client = speech.SpeechClient()

RATE = 44100
CHUNK = int(RATE / 10)


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
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        input_device_index=0,
        frames_per_buffer=CHUNK,
    )

    def generate_audio_stream():
        try:
            while True:
                audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
                # audio_chunk = stream.read(CHUNK)
                yield speech.StreamingRecognizeRequest(audio_content=audio_chunk)
        except Exception as e:
            print(f"Error: {e}")
            raise e

    install_korean_to_english_package()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="ko-KR",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    print("Real Time ASR ... (Ctrl+C to exit)")
    try:
        responses = client.streaming_recognize(
            streaming_config, generate_audio_stream()
        )
        for response in responses:
            print(response)
            if response.results and response.results[0].alternatives:
                transcript = response.results[0].alternatives[0].transcript
                print(f"Korean Text (ASR) : {transcript}")
                print(f"Translated Text   : {translate_text(transcript)}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exit ASR")
