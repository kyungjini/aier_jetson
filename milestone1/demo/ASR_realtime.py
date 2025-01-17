import os
import pyaudio
from google.cloud import speech

# Google Cloud 서비스 계정 키 파일 경로 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service_key_team5.json"

# Google Speech-to-Text 클라이언트 초기화
client = speech.SpeechClient()

# 오디오 스트림 설정
RATE = 44100  # 샘플링 레이트 (16kHz)
CHUNK = int(RATE / 10)  # 100ms 단위로 데이터 읽기


def main():
    # PyAudio로 오디오 입력 스트림 생성
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
            print(f"오디오 스트림에서 예외 발생: {e}")
            raise e

    # Google Cloud Speech API 요청 설정
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="ko-KR",  # 한국어 설정
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    # 실시간 스트림 전송 및 응답 처리
    print("실시간 음성 인식 시작 (종료하려면 Ctrl+C)...")
    try:
        responses = client.streaming_recognize(
            streaming_config, generate_audio_stream()
        )
        for response in responses:
            print(response)
            if response.results and response.results[0].alternatives:
                # 첫 번째 결과와 대안을 출력
                transcript = response.results[0].alternatives[0].transcript
                print(f"인식된 텍스트: {transcript}")

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("음성 인식을 종료합니다.")
