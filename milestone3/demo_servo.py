import time
import board
import busio
from adafruit_pca9685 import PCA9685

# class Servo:
#     def __init__(self, address=0x40, min_pulse=0.75, max_pulse=2.25, frequency=50):
#         # I2C 초기화
#         i2c = busio.I2C(board.SCL, board.SDA)
#         self.pca = PCA9685(i2c, address=address)
#         self.pca.frequency = frequency
        
#         # 서보 설정
#         self.min_duty = int((min_pulse / 20.0) * 0xFFFF)  # 20ms 주기에서 최소 펄스
#         self.max_duty = int((max_pulse / 20.0) * 0xFFFF)  # 20ms 주기에서 최대 펄스

#     def angle(self, channel, angle):
#         """
#         서보 모터의 각도를 설정합니다.
#         :param channel: PCA9685 채널 (0~15)
#         :param angle: 설정할 각도 (-90 ~ 90)
#         """
#         # 각도를 듀티 사이클 값으로 변환
#         duty_range = self.max_duty - self.min_duty
#         duty = self.min_duty + int(((angle + 90) / 180.0) * duty_range)
#         self.pca.channels[channel].duty_cycle = duty


# if __name__ == "__main__":
#     # Servo 객체 초기화 (I2C 주소 0x40)
#     servo = Servo(address=0x40)

#     try:
#         while True:
#             # 서보를 0도로 설정
#             servo.angle(0, 0.0)
#             time.sleep(2)  # 2초 대기

#             # 서보를 -90도로 설정
#             servo.angle(0, -90.0)
#             time.sleep(2)  # 2초 대기

#             # 서보를 90도로 설정
#             servo.angle(0, 90.0)
#             time.sleep(2)  # 2초 대기

#             print("loop")
#     except KeyboardInterrupt:
#         print("Program stopped.")

import time
import busio
import board
from adafruit_pca9685 import PCA9685


class JetsonServo:
    def __init__(self, address=0x40, channel=0, min_angle=-90.0, max_angle=90.0):
        # I2C 초기화
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c, address=address)
        self.pca.frequency = 50
        self.channel = self.pca.channels[channel]
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.angle(0)

    def angle(self, angle):
        """
        서보 모터의 각도를 설정합니다.
        :param angle: 설정할 각도 (-90 ~ 90)
        """
        # 각도를 서보의 허용 범위로 제한
        angle = max(self.min_angle, min(self.max_angle, angle))

        # 각도를 듀티 사이클로 변환
        pulse_min = 0.5  # 최소 펄스 (ms)
        pulse_max = 2.5  # 최대 펄스 (ms)
        duty_cycle = int(((angle + 90) / 180.0) * (pulse_max - pulse_min) * 0xFFFF / 20.0 + pulse_min * 0xFFFF / 20.0)

        # 듀티 사이클 설정
        self.channel.duty_cycle = duty_cycle


def auto_targeting(detections, servo_pan, servo_tilt, calib, frame_shape):
    """
    타겟 탐지 및 추적 로직 구현:
    - 탐지된 타겟이 중앙에 가까운지 확인
    - 서보 모터를 제어하여 타겟 중앙으로 조준
    """
    frame_height, frame_width = frame_shape[:2]
    stable_threshold = 3  # 타겟 안정성 확인을 위한 카운트 임계값
    pan_error_threshold = 10  # 허용 가능한 Pan 오차 (픽셀)
    tilt_error_threshold = 10  # 허용 가능한 Tilt 오차 (픽셀)
    stable_count = 0
    last_pan, last_tilt = 0.0, 0.0

    for det in detections:
        x1, y1, x2, y2, conf = det  # cls_id 제거 (필요 시 복원)
        if conf < 0.5:  # 확신도가 낮은 타겟은 무시
            continue

        # 타겟 중심 계산
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2

        # 화면 중심과 타겟 중심의 차이 계산
        pan_error = (target_center_x - frame_width / 2) + calib.x_correct
        tilt_error = (target_center_y - frame_height / 2) + calib.y_correct

        # Pan과 Tilt 각도 조정
        if abs(pan_error) > pan_error_threshold:
            last_pan = max(-90, min(90, last_pan - pan_error / 100))  # Pan 조정 비율
            servo_pan.angle(last_pan)

        if abs(tilt_error) > tilt_error_threshold:
            last_tilt = max(-90, min(90, last_tilt - tilt_error / 100))  # Tilt 조정 비율
            servo_tilt.angle(last_tilt)

        print(last_pan, last_tilt)
        time.sleep(10)

        # 타겟 안정성 확인
        if abs(pan_error) < pan_error_threshold and abs(tilt_error) < tilt_error_threshold:
            stable_count += 1
        else:
            stable_count = 0

        # 안정적인 타겟 상태로 전환
        if stable_count >= stable_threshold:
            print(f"[Target] Stable target detected at confidence {conf:.2f}")
            return True  # 안정적이면 추적 완료 신호 반환

    return False  # 추적 미완료


if __name__ == "__main__":
    # 서보 객체 초기화
    servo_pan = JetsonServo(channel=0)
    servo_tilt = JetsonServo(channel=1)

    # 가상 캘리브레이션 값 (사용자 설정 가능)
    class Calibration:
        x_correct = 0
        y_correct = 0

    calib = Calibration()

    # 가상 타겟 데이터 (x1, y1, x2, y2, confidence)
    detections = [
        # (0, 100, 100, 200, 0.9),  # 화면에서 특정 타겟 위치
        # (0, 100, 100, 200, 0.9),
        # (0, 100, 100, 200, 0.9),
        # (0, 100, 100, 200, 0.9),
        # (0, 100, 100, 200, 0.9),
        # (0, 100, 100, 200, 0.9),
        # (0, 100, 100, 200, 0.9),
        # (0, 100, 100, 200, 0.9),
        # (0, 100, 100, 200, 0.9),
        # (0, 100, 100, 200, 0.9),
        # (0, 100, 100, 200, 0.9),
    ]

    # 화면 해상도 (가로, 세로)
    frame_shape = (480, 640)

    # 타겟 자동 추적 테스트
    target_locked = auto_targeting(detections, servo_pan, servo_tilt, calib, frame_shape)
    if target_locked:
        print("[Result] Target successfully tracked!")
    else:
        print("[Result] Target tracking failed.")