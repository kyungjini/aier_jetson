import socket
import struct
import threading
import time
import cv2
import numpy as np
import os
import Jetson.GPIO as GPIO
from adafruit_pca9685 import PCA9685
import board
import busio
import onnxruntime

import tensorrt as trt
from cuda import cudart 

# ---------------------------------------
# (1) Servo 및 GPIO 제어 클래스
# ---------------------------------------
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
        pulse_min = 0.75  # 최소 펄스 (ms)
        pulse_max = 2.25  # 최대 펄스 (ms)
        duty_cycle = int(((angle + 90) / 180.0) * (pulse_max - pulse_min) * 0xFFFF / 20.0 + pulse_min * 0xFFFF / 20.0)

        # 듀티 사이클 설정
        self.channel.duty_cycle = duty_cycle

class JetsonGpio:
    PIN_FIRE = 29
    PIN_LASER = 31

    def __init__(self):
        GPIO.cleanup()
        GPIO.setmode(GPIO.BOARD)
        # GPIO.setup(self.PIN_FIRE, GPIO.OUT)
        # GPIO.setup(self.PIN_LASER, GPIO.OUT)

    def fire(self, value):
        GPIO.output(self.PIN_FIRE, GPIO.HIGH if value else GPIO.LOW)

    def laser(self, value):
        GPIO.output(self.PIN_LASER, GPIO.HIGH if value else GPIO.LOW)

# ---------------------------------------
# (2) 시스템 상태 및 타겟 관리
# ---------------------------------------
class SystemState:
    SAFE = 0
    PREARMED = 1
    ARMED_MANUAL = 2
    ENGAGE_AUTO = 3

class TAutoEngage:
    def __init__(self):
        self.state = SystemState.SAFE
        self.targets = []
        self.current_target = None
        self.stable_count = 0
        self.last_pan = None
        self.last_tilt = None

    def reset(self):
        self.state = SystemState.SAFE
        self.targets.clear()
        self.current_target = None
        self.stable_count = 0

# ---------------------------------------
# (3) 캘리브레이션 및 설정 파일 관리
# ---------------------------------------
class Calibration:
    def __init__(self, filename="Correct.ini"):
        self.filename = filename
        self.x_correct = 0
        self.y_correct = 0
        self.read_offsets()

    def read_offsets(self):
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                lines = f.readlines()
            try:
                self.x_correct = float(lines[0].split()[1])
                self.y_correct = float(lines[1].split()[1])
            except:
                print("[Error] Calibration file parsing error.")
        print(f"[Calib] x_correct={self.x_correct}, y_correct={self.y_correct}")

    def write_offsets(self):
        with open(self.filename, "w") as f:
            f.write(f"xCorrect {self.x_correct}\n")
            f.write(f"yCorrect {self.y_correct}\n")

# ---------------------------------------
# (4) YOLO 추론 클래스
# ---------------------------------------
class YOLOv5TRTNoPyCUDA:
    def __init__(self, engine_path, input_hw=(640, 640)):
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)
        self.input_hw = input_hw

        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if not self.engine:
            raise RuntimeError("Failed to load TensorRT engine.")

        self.context = self.engine.create_execution_context()
        self.input_idx = self.engine.get_binding_index("input")
        self.output_idx = self.engine.get_binding_index("output")

        self.input_shape = list(self.engine.get_binding_shape(self.input_idx))
        self.output_shape = list(self.engine.get_binding_shape(self.output_idx))

        self.input_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_idx))
        self.output_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_idx))

        self.host_input = np.zeros(self.input_shape, dtype=self.input_dtype)
        self.host_output = np.zeros(self.output_shape, dtype=self.output_dtype)

        self.d_input = None
        self.d_output = None

        self.context.set_binding_shape(self.input_idx, self.input_shape)

    def infer(self, frame):
        H, W = self.input_hw
        img = cv2.resize(frame, (W, H))
        img = img.astype(self.input_dtype) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        self.host_input[...] = img

        if self.d_input is None:
            n_input_bytes = self.host_input.nbytes
            n_output_bytes = self.host_output.nbytes
            self.d_input = cudart.cudaMalloc(n_input_bytes)[1]
            self.d_output = cudart.cudaMalloc(n_output_bytes)[1]

        # Host->Device
        cudart.cudaMemcpy(
            self.d_input,
            self.host_input.ctypes.data,
            self.host_input.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )

        bindings = [0] * self.engine.num_bindings
        bindings[self.input_idx] = self.d_input
        bindings[self.output_idx] = self.d_output

        self.context.execute_v2(bindings)

        # Device->Host
        cudart.cudaMemcpy(
            self.host_output.ctypes.data,
            self.d_output,
            self.host_output.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )

        return self.host_output

    def postprocess(self, output, original_hw):

        original_h, original_w = original_hw
        model_h, model_w = self.input_hw
        scale_h = original_h / model_h
        scale_w = original_w / model_w

        result = []
        for det in output[0]:
            conf = det[4]
            if conf > 0.5:
                x1, y1, x2, y2 = det[0:4]
                x1 = int(x1 * scale_w)
                x2 = int(x2 * scale_w)
                y1 = int(y1 * scale_h)
                y2 = int(y2 * scale_h)

                result.append((x1, y1, x2, y2, conf))
        return result

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
        if conf < 0.7:  # 확신도가 낮은 타겟은 무시
            continue

        print(det)

        # 타겟 중심 계산
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2

        # 화면 중심과 타겟 중심의 차이 계산
        pan_error = (target_center_x - frame_width / 2) + calib.x_correct + last_pan
        tilt_error = (target_center_y - frame_height / 2) + calib.y_correct + last_tilt

        print(f"{pan_error}, {tilt_error}")

        # Pan과 Tilt 각도 조정
        if abs(pan_error) > pan_error_threshold:
            last_pan = max(-85, min(85, last_pan - pan_error / 75))  # Pan 조정 비율
            servo_pan.angle(last_pan)

        if abs(tilt_error) > tilt_error_threshold:
            last_tilt = max(-35, min(35, last_tilt - tilt_error / 75))  # Tilt 조정 비율
            servo_tilt.angle(last_tilt)

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


def recvall(sock, count):
    buf = b""
    while count > 0:
        data = sock.recv(count)
        if not data:
            return None
        buf += data
        count -= len(data)
    return buf


# 수정된 Main Loop
def main():
    host = "192.168.0.160"
    port = 10000
    model_path = "/home/group5/AI_Cannon/milestone2/model/yolov5s_custom.engine"

    gpio = JetsonGpio()
    servo_pan = JetsonServo(channel=0)
    servo_tilt = JetsonServo(channel=1)
    calib = Calibration()
    time.sleep(3)
    yolo = YOLOv5TRTNoPyCUDA(model_path, input_hw=(640, 640))

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.connect((host, port))

    prev_time = time.time()

    try:
        while True:
            # TCP 메시지 처리
            hdr = recvall(server_socket, 8)
            if not hdr:
                print("[Server] Disconnected.")
                break
            image_size, message_type = struct.unpack(">II", hdr)
            data = recvall(server_socket, image_size)
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            frame = cv2.flip(frame, -1)

            # YOLO 추론 및 타겟 탐지
            detections = yolo.infer(frame)
            original_hw = frame.shape[:2]
            detections = yolo.postprocess(detections, original_hw)

            print(detections)

            # 타겟 추적 로직 실행
            target_stable = auto_targeting(detections, servo_pan, servo_tilt, calib, frame.shape)

            # FPS 계산 및 화면 출력
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            # 타겟 안정 상태에 따른 시각적 표시
            if target_stable:
                cv2.putText(
                    frame,
                    "Target Locked!",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            # 디텍션 결과 시각화
            for det in detections:
                # x1, y1, x2, y2, conf, cls_id = det
                x1, y1, x2, y2, conf = det
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Conf: {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            # 디스플레이 출력
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"[Error] {e}")

    finally:
        server_socket.close()
        cv2.destroyAllWindows()
        print("[Server] Shutting down.")

if __name__ == "__main__":
    main()