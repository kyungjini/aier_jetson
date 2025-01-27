import socket
import cv2
import numpy as np
import struct
import time

import tensorrt as trt
from cuda import cudart  # Requires: pip3 install nvidia-cuda-python

import posix_ipc
import mmap
import argparse

# 공유 메모리 설정
SHARED_MEMORY_NAME = "/object_detection"
MEMORY_SIZE = 1024  # 1KB


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


def start_camera_stream_client(
    server_ip="192.168.0.160",
    port=10000,
    engine_path="/home/group5/AI_Cannon/milestone2/model/yolov5s_custom.engine",
):
    # Shared Memory
    try:
        posix_ipc.unlink_shared_memory("/object_detection")
        posix_ipc.unlink_semaphore("/object_detection")
    except posix_ipc.ExistentialError:
        pass
    shared_memory = posix_ipc.SharedMemory(SHARED_MEMORY_NAME, posix_ipc.O_CREX, size=MEMORY_SIZE)
    semaphore = posix_ipc.Semaphore(SHARED_MEMORY_NAME, posix_ipc.O_CREX)
    
    memory_map = mmap.mmap(shared_memory.fd, MEMORY_SIZE)
    shared_memory.close_fd()

    # Define Yolo
    yolo = YOLOv5TRTNoPyCUDA(engine_path, input_hw=(640, 640))

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, port))
    print(f"Connected to server {server_ip}:{port}")

    def recvall(sock, count):
        buf = b""
        while count:
            newbuf = sock.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    prev_time = time.time()
    while True:
        bfr = recvall(client_socket, 4)
        if not bfr:
            print("Socket closed by server.")
            break
        image_size = int.from_bytes(bfr, "big")

        bfr = recvall(client_socket, 4)
        if not bfr:
            print("Socket closed by server.")
            break
        message_type = int.from_bytes(bfr, "big")

        stringData = recvall(client_socket, image_size)
        if not stringData:
            print("No image data.")
            break

        if message_type == 3:
            data = np.frombuffer(stringData, dtype=np.uint8)
            frame = cv2.imdecode(data, 1)
            frame = cv2.flip(frame, -1)
            if frame is None:
                continue

            original_hw = frame.shape[:2]
            raw_output = yolo.infer(frame)
            detections = yolo.postprocess(raw_output, original_hw)
            print("detection is: ", detections)

            binary_data = struct.pack('i' + 'f' * 5 * len(detections), len(detections), 
                                      *[value for bbox in detections for value in bbox])

            semaphore.acquire()
            memory_map.seek(0)
            memory_map.write(binary_data)
            semaphore.release()

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
            
            cv2.imshow("Detection", frame)

        else:
            print("Unknown message type:", message_type)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    memory_map.close()
    shared_memory.unlink()
    semaphore.unlink()
    client_socket.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera_stream_client(
        server_ip="192.168.0.160",  # raspberry pi ip
        port=10000,  # ./SendImageTCP 10000
        engine_path="/home/group5/AI_Cannon/milestone2/model/yolov5s_custom.engine",
    )
