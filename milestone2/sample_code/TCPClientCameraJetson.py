import socket
import cv2
import pickle
import struct
import numpy

def start_camera_stream_client(server_ip='192.168.1.20', port=10000):
    # Set up the socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, port))
    print(f"Connected to server {server_ip}:{port}")

    # Read read in two integers first
    
    #data = b""
    #payload_size = struct.calcsize("Q")

    def recvall(sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    while True:

        bfr = recvall(client_socket, 4)
        image_size = int.from_bytes(bfr,'big')
        #print(image_size)

        bfr = recvall(client_socket, 4)
        message_type = int.from_bytes(bfr,'big')
        #print(message_type)

        stringData = recvall(client_socket, image_size)
        if message_type==3:
            data = numpy.fromstring(stringData, dtype='uint8')
            decimg=cv2.imdecode(data,1)
            decimg = cv2.flip(decimg,-1)
            cv2.imshow('SERVER',decimg)
        else:
            print("Unknown message type")
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    client_socket.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_camera_stream_client()
