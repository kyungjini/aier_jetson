# sample_code
This folder provides a material for demonstration of TCP-based real-time object detection.

### SendImageTCP.cpp
- Compile the code using the `build.sh` script on the Raspberry Pi.
- Run the executable with the command: `./SendImageTCP <port_number>`

### TCPClientCameraJetson.py
- Update the script with the appropriate port number and IP address of the Raspberry Pi.
- Execute the script to receive a video stream from the Raspberry Pi’s camera.