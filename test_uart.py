import serial
import time

try:
    # Open the serial port
    ser = serial.Serial('/dev/serial0', 115200, timeout=1)
    time.sleep(1) # Let the connection settle
    print("UART Port Opened Successfully!")

    # Send a quick test sequence (e.g., U2 and the '!' terminator)
    test_message = "UU!\n"
    print(f"Sending to Nucleo: {test_message.strip()}")
    ser.write(test_message.encode('utf-8'))

    # Wait for the Nucleo to send "DONE" back
    print("Listening for Nucleo response...")
    while True:
        if ser.in_waiting > 0:
            response = ser.readline().decode('utf-8').strip()
            print(f"Nucleo replied: {response}")
            break
        time.sleep(0.1)

    ser.close()

except Exception as e:
    print(f"Connection failed: {e}")
