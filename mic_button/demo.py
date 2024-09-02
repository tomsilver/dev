import pyaudio
import numpy as np
import time


def callback(data: bytes, frame_count: int, time_info: dict, status: int):
    data_arr = np.frombuffer(data, dtype=np.int16)
    # Calculate the volume (RMS)
    volume = np.sqrt(np.mean(data_arr**2))
    print(volume)

    # Threshold for detecting the button press
    if volume > 7:  # You may need to adjust this threshold
        print("Button Pressed", status)

    return (data, pyaudio.paContinue)



# Initialize PyAudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
  dev = p.get_device_info_by_index(i)
  print((i,dev['name'],dev['maxInputChannels']))

# Open stream
stream = p.open(
                format=pyaudio.paInt16,
                channels=1,  # The e-stop button is mono
                rate=48000,
                input=True,
                frames_per_buffer=4800,
                input_device_index=1,
                stream_callback=callback,
            )


time.sleep(2)
while True:
    pass
