import wave
import keyboard

from pyaudio import PyAudio, paInt16

# const
CHUNK = 512
CHANNELS = 1
RECORD_TIME = 6
SAMPLE_RATE = 16000
SAMPLE_FORMAT = paInt16

def recorder(fname):
    pa = PyAudio()

    stream = pa.open(
        input=True,
        rate=SAMPLE_RATE,
        channels=CHANNELS,
        input_device_index=0,
        format=SAMPLE_FORMAT,
        frames_per_buffer=CHUNK,
    )

    frames = list()

    for _ in range(int(SAMPLE_RATE / CHUNK * RECORD_TIME)):
        data = stream.read(CHUNK)
        frames.append(data)

        if keyboard.is_pressed('q') or keyboard.is_pressed('Q'):
            break

    stream.stop_stream()
    stream.close()
    pa.terminate()

    if not frames:
        return False

    # Save wave file
    wf = wave.open(fname, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(SAMPLE_FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close

    return True