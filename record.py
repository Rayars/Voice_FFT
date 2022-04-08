import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


# import pyaudio
# import wave

# #AUDIO INPUT
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 44100
# CHUNK = 1024
# RECORD_SECONDS = 2
# WAVE_OUTPUT_FILENAME = "output.wav"

# audio = pyaudio.PyAudio()

# # start Recording
# stream = audio.open(format=FORMAT, channels=CHANNELS,
#                 rate=RATE, input=True,
#                 frames_per_buffer=CHUNK)
# while(1):
#   print "recording"
#   frames = []
#   for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#       data = stream.read(CHUNK)
#       frames.append(data)
#   waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
#   waveFile.setnchannels(CHANNELS)
#   waveFile.setsampwidth(audio.get_sample_size(FORMAT))
#   waveFile.setframerate(RATE)
#   waveFile.writeframes(b''.join(frames))
#   waveFile.close()
#   spf = wave.open(WAVE_OUTPUT_FILENAME,'r')

#   #Extract Raw Audio from Wav File
#   signal = spf.readframes(-1)
#   signal = np.fromstring(signal, 'Int16')   
#   copy= signal.copy()
# # stop Recording stream.stop_stream() stream.close() audio.terminate()