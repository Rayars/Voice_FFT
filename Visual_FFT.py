# 选择非阻塞模式的理由：因为要对语音数据进行实时处理，所以不能采用阻塞模式(blocking mode)，在回调函数中完成对语音信号的处理。
# 在回调函数中，回调函数会单独开辟一个线程。
# 处理的逻辑：处理流程是stream是一个流，实时获取麦克风语音数据，而回调函数会每次当有新的语音数据，会调用回调函数，传入frame_count帧的语音数据，回调函数会处理，处理完毕后，返回frame_count的数据，以及是否继续的标志。如果继续，即返回paContinue，当stream流依然活着，就继续调用回调函数，传入的数据是现在stream中的frame_count帧的语音数据。可以理解为stream是一直流动的，而回调函数只从这流动的线中，截取当前的一段距离进行处理。
# 为了保证主线程不死，在主线程中不断休眠。

import speech_recognition as sr
import librosa.display
# from pandas import Interval
import pyaudio
#import os     用于调试的模块
import tkinter as tk
import wave
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as line
import numpy as np
from scipy import fftpack
from scipy import signal
import librosa

CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "//home//focal//PyAudio//visual_output.wav"
# 这是我的当前路径，要根据计算机情况自己设定
data = []
Recording = True
FFT_LEN = 128
frames = []
counter = 1

fig = plt.figure()  
rt_ax = plt.subplot(212, xlim=(0, CHUNK), ylim=(-10000, 10000))  # 设置界面范围
fft_ax = plt.subplot(211)
fft_ax.set_yscale('log')
fft_ax.set_xlim(0, CHUNK / 2 + 1)  # x范围
fft_ax.set_ylim(1, 100000000)  # y范围
rt_ax.set_title("Real Time")
fft_ax.set_title("FFT Time")
rt_line = line.Line2D([], [])
fft_line = line.Line2D([], [])

rt_data = np.arange(0, CHUNK, 1)  # 0~CHUNK的数组
fft_data = np.arange(0, CHUNK / 2 + 1, 1)  # 0~CHUNK/2 + 1的数组
rt_x_data = np.arange(0, CHUNK, 1)  # x轴0~1024
fft_x_data = np.arange(0, CHUNK / 2 + 1, 1)  # x轴0~512

mfcc_fig=plt.figure()
mfccs=np.arange(0,39).reshape(13,3)

# 参数设定
def plot_init():
    rt_ax.add_line(rt_line)
    fft_ax.add_line(fft_line)
    return fft_line, rt_line,


# 更新数据
def plot_update(i):
    global rt_data
    global fft_data

    rt_line.set_xdata(rt_x_data)
    rt_line.set_ydata(rt_data)

    fft_line.set_xdata(fft_x_data)
    fft_line.set_ydata(fft_data)
    return fft_line, rt_line,

def mfcc_init():
    global mfcc_fig
    mfcc_fig=plt.figure(figsize=(25,10))
    plt.colorbar(format="%+2.f")
    return mfcc_fig

def mfcc_update():
    global mfccs
    librosa.display.specshow(mfccs,x_axis="time",sr=RATE)
    return mfcc_fig


ani = animation.FuncAnimation(fig, plot_update,
                              init_func=plot_init,
                              frames=1,
                              interval=30,
                              blit=True)

# ani = animation.FuncAnimation(mfcc_fig,mfcc_update,
#                               init_func=mfcc_init,
#                               frames=1,
#                               interval=30,
#                               blit=True)

def mfcc_show():
    global rt_data
    global mfcc_fig
    mfccs=librosa.feature.mfcc(rt_data,n_mfcc=13,sr=RATE)
    mfcc_fig=plt.figure(figsize=(25,10))
    librosa.display.specshow(mfccs,x_axis="time",sr=RATE)
    plt.colorbar(format="%+2.f")
    plt.show()

p = pyaudio.PyAudio()  # 初始化
q = queue.Queue()  # 生成队列

def audio_callback(in_data, frame_count, time_info, status):
    global ad_rdy_ev

    q.put(in_data)  # 放入数据
    ad_rdy_ev.set()  # ad_rdy_ev内部有一Boolean值，set()将其置true
    if counter <= 0:
        return (None, pyaudio.paComplete)  # 无数据，结束标志，返回paComplete
    else:
        return (None, pyaudio.paContinue)  # 仍有数据未读入，返回paContinue


# pyaudio.paComplete = 1
# This was the last block of audio data
# pyaudio.paContinue = 0
# There is more audio data to come


# 新建一个流Open a new stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=False,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback)

if Recording:
#    print('有记录')   用于调试
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)  # 声道
    wf.setsampwidth(p.get_sample_size(FORMAT))  # 采样字节
    wf.setframerate(RATE)  # 采样频率

print("Start Recording")
stream.start_stream()  # Start the stream.

# processing block

window = signal.hamming(CHUNK)  # 返回hamming window，hamming window 为：w(n)=0.54−0.46cos(2πnM−1)0≤n≤M−1


def read_audio_thread(q, stream, frames, ad_rdy_ev):
    global rt_data
    global fft_data
    global mfccs

    while stream.is_active():  # is_active():Returns whether the stream is active.
        ad_rdy_ev.wait(timeout=1000)
        if not q.empty():
            # process audio data here
            data = q.get()  # data得到数据
            # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            #    data = stream.read(CHUNK)
            while not q.empty():
                q.get()
            rt_data = np.frombuffer(data, np.dtype('<i2'))  # 将data转化为一维数组
            rt_data = rt_data * window
            fft_temp_data = fftpack.fft(rt_data, rt_data.size, overwrite_x=True)  # 傅里叶变换
            fft_data = np.abs(fft_temp_data)[0:fft_temp_data.size // 2 + 1]  # 绝对值+向下取整
            mfccs=librosa.feature.mfcc(rt_data,n_mfcc=13,sr=RATE)         
            if Recording:
                frames.append(data)
        ad_rdy_ev.clear()


ad_rdy_ev = threading.Event()  # 创建一对象

t = threading.Thread(target=read_audio_thread, args=(q, stream, frames, ad_rdy_ev))  # 构造函数

t.daemon = True  # 是否为守护进程
t.start()  # 进程开始

# 显示图像界面
plt.show()
mfcc_show()

# 结束进程
stream.stop_stream()
stream.close()
p.terminate()

print("* done recording")
if Recording:
    wf.writeframes(b''.join(frames))
    wf.close()


#用于调试   print(os.getcwd())     #获取路径，用于检查是否在当前目录下