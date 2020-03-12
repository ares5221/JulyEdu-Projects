#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# 建立一个基于目标行业的database
# 比如 这里我们用python自带的graph
graph = {'上海': ['苏州', '常州'],
         '苏州': ['常州', '镇江'],
         '常州': ['镇江'],
         '镇江': ['常州'],
         '盐城': ['南通'],
         '南通': ['常州']}

# 明确如何找到从A到B的路径
def find_path(start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if start not in graph:
        return None
    for node in graph[start]:
        if node not in path:
            newpath = find_path(node, end, path)
            if newpath: return newpath
    return None
print(find_path('上海', "镇江"))

from gtts import gTTS
import os
tts = gTTS(text='您好，我是您的私人助手，我叫小辣椒', lang='zh-tw')
tts.save("hello.mp3")
os.system("mpg321 hello.mp3")

import speech_recognition as sr
from time import ctime
import time
import os
from gtts import gTTS
import sys


# 讲出来AI的话
def speak(audioString):
    print(audioString)
    tts = gTTS(text=audioString, lang='en')
    tts.save("audio.mp3")
    os.system("mpg321 audio.mp3")


# 录下来你讲的话
def recordAudio():
    # 用麦克风记录下你的话
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)

    # 用Google API转化音频
    data = ""
    try:
        data = r.recognize_google(audio)
        print("You said: " + data)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

    return data


# 自带的对话技能（rules）
def jarvis():
    while True:

        data = recordAudio()

        if "how are you" in data:
            speak("I am fine")

        if "what time is it" in data:
            speak(ctime())

        if "where is" in data:
            data = data.split(" ")
            location = data[2]
            speak("Hold on Tony, I will show you where " + location + " is.")
            os.system("open -a Safari https://www.google.com/maps/place/" + location + "/&amp;")

        if "bye" in data:
            speak("bye bye")
            break


# 初始化
time.sleep(2)
speak("Hi Tony, what can I do for you?")

# 跑起
jarvis()