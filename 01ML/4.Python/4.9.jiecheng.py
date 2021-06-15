#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import sys
import threading
sys.setrecursionlimit(20000)  # 这里设置大一些
def FirstFactorial(num):
    if num == 0:
        return 1
    else:
        return num * FirstFactorial(num - 1)



if __name__ == '__main__':
    # keep this function call here
    # threading.stack_size(200000000)
    # thread = threading.Thread(target=)
    # thread.start()
    print("%d" %FirstFactorial(10000))