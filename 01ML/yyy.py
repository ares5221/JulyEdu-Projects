#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import math
s1 = math.pow(0.6,5)
s2 = math.pow(0.4,5)
print(s1)
print(s2)
pa = 4*9*7*s1*s2

s3 = math.pow(0.5,5)
s4 = math.pow(0.5,5)
print(s3)
print(s4)
pb = 4*9*7*s3*s4

print(pa,pb)

print(pa/(pa+pb))