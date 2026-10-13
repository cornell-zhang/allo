import allo
from allo.ir.types import float32, int32, float64
import numpy as np

N = 1024


def fft(real: float64[N], img: float64[N], real_twid: float64[N//2], img_twid: float64[N//2]):

    log: int32 = 0
    odd: int32
    even: int32
    temp: float32 
    span: int32 = N >> 1
    rootindex: int32

    while span > 0:
        odd = span
        while odd < N:
            odd |= span
            even = odd ^ span

            temp = real[even] + real[odd]
            real[odd] = real[even] - real[odd]
            real[even] = temp

            temp = img[even] + img[odd]
            img[odd] = img[even] - img[odd]
            img[even] = temp

            rootindex = (even << log) & (N-1)

            if(rootindex != 0):
                temp = real_twid[rootindex] * real[odd] - img_twid[rootindex] * img[odd]
                
                img[odd] = real_twid[rootindex] * img[odd] + img_twid[rootindex] * real[odd]
                real[odd] = temp

        
            odd = odd + 1

        span = span >> 1
        log = log + 1
    

s = allo.customize(fft)
mod = s.build(target="llvm")