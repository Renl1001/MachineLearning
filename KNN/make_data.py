# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random

if __name__ == "__main__":

    file = open("data.txt", 'w')
    n = random.randint(100,500)
    for i in range(n):
        x = random.random()*10
        y = random.random()*10
        if x > y:
            if random.randint(0,10) < 1: # 随机加入干扰点
                label = 0
            else:
                label = 1
        else:
            if random.randint(0,10) < 1: # 随机加入干扰点
                label = 1
            else:
                label = 0
        file.write('{}\t{}\t{}\n'.format(x, y, label))

    file.close()