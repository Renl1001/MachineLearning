# -*- coding: utf-8 -*-

import numpy as np
import random

if __name__ == "__main__":

    file = open("train.txt", 'w')
    n = 200
    bag = [[-5,-5], [-5,5],[5,-5],[5,5]]
    for label in range(4):    
        for i in range(n):
            x = random.random()*5
            y = random.random()*5
            x += bag[label][0]
            y += bag[label][1]

            # if random.randint(0,10) < 1: # 随机加入干扰点
            #     lb = random.randint(0, 3)
            # else:
            #     lb = label
            lb = label
            file.write('{}\t{}\t{}\n'.format(x, y, lb))

    file.close()

    file = open("test.txt", 'w')
    n = random.randint(100,200)
    for i in range(n):
        x = random.random()*20-10
        y = random.random()*20-10
        file.write('{}\t{}\n'.format(x, y))

    file.close()