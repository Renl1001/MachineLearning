# -*- coding: utf-8 -*-

import numpy as np
import random

if __name__ == "__main__":

    file = open("train.txt", 'w')
    n = 1000
    for i in range(n):
        x = random.random()*20-10
        y = random.random()*20-10
        if(x**2+y**2 > 25):
            lb = 1
        else:
            lb = 0

        file.write('{}\t{}\t{}\n'.format(x, y, lb))

    file.close()

    file = open("test.txt", 'w')
    n = random.randint(100,200)
    for i in range(n):
        x = random.random()*20-10
        y = random.random()*20-10
        file.write('{}\t{}\n'.format(x, y))

    file.close()