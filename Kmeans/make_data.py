# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random

if __name__ == "__main__":

    file = open("data.txt", 'w')
    n = random.randint(50,100)
    for i in range(n):
        x = random.random()*10
        y = random.random()*10
        file.write('{}\t{}\n'.format(x, y))

    file.close()