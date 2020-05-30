#
# Program to test running the script
# Also test that Pandas and NumPy are
#   available.
#
print("Hello, Welcome to Python Capstone ")

import numpy as np
# Standard printing
print(np.__version__)

import pandas as pd
print('We are using {} version {}'.format('Pandas',pd.__version__))
# {} gets replaced with arguments from format

for aaa in range(1,21):
    print("%2d --- %3d"%(aaa,aaa**2))
# positional substituition

import math
print('The value of pi is %5.3f'%math.pi)
# formatting like C language