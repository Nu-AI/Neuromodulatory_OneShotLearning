import sys, platform
import ctypes, ctypes.util

import numpy as np

path_libc = ctypes.util.find_library("c")

try:
    libc = ctypes.CDLL(path_libc)
except OSError:
    print ('Cannot load the library')
    sys.exit()

print (f'was able to load the c library from "{path_libc}"')
