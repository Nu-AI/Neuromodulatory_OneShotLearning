import sys, platform
import ctypes, ctypes.util
#import ConvFC
from ctypes import cdll
import numpy as np


#path_libc = ctypes.util.find_library("ConvFC")
Fixedlib = ctypes.CDLL("FixedPoint.so")
#getattr(cdll.msvcrt)
#Fixedlib.main()

Fixedlib.Float_to_Fixed.restype = ctypes.c_int32
Fixedlib.Float_to_Fixed.argtypes = (ctypes.c_float, ctypes.c_int, ctypes.c_int)
def Float_to_Fixed(number, integer, fraction):
    results = Fixedlib.Float_to_Fixed(number, integer, fraction)
    return(results)

__main__
try:
    libc = ctypes.CDLL(path_libc)
except OSError:
    print ('Cannot load the library')
    sys.exit()
#Convlib = cdll.LoadLibrary("ConvFC.so")
#Convlib.main()
float_to_fixed = Float_to_Fixed(0.7,1,10)
print (float_to_fixed)
#print (Fixedlib.Float_to_Fixed(7.2345,3,7))
#print (f'was able to load the c library from')
