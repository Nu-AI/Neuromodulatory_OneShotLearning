import sys, platform
import ctypes, ctypes.util
from ctypes import cdll
import numpy as np

Fixedlib = ctypes.CDLL("FixedPoint.so")

Fixedlib.Float_to_Fixed.restype = ctypes.c_int32
Fixedlib.Float_to_Fixed.argtypes = (ctypes.c_float, ctypes.c_int, ctypes.c_int)

Fixedlib.Fixed_to_Float.restype = ctypes.c_float
Fixedlib.Fixed_to_Float.argtypes = (ctypes.c_float, ctypes.c_int)

Fixedlib.Fixed_to_Float2.restype = ctypes.c_float
Fixedlib.Fixed_to_Float2.argtypes = (ctypes.c_float, ctypes.c_int)

Fixedlib.Fixed_Mul.restype = ctypes.c_int32
Fixedlib.Fixed_Mul.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int)

Fixedlib.Fixed_ACC.restype = ctypes.c_float
Fixedlib.Fixed_ACC.argtypes = (np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int)

def Float_to_Fixed(number, integer, fraction):
    result = Fixedlib.Float_to_Fixed(number, integer, fraction)
    return result

def Fixed_to_Float(number, fraction):
    result = Fixedlib.Fixed_to_Float(number, fraction)
    return result

def Fixed_to_Float2(number, fraction):
    result = Fixedlib.Fixed_to_Float2(number, fraction)
    return result

def Fixed_Mul(input1, input2, integer, fraction):
    result = Fixedlib.Fixed_Mul(input1, input2, integer, fraction)
    return result

def Fixed_ACC(Product, shape):
    result = Fixedlib.Fixed_ACC(Product, shape)
    return result

float_to_fixed = Float_to_Fixed(0.7,1,8)
print (float_to_fixed)

fixed_to_float = Fixed_to_Float(float_to_fixed, 8)
print (fixed_to_float)

mult_result = Fixed_Mul(0.4,0.6,1,10)
mult_result2 = Fixed_Mul(0.5,0.7,1,10)

print (mult_result, mult_result2)

updated_mult_result = Fixed_to_Float2(mult_result,10)
updated_mult_result2 = Fixed_to_Float2(mult_result2,10)
print(updated_mult_result, updated_mult_result2)

a = np.array([mult_result,mult_result2], dtype=np.float32)
acc_result = Fixed_ACC(a,2)
print (acc_result)
