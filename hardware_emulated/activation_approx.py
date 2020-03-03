import numpy as np

a  =np.linspace(-2,2, 500)
print (a.shape)

import matplotlib.pyplot as plt

plt.plot(a,np.tanh(a))
plt.grid(linestyle='-')
plt.show()


#However this function has high error rates, need to find a better function
def approx_tanh(input):

    if (input > 3):
        return 1
    elif ( input > -3 and input < -1 ):
        return (0.11673*input - 0.64486)
    elif ( input > -1 and input<0 ):
        return 0.76159*input
    elif ( input >0  and input < 2 ):
        return 0.4820*input
    else:
        return -1
