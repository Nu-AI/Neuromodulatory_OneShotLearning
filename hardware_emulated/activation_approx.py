import numpy as np

a  =np.linspace(-3,3, 500)
print (a.shape)

import matplotlib.pyplot as plt

# plt.plot(a,np.tanh(a))
# plt.grid(linestyle='-')
# plt.show()


class approx_activation:
    def __init__(self,activation):
        self.activation = activation

        #However this function has high error rates, need to find a better function
    def approx_tanh(input):

        if (input > 3):
            return 1
        elif ( input > -3 and input < -1 ):
            return (0.11673*input - 0.64486)
        elif ( input >= -1 and input<0 ):
            return 0.76159*input
        elif ( input >=0  and input < 2 ):
            return 0.4820*input

        else:
            return -1


    def approx_sigmoid(self,input):

        # Lets decide first which sub plots to break it down to
        if ( abs(input) >= 5 ):
            return 1
        elif( abs(input) < 5 and abs(input) >= 2.375):
            return (0.03125*input + 0.84375)
        elif( abs(input) < 2.375 and abs(input) >= 1):
            return (0.125*input + 0.625)
        else:
            return 1

    def approx_sigmoid_2(self,input):
        if (input <= -2):
            return 0
        elif (input > -2 and input <2):
            return (input/4 + 0.5)
        else:
            return 1

    def approx_tanh_2(self,input):
        return 2*(approx_sigmoid_2(2*input)) - 1


    def approx_tanh_3(self,input):
        out = 0
        if abs(input) >= 3:
            out =  1
        elif (abs(input)>=1.5 and abs(input) <3):
            out =  0.06*abs(input) + 0.815
        elif(abs(input)>=0.5 and abs(input)<1.5):
            out =  0.443*abs(input) + 0.24
        elif (abs(input)>=0 and abs(input) <0.5):
            out =  0.924*abs(input)
        else:
            out = 1
        if input < 0:
            return -1*out
        else:
            return out

    def apply_act(self,input):
        if (self.activation =='tanh'):
            return self.approx_tanh_3(input)
        else:
            return self.approx_sigmoid_2(input)
