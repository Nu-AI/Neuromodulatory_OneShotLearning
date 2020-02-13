import numpy as np
class  meoww:
    def __init__(self, temp_variable):
        self.temp_variable = temp_variable

    def catty (self,temp2, image):
        return (self.temp_variable + image)

    def kitty(self, image):
        image  = 10
        tempvar = self.catty(self.temp_variable,image)
        return tempvar

a = meoww(10)
print (a.catty(2,25))
print (a.kitty(15))
