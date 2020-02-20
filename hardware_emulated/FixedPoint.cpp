// The Code For Fixed Point Function and the rounding mode is round to nearest and trancate for overflow and underflow
//

#include <stdint.h>
#include <iostream>
#include <math.h>

using namespace std;

int32_t Float_to_Fixed(float number, int integer, int fraction);
int32_t OverflowCorrection(int number,int bit_number);
float Fixed_to_Float(float input, int integer, int Frcation);
float Fixed_to_Float2(float input, int integer, int Frcation);
int32_t Fixed_Mul(float input1, float input2, int fraction, int integer);
float Fixed_ACC(float Product[], int shape);


// int main()
// {
//     float number1=0.4;
//     float number2=0.6;
//     float number3=0.5;
//     float number4=0.7;
//     float Result1= 0.0;
//     float Result2= 0.0;
//     float Result3= 0.0;
//     float float_convert;
//     float Product[2];
//     Result1=Fixed_Mul(number1,number2,1,10);
//     cout<< Result1 << endl;
//     Result2=Fixed_Mul(number3,number4,1,10);
//     cout<< Result2 << endl;
//     Product[0]= Result1;
//     Product[1]=Result2;
//     Result3= Fixed_ACC(Product, 2);
//     int temp_result;
//     temp_result = Float_to_Fixed(number4,1,10);
//     cout<< temp_result<< endl;
//     cout << "the temp result is above" << endl;
//     float_convert = Fixed_to_Float(temp_result, 1, 10);
//     cout << float_convert<< endl;
//     cout<< Result3 << endl;
//
//     return 0;
// }

int32_t Float_to_Fixed(float input, int integer, int fraction)
{
    float number = 0.0;
    float number_shift = 0.0;
    float number_round = 0.0;
    float number_int = 0.0;
    float number_ovf = 0.0;
    int bit_number = integer + fraction;
    number= input;
    //cout<< number << endl;
    number_shift= number * pow(2.0, fraction);
    //cout<< number_shift<<endl;
    number_round= round(number_shift);
    //cout<< number_round<< endl;
    number_int = int16_t(number_round);
    number_ovf=  OverflowCorrection(number_int, bit_number);
    return number_ovf;
}

int32_t OverflowCorrection(int number,int bit_number)
{
    int Max_number= 0;
    int Min_number= 0;

    Max_number= pow(2, (bit_number-1))-1;
    Min_number= -1 * pow(2, (bit_number-1));
    if ( number > Max_number)
        number= Max_number;
    if (number < Min_number)
        number = Min_number;
    return number;
}

float Fixed_to_Float2(float number, int integer, int Fraction)
{
    return float(number * pow(2,-1*2*Fraction));
}

float Fixed_to_Float(float number, int integer, int Fraction)
{
    return float(number * pow(2,-1*Fraction));
}

int32_t Fixed_Mul(float input1, float input2, int integer, int fraction)
{
    //int fraction = fraction;
    //int integer= integer ;
    int Fixed_number1=0;
    int Fixed_number2=0;
    int multiply1=0;
    float multiply=0.0;
    Fixed_number1 = Float_to_Fixed(input1, integer, fraction);
    //cout<< Fixed_number1<<endl;
    Fixed_number2 = Float_to_Fixed(input2, integer, fraction);
    //cout<< Fixed_number2<<endl;
    multiply1 = Fixed_number1 * Fixed_number2;
    cout<< multiply1<<endl;
    //multiply = Fixed_to_Float(multiply1, integer, fraction);
    //cout<< multiply<< endl;
    return multiply1;
}

float Fixed_ACC(float Product[], int shape)
{
    int integer= 1;
    int fraction= 10;
    int32_t MAC_Result = 0;
    float MAC_Result1 = 0.0;
    float MAC_Result2= 0.0;
    float MAC=0.0;
    int16_t temp= 0;
    //int temp=0;
    for(int i=0; i< shape; i++) {
        //temp = Float_to_Fixed(Product[i], integer, fraction);
        //MAC_Result += temp;
        MAC_Result += Product[i];
    }
    MAC_Result1 = Fixed_to_Float2(MAC_Result, integer, fraction);
    //temp = Float_to_Fixed(MAC_Result1, integer,fraction);
    //MAC_Result2 = Fixed_to_Float(temp, integer, fraction);
    return MAC_Result1;
}
