//
// Created by zach on 10/11/18.
//

#ifndef LAYERS_FIXED_H
#define LAYERS_FIXED_H

#include <stdint.h>
#include <math.h>

#include "common.h"

#ifdef TF_OP

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

#endif

//#define I_Part  1 // Fixed Point Integer Part
//#define F_Part 10 // Fraction Part


namespace layers_fixed {

// Convert the Float to the Fixed of integer/Fraction
int64_t Float_to_Fixed(double input, int64_t integer, int64_t fraction);

// Overflow Correction
int64_t OverflowCorrection(int64_t number, int64_t bit_number);

// Fixed Point to Float2 Conversion for multiplication
//double Fixed_to_Float(double number, int64_t Fraction);

// Fixed Point Multiplier
int64_t Fixed_Mul(double input1, double input2, int64_t integer, int64_t fraction);

// Fixed Point Accumulator
//double Fixed_ACC(const int64_t Product[], int64_t shape);

// === Start implementations ===

#ifdef TF_OP

using namespace tensorflow;

//template<typename idx>
//double Fixed_to_Float2(int64_t number, idx Fraction);

template<typename dtype, typename idx>
void fully_connected(const typename TTypes<dtype>::ConstMatrix Fc_fm_input,
                     const typename TTypes<dtype>::ConstMatrix Fc_weight,
                     const typename TTypes<dtype>::ConstVec bias_f,
                     typename TTypes<dtype>::Matrix Fc_fm_output,
                     idx n_output_row, idx n_output_column, idx n_output_middle,
                     idx ipart, idx fpart) {
    for (idx i = 0; i < n_output_row; ++i)
        for (idx j = 0; j < n_output_column; ++j) {
            Fc_fm_output(i, j) = bias_f(j); // each filter has one bias
            int64_t accSum = 0;
            for (int64_t k = 0; k < n_output_middle; ++k) {
                accSum += Fixed_Mul(Fc_fm_input(i, k), Fc_weight(k, j), ipart, fpart);
            }
            // Round and correct for overflow
            int64_t sticky = (accSum & ((1 << fpart) - 1)) ? 1 : 0;
            int64_t round_check = ((accSum >> (fpart - 1)) & 0x1) & (sticky | ((accSum >> fpart) & 0x1));
            accSum = OverflowCorrection((accSum >> fpart) + round_check, ipart + fpart);
            // Convert to double
            auto accSumF = static_cast<dtype>(accSum) / static_cast<dtype>(1 << fpart);
            Fc_fm_output(i, j) += accSumF;
        }
}

template<typename dtype, typename idx>
inline void round_ftype(const typename TTypes<dtype>::ConstFlat input, typename TTypes<dtype>::Flat output,
                        const idx integer, const idx fraction, const idx nElements) {
    dtype number_shift, number_round;
    int64_t number_ovf, number_int;
    int64_t bit_number = integer + fraction;
    auto shiftToFixed = static_cast<dtype>(1 << fraction);

    for (idx i = 0; i < nElements; ++i) {
        number_shift = input(i) * shiftToFixed;
        number_round = round(number_shift);
        number_int = static_cast<int64_t>(number_round);
        number_ovf = OverflowCorrection(number_int, bit_number);
        output(i) = static_cast<dtype>(number_ovf) / static_cast<dtype>(1 << fraction);
    }
}

//template<typename idx>
//double Fixed_to_Float2(const int64_t number, const idx Fraction) {
//    return static_cast<double>(number) * pow(2.0, -Fraction);
//}

#endif

//void fully_connected(const double Fc_weight[][output_column], const double Fc_fm_input[][output_middle],
//                     double Fc_fm_output[][output_column], const double bias_f[]) {
//    int64_t shape = output_middle;
//    static int64_t Product[output_middle] = {};
//
//    for (int64_t i = 0; i < output_row; i++)
//        for (int64_t j = 0; j < output_column; j++) {
//            Fc_fm_output[i][j] = bias_f[j]; // each filter has one bias
//            for (int64_t k = 0; k < output_middle; k++) {
//                Product[k] = Fixed_Mul(Fc_fm_input[i][k], Fc_weight[k][j], I_Part, F_Part);
//            }
//            Fc_fm_output[i][j] += Fixed_ACC(Product, shape);
//        }
//}

inline int64_t Float_to_Fixed(const double input, const int64_t integer, const int64_t fraction) {
    double number_shift; //, number_round;
//    int64_t number_ovf;
    int64_t number_int;
//    int64_t bit_number = integer + fraction;

    number_shift = input * static_cast<double>(1 << fraction);
//    number_round = round(number_shift);
    number_int = static_cast<int64_t>(number_shift);
//    number_ovf = OverflowCorrection(number_int, bit_number);

    return number_int;
}

inline int64_t OverflowCorrection(int64_t number, const int64_t bit_number) {
    int64_t shifted = (1 << (bit_number - 1));
    int64_t Max_number = shifted - 1;
    int64_t Min_number = -shifted;

    if (number > Max_number)
        number = Max_number;
    if (number < Min_number)
        number = Min_number;

    return number;
}

inline int64_t Fixed_Mul(const double input1, const double input2, const int64_t integer, const int64_t fraction) {
    int64_t Fixed_number1, Fixed_number2;

    Fixed_number1 = Float_to_Fixed(input1, integer, fraction);
    Fixed_number2 = Float_to_Fixed(input2, integer, fraction);
    return Fixed_number1 * Fixed_number2;
}

//double Fixed_ACC(const int64_t Product[], const int64_t shape) {
//    int64_t fraction = 10;  // TODO
//    int64_t MAC_Result = 0;
//    double MAC_Result1;
//
//    for (int64_t i = 0; i < shape; i++) {
//        MAC_Result += Product[i];
//    }
//    MAC_Result1 = Fixed_to_Float(MAC_Result, fraction);
//    return MAC_Result1;
//}

//double Fixed_to_Float(const double number, const int64_t Fraction) {
//    return number * pow(2, -1 * 2 * Fraction);
//}

}

#endif //LAYERS_FIXED_H
