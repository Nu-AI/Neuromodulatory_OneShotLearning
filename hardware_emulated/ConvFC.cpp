// C++ version of implementing naive Convolutional layer and Fully Connected Layer. The code is functionally is worked
// version of 10/3/2018


#include <iostream>

using namespace std;

// Convolution Parameters
#define input_w 32// input width
#define input_h 32 // input height
#define filter_w 5 // filter width
#define filter_h 5 // filter height
#define stride 1 // stride
#define input_batch 32// Input_batch
#define batch_reorder 32 // batch_reorder
#define number_filter 32// number of filter
#define filter_channel 3 // number of channel
#define F 28
#define E 28

// Fully-Connected Parameters
#define output_row  10 // input_row
#define output_middle 15 // output_middle
#define output_column 20 // output_column


// Convolution Layer
void Convolutional_function(float weight[][filter_channel][filter_w][filter_h],float fm_input[][filter_channel][input_w][input_h],float fm_output[][number_filter][F][E],float bias_c[]);

// Reordering input from Tensorflow to C++ format
void reordering_inputs(float fm_input[][input_w][input_h][filter_channel], float fm_input_r[][filter_channel][input_w][input_h]);

// Reordering weights from Tensorflow to C++ format
void reordering_weights(float weight[][filter_w][filter_channel][number_filter], float weight_r[][filter_channel][filter_w][filter_h]);

// Reordeing outputs from Tensorflow to C++ format
void reordering_outputs(float fm_output[][number_filter][F][E], float fm_output_r[][F][E][number_filter]);

// Reording fully connected layer
void fully_connected(float Fc_weight[][output_column],float Fc_fm_input[][output_middle],float Fc_fm_output[][output_column], float bias_f[]);


int main() {
    cout << "Hello, World!" << endl;// parameters
    float weight[filter_h][filter_w][filter_channel][number_filter]={}; //[filter_h,filter_w,filter_channel,number_filter]
    float weight_r[number_filter][filter_channel][filter_w][filter_h]={}; // [number_filter,number_channel,filter_w,filter_h]
    float fm_input[input_batch][input_w][input_h][filter_channel]={}; // [input_batch,input_w,input_h,filter_channel]
    float fm_input_r[input_batch][filter_channel][input_w][input_h]={}; // [input_batch,filter_channel,input_w,input_h]
    float fm_output[input_batch][number_filter][F][E]={}; // [input_batch,number_filter,F,E]
    float fm_output_r[input_batch][F][E][number_filter]={}; // [input_batch,F,E,number_filter]
    float Fc_weight[output_middle][output_column]={};
    float Fc_fm_input[output_row][output_middle]={};
    float Fc_fm_output[output_row][output_column]={};
    float bias_c[number_filter]={};
    float bias_f[output_column]={};
    int flag=0;
    // operation reordering and convolution or fully-connected layer
    reordering_weights(weight,weight_r);
    reordering_inputs(fm_input,fm_input_r);
    if (flag==0)
        Convolutional_function(weight_r, fm_input_r, fm_output, bias_c);
    else
        fully_connected(Fc_weight, Fc_fm_input, Fc_fm_output, bias_f);


    reordering_outputs(fm_output,fm_output_r);
    cout<< fm_output[0][0][0][0];
    return 0;
}

void Convolutional_function(float weight[][filter_channel][filter_w][filter_h],float fm_input[][filter_channel][input_w][input_h], float fm_output[][number_filter][F][E],float bias_c[])

{
    //int F=0;
    //int E=0;
    //F = (input_w - filter_w + stride) / stride;
    //E = (input_h - filter_h + stride) / stride;
    //float fm_output[input_batch][number_filter][F][E];

    for(int n=0; n < input_batch; n++){
        for (int m=0; m < number_filter; m++){
            for(int x=0; x < F; x++) {
                for(int y=0; y < E; y++) {
                    fm_output[n][m][x][y]= bias_c[m];
                    for(int i=0; i < filter_w; i++) {
                        for(int j=0; j < filter_h; j++){
                            for(int k=0; k < filter_channel; k++) {
                                fm_output[n][m][x][y] +=  (fm_input[n][k][stride * x + i][stride * y + j] * weight[m][k][i][j]);
                            }
                        }
                    }

                }
            }
        }

    }

}

void reordering_weights(float weight[][filter_w][filter_channel][number_filter], float weight_r[][filter_channel][filter_w][filter_h])
{

    for(int i=0; i< number_filter; i++)
        for(int j=0; j< filter_channel; j++)
            for(int x=0; x< filter_w; x++)
                for(int y=0; y< filter_h; y++)
                    weight_r[i][j][x][y]= weight[x][y][j][i]; // [filter_h,filter_w,filter_channel,number_filter] // The weights order
}

void reordering_inputs(float fm_input[][input_w][input_h][filter_channel], float fm_input_r[][filter_channel][input_w][input_h])
{
    for(int i=0; i< input_batch; i++)
        for(int j=0; j< filter_channel; j++)
            for(int x=0; x< input_w; x++)
                for(int y=0; y< input_h; y++)
                    fm_input_r[i][j][x][y]= fm_input[i][x][y][j];  // [input_batch,input_w,input_h,filter_channel] // Tensorflow last channel
}

void fully_connected(float Fc_weight[][output_column], float Fc_fm_input[][output_middle],float Fc_fm_output[][output_column],float bias_f[])
{

     for(int i=0; i< output_row; i++)
         for(int j=0; j< output_column; j++ ) {
             Fc_fm_output[i][j] = bias_f[j]; // each filter has one bias
             for (int k = 0; k < output_middle; k++)
                 Fc_fm_output[i][j] += Fc_fm_input[i][k] * Fc_weight[k][j];
         }
}

void reordering_outputs(float fm_output[][number_filter][F][E], float fm_output_r[][F][E][number_filter])
{
    for(int i=0; i< input_batch; i++)
        for(int j=0; j< F; j++)
            for(int x=0; x< E; x++)
                for(int y=0; y< number_filter; y++)
                    fm_output_r[i][j][x][y]= fm_output[i][y][j][x];
}
