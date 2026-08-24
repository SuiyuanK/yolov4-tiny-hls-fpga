#include <iostream>
#include <fstream>
using namespace std;

#include "../src/top.h"

//===============================
// config
#define C 32   // input channel
#define SIZE 208

#define C_OUT 64  // output channel
#define SIZE_OUT SIZE/S
#define OUT_SIZE C_OUT*SIZE_OUT*SIZE_OUT
//===============================
// // K3 P1 S1
// #define K 3
// #define P 1
// #define S 1

// K3 P1 S2
#define K 3
#define P 1
#define S 2

// // K1 P0 S1 (pwconv)
// #define K 1
// #define P 0
// #define S 1


int tb_conv2d() {
    // file path
	const char* weights_path = "D:/SuiyuanBa/Desktop/AI/yolov4-tiny/yolov4-tiny-pytorch/HLS_pj/HLS_TB/conv2d/weight.txt";
	const char* bias_path    = "D:/SuiyuanBa/Desktop/AI/yolov4-tiny/yolov4-tiny-pytorch/HLS_pj/HLS_TB/conv2d/bias.txt";
    const char* inputs_path  = "D:/SuiyuanBa/Desktop/AI/yolov4-tiny/yolov4-tiny-pytorch/HLS_pj/HLS_TB/conv2d/inputs.txt";
    const char* hls_path     = "D:/SuiyuanBa/Desktop/AI/yolov4-tiny/yolov4-tiny-pytorch/HLS_pj/HLS_TB/conv2d/hls.txt";
    // open file
    std::ifstream file1(weights_path);
    std::ifstream file_bias(bias_path);
    std::ifstream file2(inputs_path);
    std::ofstream file3(hls_path);
    if (!file1.is_open()) {
        cerr << "Failed to open file: " << weights_path << std::endl;
        return 1;
    }
    if (!file2.is_open()) {
        cerr << "Failed to open file: " << inputs_path << std::endl;
        return 1;
    }
    if (!file3.is_open()) {
       std::cerr << "Failed to open file: " << hls_path << std::endl;
       return 1;
    }
    if (!file_bias.is_open()) {
    	std::cerr << "Failed to open file: " << bias_path << std::endl;
        return 1;
    }

    // define input and weight buffer
    cout<<"TB B1"<<endl;//for debug
    data_t inputs[C][SIZE][SIZE];
    data_t weights[C_OUT][C][K][K];
    cout<<"TB B2"<<endl;//for debug

    // 1.1 read weight file and assign values to the weights array
        for (int n = 0; n < C_OUT; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < K; ++h) {
                    for (int w = 0; w < K; ++w) {
                        if (file1.eof()) {
                            std::cerr << "Reached end of file before filling the entire array." << std::endl;
                            return 1;
                        }
                        data_t temp;
                        file1 >> temp;
                        //std::cout << temp << " ";
                        weights[n][c][h][w] = static_cast<data_t>(temp);
                        //cout << weights[n][c][h][w] << " ";
                    }
                }
            }
        }
    cout<<"tb_conv2d B0"<<endl;
    // 1.2 read ifm file and assign values to the inputs array
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < SIZE; ++h) {
            for (int w = 0; w < SIZE; ++w) {
                if (file2.eof()) {
                    std::cerr << "Reached end of file before filling the entire array." << std::endl;
                    return 1;
                }
                file2 >> inputs[c][h][w];
                //cout << inputs[c][h][w] << " ";
            }
        }
    }
    // 1.3 read bias
    data_t bias[C_OUT];
    /*
    for (int c = 0; c < C_OUT; ++c) {
    	bias[c] = 0;
    }*/
    for (int c=0; c<C_OUT; ++c) {
    	if (file_bias.eof()) {
    		std::cerr << "Reached end of file before filling the entire array." << std::endl;
            return 1;
        }
    	file_bias >> bias[c];
        cout<<"bias:"<<bias[c]<<"\n";
    }
    // 2 compute
    // Convert a three-dimensional array to a one-dimensional array
    data_t *in        = reinterpret_cast<data_t*>(inputs);
    data_t *weight    = reinterpret_cast<data_t*>(weights);
    data_t *bias_pt   = reinterpret_cast<data_t*>(bias);

    data_t *output_pt = (data_t*)malloc((OUT_SIZE)*sizeof(data_t));
    for (int w = 0; w < OUT_SIZE; ++w) {
    	output_pt[w] = -1;
    }
    /*
    // print in
    for (int i = 0; i < C * SIZE * SIZE; i++) {
       cout << in[i] << " ";
    }*/
    cout<<"tb_conv2d B1"<<endl;
    conv((axi_t*)in,
         (axi_t*)weight,
    	 (axi_t*)bias_pt,
    	 (axi_t*)output_pt,
         output_pt,
    	 C,C_OUT,SIZE,S,K,0,128,13);
    cout<<"tb_conv2d B2"<<endl;
    // 3 Save results
    for (int w = 0; w < OUT_SIZE; ++w) {
    	 data_t data = *((data_t*)output_pt + w);
    	 file3 << data;
    	 //cout<<data<< " ";
         file3<<endl; // Wrap to write next line of data
    }
    // 4 close files
    file1.close();
    file2.close();
    file3.close();
    //
    cout<< "\n Conv2D test successfully!" <<endl;
    /*
    // ÊÍ·ÅÄÚ´æ
    delete[] inputs;
    delete[] weights;
    delete[] bias_pt;
    delete output_pt;*/

    return 0;
}


int main(){
	int result=0;
	cout<<"input shape:"<<C<<" "<<SIZE<<" "<<SIZE<<endl;
	cout<<"output shape:"<<C_OUT<<" "<<SIZE_OUT<<" "<<SIZE_OUT<<endl;
	result = tb_conv2d();
	if(result==0){
		cout<< "\n Conv2D PASS! :)" <<endl;
	}else{
		cout<< "\n Conv2D Failed! :(" <<endl;
	}
	return 0;
}


