#include <iostream>
#include <fstream>

using namespace std;

#include "../src/maxpool.h"
//============================
// config
#define C 512
#define SIZE 26 //default 416
// K2 S2 P0 maxpool2d
#define K 2
#define S 2
#define P 0
//
#define C_OUT C
#define SIZE_OUT SIZE/K
//============================
int tb_maxpool() {
	// your files path
    const char* inputs_path  = "D:/SuiyuanBa/Desktop/AI/yolov4-tiny/yolov4-tiny-pytorch/HLS_pj/HLS_TB/maxpool2d/inputs.txt";
    const char* hls_path     = "D:/SuiyuanBa/Desktop/AI/yolov4-tiny/yolov4-tiny-pytorch/HLS_pj/HLS_TB/maxpool2d/hls.txt";
    // open file
    std::ifstream file2(inputs_path);
    std::ofstream file3(hls_path);
    if (!file2.is_open()) {
            cerr << "Failed to open file: " << inputs_path << std::endl;
            return 1;
        }
    if (!file3.is_open()) {
                std::cerr << "Failed to open file: " << hls_path << std::endl;
                return 1;
            }
    // define inputs array
    data_t inputs[C][SIZE][SIZE];
    // 1 read inputs
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

    // 2 Start compute
    // Convert a three-dimensional array to a one-dimensional array
    data_t *in        = reinterpret_cast<data_t*>(inputs);

    data_t *output_pt = (data_t*)malloc((C_OUT*SIZE_OUT*SIZE_OUT)*sizeof(data_t));

    maxpool((axi_t*)in,(axi_t*)output_pt,output_pt,C,SIZE,SIZE); // call operator

    // 3 Save results
    for (int w = 0; w < C_OUT*SIZE_OUT*SIZE_OUT; ++w) {
    	 data_t data = *((data_t*)output_pt + w);
    	 file3 << data;
    	 //cout<<data<< " ";
         file3<<endl; // Wrap to write next line of data
    }
    // 4 Close file
    file2.close();
    file3.close();

    cout<< "\n Maxpool test successfully!" <<endl;

    return 0;
}


int main(){
	int result=0;
	cout<<"input shape:"<<C<<" "<<SIZE<<" "<<SIZE<<endl;
	cout<<"output shape:"<<C_OUT<<" "<<SIZE_OUT<<" "<<SIZE_OUT<<endl;
	result = tb_maxpool();
	if(result==0){
		cout<< "\n Test PASS! :)" <<endl;
	}else{
		cout<< "\n Test Failed! :(" <<endl;
	}
	return 0;
}


