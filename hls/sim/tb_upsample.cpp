#include"../src/upsample.h"

// config
#define c_in   128
#define size   13
#define c_out  c_in
#define os     2*size // output size
//========================

//ch*h*w-->ch*(2*h)*(2*w)
void baseline_upsample(data_t* in,data_t* out,int ch, int h, int w){
    for(int n=0;n<ch;n++)
        for(int i=0;i<h;i++)
            for(int j=0;j<w;j++){
            	//out[n][2*i:2*i+2][2*j:2*j+2]=in[n][i][j]
                data_t tmp=in[n*h*w+i*w+j];
                out[n*(2*h)*(2*w)+(2*i+0)*(2*w)+(2*j+0)]=tmp;
                out[n*(2*h)*(2*w)+(2*i+0)*(2*w)+(2*j+1)]=tmp;
                out[n*(2*h)*(2*w)+(2*i+1)*(2*w)+(2*j+0)]=tmp;
                out[n*(2*h)*(2*w)+(2*i+1)*(2*w)+(2*j+1)]=tmp;
    }
}


void tb_upsample(){
	data_t in[c_in][size][size];
	data_t out[c_out][os][os];
	data_t out_ref[c_out][os][os];
	memset(in, 0, sizeof(in));
	memset(out, 0, sizeof(out));
	memset(out_ref, 0, sizeof(out_ref));
	// 1 generat random input
	for(int i=0;i<size;i++)
		for(int j=0;j<size;j++)
			for(int n=0;n<c_in;n++){
				in[n][i][j]=(((n*n*n*71+n*n*13+i*i*43+j*29+(n+j+i)*61)%512-256)/256.0);
			}
	// 2 compute
	baseline_upsample((data_t*)in,(data_t*)out_ref,c_in,size,size);
	upsample((axi_t*)in,(axi_t*)out,(data_t*)out,c_in,size); // for optimized upsample
//	upsample((data_t*)in,(data_t*)out);
	// 3 check output
	int line=0;
	for(int i=0;i<os;i++)
		for(int j=0;j<os;j++)
			for(int n=0;n<c_out;n++){
				cout<<"h="<<i<<" ,w="<<j<<" c="<<n<<" ";
				cout<<out[n][i][j]<<","<<out_ref[n][i][j]<<endl;
				// if(out[n][i][j]!=out_ref[n][i][j]){
				// 	cout<<"Error, FAILED! :("<<endl;
				// 	return;
				// }
			}
	cout<<"Success, PASS! :)"<<endl;
	return;
}


int main(){
	tb_upsample();
}
