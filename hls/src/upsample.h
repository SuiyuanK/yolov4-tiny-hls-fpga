#include"type.h"

// (c,size,size) -> (c,2*size,2*size)
void upsample(const axi_t* in, axi_t* out1, data_t* out2, int c, int size);