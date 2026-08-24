#ifndef NMS_H
#define NMS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void decode_box_top(float* in0, float* in1, float* out, int ch);

void detector(float* in0, float* in1, int ch,
              float conf_thres, float nms_thres, float image_shape[2]);

#ifdef __cplusplus
}
#endif

#endif /* NMS_H */