#ifndef TIMER_LOG_H
#define TIMER_LOG_H

#include <stdint.h>

/* Timer IDs: keep in sync with where they are used */
enum {
    TIMER_BASIC_CONV1 = 0,
    TIMER_BASIC_CONV2,
    TIMER_RESBLOCK1,
    TIMER_RESBLOCK2,
    TIMER_RESBLOCK3,
    TIMER_BASIC_CONV3,

    TIMER_CONV_FOR_P5,
    TIMER_YOLO_HEADP5,
    TIMER_UPSAMPLE_CONV,
    TIMER_UPSAMPLE,
    TIMER_YOLO_HEADP4,

    TIMER_POST_PROCESS,

    TIMER_MAX_ID
};

void log_timer_start(int id);
void log_timer_end(int id);
uint32_t timer_get_delta_ticks(int id);
void timer_clear_all(void);

#endif
