#ifndef YOLO_TIMER_H
#define YOLO_TIMER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "confreg_time.h"

static inline uint32_t yolo_timer_get_ticks(void)
{
    return (uint32_t)get_clock_count();
}

static inline uint32_t yolo_timer_ticks_to_us(uint32_t start_ticks, uint32_t end_ticks)
{
    const uint32_t delta_ticks = (uint32_t)(end_ticks - start_ticks);
    uint32_t ticks_per_us;

#ifdef USE_CPU_CLOCK_COUNT
    ticks_per_us = (uint32_t)(CORE_CLOCKS_PER_SEC / USEC_PER_SEC);
#else
    ticks_per_us = (uint32_t)(CONFREG_CLOCKS_PER_SEC / USEC_PER_SEC);
#endif

    if (ticks_per_us == 0u) {
        return 0u;
    }

    return delta_ticks / ticks_per_us;
}

#ifdef __cplusplus
}
#endif

#endif
