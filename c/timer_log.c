#include "timer_log.h"
#include "confreg_time.h"

#include <stdint.h>

static uint32_t start_ticks[TIMER_MAX_ID];
static uint32_t end_ticks[TIMER_MAX_ID];
static uint8_t started[TIMER_MAX_ID];

void log_timer_start(int id)
{
    if (id < 0 || id >= TIMER_MAX_ID) return;
    started[id] = 1;
    start_ticks[id] = get_clock_count();
}

void log_timer_end(int id)
{
    if (id < 0 || id >= TIMER_MAX_ID) return;
    if (!started[id]) return;
    end_ticks[id] = get_clock_count();
}

uint32_t timer_get_delta_ticks(int id)
{
    if (id < 0 || id >= TIMER_MAX_ID) return 0;
    return (uint32_t)(end_ticks[id] - start_ticks[id]);
}

void timer_clear_all(void)
{
    for (int i = 0; i < TIMER_MAX_ID; i++) {
        started[i] = 0;
        start_ticks[i] = 0;
        end_ticks[i] = 0;
    }
}
