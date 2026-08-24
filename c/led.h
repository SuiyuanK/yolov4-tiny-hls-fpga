#ifndef LED_H
#define LED_H

#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LEDS_BASEADDR 0x1f20f020

#define LEDS_GPIO_DATA (LEDS_BASEADDR + 0x00)

// set leds pin
void setLedPin(uint32_t data);
void toggleLedPin(uint32_t data);

#ifdef __cplusplus
}
#endif

#endif 
