#ifndef SD_DRIVER_H
#define SD_DRIVER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int sd_dma_read_sectors(unsigned int ddr_dst_addr,
						unsigned int start_sector,
						unsigned int sector_num);
unsigned int sd_dma_get_progress_bytes(void);
int read_params(unsigned int start_sector, int16_t *dst, unsigned int length);

#ifdef __cplusplus
}
#endif

#endif
