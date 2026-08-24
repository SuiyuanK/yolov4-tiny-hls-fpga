#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "common_func.h"
#include "sd_driver.h"

#define SD_AXI_LITE_BASE_ADDR (0x1F400000u)

#define SD_REG_CTRL           (0x00u)
#define SD_REG_STAT           (0x04u)
#define SD_REG_DSTADDR        (0x08u)
#define SD_REG_STARTSECTOR    (0x0Cu)
#define SD_REG_SECTORNUM      (0x10u)
#define SD_REG_PROGRESS       (0x14u)
#define SD_REG_RESET          (0x18u)

#define SD_CTRL_START         (0x1u)
#define SD_STAT_BUSY_MASK     (0x1u)
#define SD_STAT_DMA_ERR_MASK  (0x2u)
#define SD_WAIT_TIMEOUT       (20000000u)

#ifndef SD_LOG_LEVEL
#define SD_LOG_LEVEL 0
#endif

#define SD_LOGE(...) printf(__VA_ARGS__)
#if SD_LOG_LEVEL >= 1
#define SD_LOGI(...) printf(__VA_ARGS__)
#else
#define SD_LOGI(...)
#endif
#if SD_LOG_LEVEL >= 2
#define SD_LOGD(...) printf(__VA_ARGS__)
#else
#define SD_LOGD(...)
#endif

static unsigned int sd_reg_addr(unsigned int offset)
{
    return SD_AXI_LITE_BASE_ADDR + offset;
}

static int sd_wait_idle(unsigned int timeout)
{
    while ((RegRead(sd_reg_addr(SD_REG_STAT)) & SD_STAT_BUSY_MASK) != 0u) {
        if (timeout == 0u) {
            return -1;
        }
        timeout--;
    }
    return 0;
}

int sd_dma_read_sectors(unsigned int ddr_dst_addr,
                        unsigned int start_sector,
                        unsigned int sector_num)
{
    unsigned int stat;

    // 扇区数必须为偶数
    if (sector_num == 0u) {
        SD_LOGE("SD invalid argument: sector_num is 0\n");
        return -4;
    }

    SD_LOGD("========== SD DMA start ==========\n");
    SD_LOGD("AXI-Lite base: 0x%08X\n", SD_AXI_LITE_BASE_ADDR);
    SD_LOGD("start sector: %u, sectors: %u, dst: 0x%08X\n",
            start_sector, sector_num, ddr_dst_addr);

    RegWrite(sd_reg_addr(SD_REG_RESET), 1u);
    if (sd_wait_idle(SD_WAIT_TIMEOUT) != 0) {
        SD_LOGE("SD timeout after reset\n");
        return -1;
    }

    RegWrite(sd_reg_addr(SD_REG_DSTADDR), ddr_dst_addr);
    RegWrite(sd_reg_addr(SD_REG_STARTSECTOR), start_sector);
    RegWrite(sd_reg_addr(SD_REG_SECTORNUM), sector_num);
    RegWrite(sd_reg_addr(SD_REG_CTRL), SD_CTRL_START);

    if (sd_wait_idle(SD_WAIT_TIMEOUT) != 0) {
        SD_LOGE("SD timeout while waiting DMA done\n");
        return -2;
    }

    stat = RegRead(sd_reg_addr(SD_REG_STAT));
    if ((stat & SD_STAT_DMA_ERR_MASK) != 0u) {
        SD_LOGE("SD DMA error, STAT=0x%08X\n", stat);
        return -3;
    }

    SD_LOGI("SD DMA done, PROGRESS=0x%08X bytes\n", RegRead(sd_reg_addr(SD_REG_PROGRESS)));
    return 0;
}

unsigned int sd_dma_get_progress_bytes(void)
{
    return RegRead(sd_reg_addr(SD_REG_PROGRESS));
}

int read_params(unsigned int start_sector, int16_t *dst, unsigned int length)
{
    unsigned int bytes_needed;
    unsigned int sector_num;
    unsigned int dma_sector_num;
    unsigned int dma_elems;
    unsigned int i;
    int16_t* dma_dst;
    int16_t* tmp_buf = NULL;
    int ret;

    if (dst == NULL || length == 0u) {
        SD_LOGE("read_params invalid argument\n");
        return -4;
    }

    bytes_needed = length * (unsigned int)sizeof(int16_t);
    sector_num = (bytes_needed + 511u) / 512u;

    dma_sector_num = sector_num;
    if ((dma_sector_num & 0x1u) != 0u) {
        dma_sector_num++;
    }

    dma_elems = (dma_sector_num * 512u) / (unsigned int)sizeof(int16_t);

    if (dma_elems > length) {
        tmp_buf = (int16_t*)malloc((size_t)dma_elems * sizeof(int16_t));
        if (tmp_buf == NULL) {
            SD_LOGE("read_params scratch alloc failed\n");
            return -5;
        }
        dma_dst = tmp_buf;
    } else {
        dma_dst = dst;
    }

    ret = sd_dma_read_sectors((unsigned int)(uintptr_t)dma_dst, start_sector, dma_sector_num);
    if (ret != 0) {
        free(tmp_buf);
        return ret;
    }

    if (dma_dst != dst) {
        memcpy(dst, dma_dst, length * sizeof(int16_t));
    }

    free(tmp_buf);

    return 0;
}


