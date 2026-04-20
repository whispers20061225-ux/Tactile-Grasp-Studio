/*
 * tactile_api.c
 *
 *  Created on: Apr 8, 2026
 *      Author: whispers
 */

#include "tactile_api.h"

#include "paxini_uart_proto.h"
#include "tactile_bus.h"

static uint8_t s_tx_buf[32];

static int8_t to_s8(uint8_t value)
{
    return (value >= 128U) ? (int8_t)((int)value - 256) : (int8_t)value;
}

static int tactile_read_app_frame_with_timeout(uint8_t dev_addr,
                                               uint32_t start_addr,
                                               uint16_t read_len,
                                               uint8_t *out,
                                               uint16_t out_cap,
                                               uint16_t *out_len,
                                               uint32_t tx_timeout_ms,
                                               uint32_t rx_timeout_ms)
{
    size_t tx_len = 0;
    int rc = 0;

    if (out == NULL || out_len == NULL || read_len == 0U) {
        return -1;
    }

    tx_len = paxini_pack_read_app(dev_addr, start_addr, read_len, s_tx_buf, sizeof(s_tx_buf));
    if (tx_len == 0U) {
        return -2;
    }

    rc = tactile_bus_send(s_tx_buf, (uint16_t)tx_len, tx_timeout_ms);
    if (rc != 0) {
        return rc;
    }

    rc = tactile_bus_recv_frame(out, out_cap, rx_timeout_ms);
    if (rc < 0) {
        return rc;
    }

    *out_len = (uint16_t)rc;
    return 0;
}

void tactile_api_init(void)
{
}

int tactile_read_app_frame(uint8_t dev_addr,
                           uint32_t start_addr,
                           uint16_t read_len,
                           uint8_t *out,
                           uint16_t out_cap,
                           uint16_t *out_len)
{
    return tactile_read_app_frame_with_timeout(
        dev_addr,
        start_addr,
        read_len,
        out,
        out_cap,
        out_len,
        50U,
        100U
    );
}

int tactile_read_block_1038_32(uint8_t dev_addr,
                               uint8_t *out,
                               uint16_t out_cap,
                               uint16_t *out_len)
{
    return tactile_read_app_frame(dev_addr, 1038U, 32U, out, out_cap, out_len);
}

int tactile_read_m2020_totals(uint8_t dev_addr,
                              int8_t *fx_out,
                              int8_t *fy_out,
                              uint8_t *fz_out,
                              uint32_t tx_timeout_ms,
                              uint32_t rx_timeout_ms)
{
    uint8_t frame[32];
    uint16_t frame_len = 0U;
    uint16_t payload_len = 0U;
    uint32_t start_addr = 0U;
    int rc = tactile_read_app_frame_with_timeout(
        dev_addr,
        1008U,
        3U,
        frame,
        sizeof(frame),
        &frame_len,
        tx_timeout_ms,
        rx_timeout_ms
    );

    if (rc != 0) {
        return rc;
    }
    if (frame_len < 18U) {
        return -10;
    }
    if (frame[0] != PAXINI_UART_RSP_H1 || frame[1] != PAXINI_UART_RSP_H2) {
        return -11;
    }

    start_addr = (uint32_t)frame[7] |
                 ((uint32_t)frame[8] << 8) |
                 ((uint32_t)frame[9] << 16) |
                 ((uint32_t)frame[10] << 24);
    if (start_addr != 1008U) {
        return -12;
    }

    payload_len = (uint16_t)(frame_len - 15U);
    if (payload_len < 3U) {
        return -13;
    }

    if (fx_out != NULL) {
        *fx_out = to_s8(frame[14]);
    }
    if (fy_out != NULL) {
        *fy_out = to_s8(frame[15]);
    }
    if (fz_out != NULL) {
        *fz_out = frame[16];
    }

    return 0;
}
