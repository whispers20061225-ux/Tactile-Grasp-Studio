/*
 * host_link.c
 *
 * Text command bridge between USB CDC and the STM32 application modules.
 */

#include "host_link.h"

#include "build_info.h"
#include "gripper_control.h"
#include "paxini_uart_proto.h"
#include "servo_api.h"
#include "tactile_api.h"
#include "tactile_bus.h"
#include "usbd_cdc_if.h"
#include "stm32f1xx_hal.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE_BUF_SIZE 256
#define TX_BUF_SIZE   256
#define TACTILE_MAX_READ_LEN 64U

static char s_line_buf[LINE_BUF_SIZE];
static size_t s_line_len = 0U;
static volatile uint8_t s_line_ready = 0U;

static uint8_t s_tx_buf[TX_BUF_SIZE];
static uint16_t s_tx_len = 0U;
static volatile uint8_t s_tx_pending = 0U;

static int32_t round_to_i32_local(float value)
{
    return (value >= 0.0f) ? (int32_t)(value + 0.5f) : (int32_t)(value - 0.5f);
}

static void format_fixed_local(char *out, size_t out_cap, float value, uint8_t decimals)
{
    uint8_t digits = decimals;
    int32_t scale = 1;
    int32_t scaled = 0;
    uint8_t negative = 0U;
    int32_t abs_scaled = 0;
    int32_t whole = 0;
    int32_t fraction = 0;

    if (out == NULL || out_cap == 0U) {
        return;
    }

    while (digits > 0U) {
        scale *= 10;
        digits--;
    }

    scaled = round_to_i32_local(value * (float)scale);
    if (scaled < 0) {
        negative = 1U;
        abs_scaled = -scaled;
    } else {
        abs_scaled = scaled;
    }

    whole = abs_scaled / scale;
    fraction = abs_scaled % scale;

    if (decimals == 0U) {
        snprintf(out, out_cap, "%s%ld", negative ? "-" : "", (long)whole);
        return;
    }

    snprintf(
        out,
        out_cap,
        "%s%ld.%0*ld",
        negative ? "-" : "",
        (long)whole,
        (int)decimals,
        (long)fraction
    );
}

static void trim_inplace(char *s)
{
    char *start = s;
    size_t len = 0U;

    if (s == NULL) {
        return;
    }

    while (*start != '\0' && isspace((unsigned char)*start)) {
        start++;
    }

    if (start != s) {
        memmove(s, start, strlen(start) + 1U);
    }

    len = strlen(s);
    while (len > 0U && isspace((unsigned char)s[len - 1U])) {
        s[len - 1U] = '\0';
        len--;
    }
}

static uint8_t tx_queue_bytes(const uint8_t *data, size_t len)
{
    if (data == NULL || len == 0U || s_tx_pending) {
        return 0U;
    }

    if (len >= TX_BUF_SIZE) {
        len = TX_BUF_SIZE - 1U;
    }

    memcpy(s_tx_buf, data, len);
    s_tx_len = (uint16_t)len;
    s_tx_pending = 1U;
    return 1U;
}

static uint8_t tx_send_line(const char *s)
{
    if (s == NULL) {
        return 0U;
    }

    return tx_queue_bytes((const uint8_t *)s, strlen(s));
}

static void try_flush_tx(void)
{
    if (!s_tx_pending) {
        return;
    }

    if (CDC_Transmit_FS(s_tx_buf, s_tx_len) == USBD_OK) {
        s_tx_pending = 0U;
        s_tx_len = 0U;
    }
}

static int parse_u32_token(const char **cursor, uint32_t *out_value)
{
    char *end = NULL;
    unsigned long value = 0UL;

    if (cursor == NULL || *cursor == NULL || out_value == NULL) {
        return -1;
    }

    while (**cursor != '\0' && isspace((unsigned char)**cursor)) {
        (*cursor)++;
    }

    if (**cursor == '\0') {
        return -1;
    }

    value = strtoul(*cursor, &end, 0);
    if (end == *cursor) {
        return -1;
    }

    *out_value = (uint32_t)value;
    *cursor = end;
    return 0;
}

static int parse_i32_token(const char **cursor, int32_t *out_value)
{
    char *end = NULL;
    long value = 0L;

    if (cursor == NULL || *cursor == NULL || out_value == NULL) {
        return -1;
    }

    while (**cursor != '\0' && isspace((unsigned char)**cursor)) {
        (*cursor)++;
    }

    if (**cursor == '\0') {
        return -1;
    }

    value = strtol(*cursor, &end, 0);
    if (end == *cursor) {
        return -1;
    }

    *out_value = (int32_t)value;
    *cursor = end;
    return 0;
}

static int parse_f32_token(const char **cursor, float *out_value)
{
    char *end = NULL;
    double value = 0.0;

    if (cursor == NULL || *cursor == NULL || out_value == NULL) {
        return -1;
    }

    while (**cursor != '\0' && isspace((unsigned char)**cursor)) {
        (*cursor)++;
    }

    if (**cursor == '\0') {
        return -1;
    }

    value = strtod(*cursor, &end);
    if (end == *cursor) {
        return -1;
    }

    *out_value = (float)value;
    *cursor = end;
    return 0;
}

static int cursor_expect_end(const char *cursor)
{
    if (cursor == NULL) {
        return -1;
    }

    while (*cursor != '\0' && isspace((unsigned char)*cursor)) {
        cursor++;
    }

    return (*cursor == '\0') ? 0 : -1;
}

static int parse_gripper_u8_arg(const char *args, uint8_t *value_out)
{
    uint32_t parsed = 0U;
    const char *cursor = args;

    if (parse_u32_token(&cursor, &parsed) != 0 || cursor_expect_end(cursor) != 0 || parsed > 255U) {
        return -1;
    }

    *value_out = (uint8_t)parsed;
    return 0;
}

static int parse_gripper_i8_arg(const char *args, int8_t *value_out)
{
    int32_t parsed = 0;
    const char *cursor = args;

    if (parse_i32_token(&cursor, &parsed) != 0 || cursor_expect_end(cursor) != 0) {
        return -1;
    }
    if (parsed < -128 || parsed > 127) {
        return -1;
    }

    *value_out = (int8_t)parsed;
    return 0;
}

static int parse_gripper_u16_arg(const char *args, uint16_t *value_out)
{
    uint32_t parsed = 0U;
    const char *cursor = args;

    if (parse_u32_token(&cursor, &parsed) != 0 || cursor_expect_end(cursor) != 0 || parsed > 0xFFFFU) {
        return -1;
    }

    *value_out = (uint16_t)parsed;
    return 0;
}

static int parse_gripper_float_arg(const char *args, float *value_out)
{
    const char *cursor = args;
    if (parse_f32_token(&cursor, value_out) != 0 || cursor_expect_end(cursor) != 0) {
        return -1;
    }
    return 0;
}

static int parse_gripper_u16_pair_args(const char *args, uint16_t *first_out, uint16_t *second_out)
{
    uint32_t first = 0U;
    uint32_t second = 0U;
    const char *cursor = args;

    if (parse_u32_token(&cursor, &first) != 0 ||
        parse_u32_token(&cursor, &second) != 0 ||
        cursor_expect_end(cursor) != 0 ||
        first > 0xFFFFU ||
        second > 0xFFFFU) {
        return -1;
    }

    *first_out = (uint16_t)first;
    *second_out = (uint16_t)second;
    return 0;
}

static int parse_gripper_float_triplet_args(const char *args,
                                            float *first_out,
                                            float *second_out,
                                            float *third_out)
{
    const char *cursor = args;

    if (parse_f32_token(&cursor, first_out) != 0 ||
        parse_f32_token(&cursor, second_out) != 0 ||
        parse_f32_token(&cursor, third_out) != 0 ||
        cursor_expect_end(cursor) != 0) {
        return -1;
    }

    return 0;
}

static int parse_gripper_move_args(const char *args, uint16_t *pos_out, uint16_t *move_ms_out)
{
    uint32_t pos = 0U;
    uint32_t move_ms = 0U;
    const char *cursor = args;

    if (parse_u32_token(&cursor, &pos) != 0) {
        return -1;
    }

    while (*cursor != '\0' && isspace((unsigned char)*cursor)) {
        cursor++;
    }

    if (*cursor == '\0') {
        *pos_out = (uint16_t)pos;
        *move_ms_out = 0U;
        return (pos <= 0xFFFFU) ? 0 : -1;
    }

    if (parse_u32_token(&cursor, &move_ms) != 0 ||
        cursor_expect_end(cursor) != 0 ||
        pos > 0xFFFFU ||
        move_ms > 0xFFFFU) {
        return -1;
    }

    *pos_out = (uint16_t)pos;
    *move_ms_out = (uint16_t)move_ms;
    return 0;
}

static int parse_tactile_read_args(const char *args,
                                   uint8_t *dev_addr,
                                   uint32_t *start_addr,
                                   uint16_t *read_len)
{
    uint32_t dev = 0U;
    uint32_t addr = 0U;
    uint32_t len = 0U;
    const char *cursor = args;

    if (parse_u32_token(&cursor, &dev) != 0 ||
        parse_u32_token(&cursor, &addr) != 0 ||
        parse_u32_token(&cursor, &len) != 0) {
        return -1;
    }

    while (*cursor != '\0' && isspace((unsigned char)*cursor)) {
        cursor++;
    }

    if (*cursor != '\0') {
        return -1;
    }

    if (dev > 255U || len == 0U || len > TACTILE_MAX_READ_LEN) {
        return -2;
    }

    *dev_addr = (uint8_t)dev;
    *start_addr = addr;
    *read_len = (uint16_t)len;
    return 0;
}

static int tactile_send_request(uint8_t dev_addr,
                                uint32_t start_addr,
                                uint16_t read_len,
                                uint8_t *tx_buf,
                                size_t tx_cap)
{
    size_t tx_len = paxini_pack_read_app(dev_addr, start_addr, read_len, tx_buf, tx_cap);
    if (tx_len == 0U) {
        return -2;
    }

    return tactile_bus_send(tx_buf, (uint16_t)tx_len, 50U);
}

static void send_hex_payload_line(const char *prefix, const uint8_t *data, int data_len)
{
    char out[TX_BUF_SIZE];
    int pos = snprintf(out, sizeof(out), "%s n=%d ", prefix, data_len);

    if (pos < 0) {
        return;
    }

    for (int i = 0; i < data_len && pos < (int)sizeof(out) - 4; i++) {
        pos += snprintf(out + pos, sizeof(out) - (size_t)pos, "%02X ", data[i]);
    }

    if (pos < (int)sizeof(out) - 2) {
        out[pos++] = '\n';
        out[pos] = '\0';
    } else {
        out[sizeof(out) - 2] = '\n';
        out[sizeof(out) - 1] = '\0';
    }

    (void)tx_send_line(out);
}

static void send_servo_reg_line(uint8_t id, uint8_t addr, const uint8_t *data, uint8_t data_len)
{
    char out[TX_BUF_SIZE];
    int pos = snprintf(
        out,
        sizeof(out),
        "REG %u 0x%02X len=%u ",
        (unsigned)id,
        (unsigned)addr,
        (unsigned)data_len
    );

    if (pos < 0) {
        return;
    }

    for (uint8_t i = 0U; i < data_len && pos < (int)sizeof(out) - 4; i++) {
        pos += snprintf(out + pos, sizeof(out) - (size_t)pos, "%02X ", data[i]);
    }

    if (pos < (int)sizeof(out) - 2) {
        out[pos++] = '\n';
        out[pos] = '\0';
    } else {
        out[sizeof(out) - 2] = '\n';
        out[sizeof(out) - 1] = '\0';
    }

    (void)tx_send_line(out);
}

static int parse_servo_reg_read_args(const char *args,
                                     uint8_t *servo_id,
                                     uint8_t *addr,
                                     uint8_t *read_len)
{
    uint32_t parsed_id = 0U;
    uint32_t parsed_addr = 0U;
    uint32_t parsed_len = 0U;
    const char *cursor = args;

    if (parse_u32_token(&cursor, &parsed_id) != 0 ||
        parse_u32_token(&cursor, &parsed_addr) != 0 ||
        parse_u32_token(&cursor, &parsed_len) != 0) {
        return -1;
    }

    while (*cursor != '\0' && isspace((unsigned char)*cursor)) {
        cursor++;
    }

    if (*cursor != '\0') {
        return -1;
    }

    if (parsed_id == 0U || parsed_id > 250U ||
        parsed_addr > 255U ||
        parsed_len == 0U || parsed_len > 32U) {
        return -2;
    }

    *servo_id = (uint8_t)parsed_id;
    *addr = (uint8_t)parsed_addr;
    *read_len = (uint8_t)parsed_len;
    return 0;
}

static int parse_servo_reg_write8_args(const char *args,
                                       uint8_t *servo_id,
                                       uint8_t *addr,
                                       uint8_t *value)
{
    uint32_t parsed_id = 0U;
    uint32_t parsed_addr = 0U;
    uint32_t parsed_value = 0U;
    const char *cursor = args;

    if (parse_u32_token(&cursor, &parsed_id) != 0 ||
        parse_u32_token(&cursor, &parsed_addr) != 0 ||
        parse_u32_token(&cursor, &parsed_value) != 0) {
        return -1;
    }

    while (*cursor != '\0' && isspace((unsigned char)*cursor)) {
        cursor++;
    }

    if (*cursor != '\0') {
        return -1;
    }

    if ((parsed_id == 0U || parsed_id > 250U) && parsed_id != 254U) {
        return -2;
    }
    if (parsed_addr > 255U || parsed_value > 255U) {
        return -2;
    }

    *servo_id = (uint8_t)parsed_id;
    *addr = (uint8_t)parsed_addr;
    *value = (uint8_t)parsed_value;
    return 0;
}

static int parse_servo_reg_write16_args(const char *args,
                                        uint8_t *servo_id,
                                        uint8_t *addr,
                                        uint16_t *value)
{
    uint32_t parsed_id = 0U;
    uint32_t parsed_addr = 0U;
    uint32_t parsed_value = 0U;
    const char *cursor = args;

    if (parse_u32_token(&cursor, &parsed_id) != 0 ||
        parse_u32_token(&cursor, &parsed_addr) != 0 ||
        parse_u32_token(&cursor, &parsed_value) != 0) {
        return -1;
    }

    while (*cursor != '\0' && isspace((unsigned char)*cursor)) {
        cursor++;
    }

    if (*cursor != '\0') {
        return -1;
    }

    if ((parsed_id == 0U || parsed_id > 250U) && parsed_id != 254U) {
        return -2;
    }
    if (parsed_addr > 255U || parsed_value > 0xFFFFU) {
        return -2;
    }

    *servo_id = (uint8_t)parsed_id;
    *addr = (uint8_t)parsed_addr;
    *value = (uint16_t)parsed_value;
    return 0;
}

static int parse_servo_set_id_args(const char *args,
                                   uint8_t *current_id,
                                   uint8_t *new_id)
{
    uint32_t parsed_current_id = 0U;
    uint32_t parsed_new_id = 0U;
    const char *cursor = args;

    if (parse_u32_token(&cursor, &parsed_current_id) != 0 ||
        parse_u32_token(&cursor, &parsed_new_id) != 0) {
        return -1;
    }

    while (*cursor != '\0' && isspace((unsigned char)*cursor)) {
        cursor++;
    }

    if (*cursor != '\0') {
        return -1;
    }

    if (parsed_current_id == 0U || parsed_current_id > 250U) {
        return -2;
    }
    if (parsed_new_id == 0U || parsed_new_id > 250U || parsed_new_id == 254U) {
        return -2;
    }

    *current_id = (uint8_t)parsed_current_id;
    *new_id = (uint8_t)parsed_new_id;
    return 0;
}

static void handle_tactile_raw_read(const char *prefix,
                                    uint8_t dev_addr,
                                    uint32_t start_addr,
                                    uint16_t read_len)
{
    uint8_t tx[32];
    uint8_t raw[128];
    char out[TX_BUF_SIZE];
    int rc = tactile_send_request(dev_addr, start_addr, read_len, tx, sizeof(tx));

    if (rc != 0) {
        snprintf(out, sizeof(out), "%s FAIL send %d\n", prefix, rc);
        (void)tx_send_line(out);
        return;
    }

    rc = tactile_bus_recv_raw(raw, sizeof(raw), 100U);
    if (rc < 0) {
        snprintf(out, sizeof(out), "%s FAIL raw %d\n", prefix, rc);
        (void)tx_send_line(out);
        return;
    }

    send_hex_payload_line(prefix, raw, rc);
}

static void handle_tactile_frame_read(const char *prefix,
                                      uint8_t dev_addr,
                                      uint32_t start_addr,
                                      uint16_t read_len)
{
    uint8_t frame[128];
    uint16_t frame_len = 0U;
    char out[TX_BUF_SIZE];
    int rc = tactile_read_app_frame(dev_addr, start_addr, read_len, frame, sizeof(frame), &frame_len);

    if (rc != 0) {
        snprintf(out, sizeof(out), "%s FAIL %d\n", prefix, rc);
        (void)tx_send_line(out);
        return;
    }

    {
        int pos = snprintf(out, sizeof(out), "%s OK len=%u ", prefix, (unsigned)frame_len);
        for (uint16_t i = 0; i < frame_len && pos < (int)sizeof(out) - 4; i++) {
            pos += snprintf(out + pos, sizeof(out) - (size_t)pos, "%02X ", frame[i]);
        }

        if (pos < (int)sizeof(out) - 2) {
            out[pos++] = '\n';
            out[pos] = '\0';
        } else {
            out[sizeof(out) - 2] = '\n';
            out[sizeof(out) - 1] = '\0';
        }
    }

    (void)tx_send_line(out);
}

void host_link_init(void)
{
    s_line_len = 0U;
    s_line_buf[0] = '\0';
    s_line_ready = 0U;
    s_tx_len = 0U;
    s_tx_pending = 0U;
}

void host_link_send(const uint8_t *data, size_t len)
{
    (void)tx_queue_bytes(data, len);
}

void host_link_on_rx(const uint8_t *data, size_t len)
{
    if (data == NULL || len == 0U) {
        return;
    }

    for (size_t i = 0U; i < len; i++) {
        char c = (char)data[i];

        if (c == '\r') {
            continue;
        }

        if (c == '\n') {
            s_line_buf[s_line_len] = '\0';
            s_line_ready = 1U;
            s_line_len = 0U;
            return;
        }

        if (s_line_len < (LINE_BUF_SIZE - 1U)) {
            s_line_buf[s_line_len++] = c;
        } else {
            s_line_len = 0U;
        }
    }
}

static void handle_line(char *s)
{
    char out[TX_BUF_SIZE];

    if (s == NULL) {
        return;
    }

    trim_inplace(s);
    if (s[0] == '\0') {
        return;
    }

    if (strcmp(s, "PING") == 0) {
        (void)tx_send_line("PONG\n");
        return;
    }

    if (strcmp(s, "VER") == 0) {
        snprintf(out, sizeof(out), "VER %s %s %s %s\n", FW_NAME, FW_VERSION, __DATE__, __TIME__);
        (void)tx_send_line(out);
        return;
    }

    if (strncmp(s, "ECHO ", 5) == 0) {
        snprintf(out, sizeof(out), "ECHO %.*s\n", (int)(sizeof(out) - 7U), s + 5);
        (void)tx_send_line(out);
        return;
    }

    if (strcmp(s, "TSTAT RESET") == 0) {
        tactile_bus_reset_stats();
        (void)tx_send_line("TSTAT RESET OK\n");
        return;
    }

    if (strcmp(s, "TSTAT") == 0) {
        tactile_bus_stats_t st;
        tactile_bus_get_stats(&st);
        snprintf(out,
                 sizeof(out),
                 "TSTAT bytes=%lu ovf=%lu pe=%lu ne=%lu fe=%lu ore=%lu rearm=%lu\n",
                 (unsigned long)st.rx_bytes,
                 (unsigned long)st.rx_overflow,
                 (unsigned long)st.rx_pe,
                 (unsigned long)st.rx_ne,
                 (unsigned long)st.rx_fe,
                 (unsigned long)st.rx_ore,
                 (unsigned long)st.rx_rearm_fail);
        (void)tx_send_line(out);
        return;
    }

    if (strncmp(s, "TRAWX ", 6) == 0) {
        uint8_t dev_addr = 0U;
        uint32_t start_addr = 0U;
        uint16_t read_len = 0U;
        int rc = parse_tactile_read_args(s + 6, &dev_addr, &start_addr, &read_len);
        if (rc != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }
        handle_tactile_raw_read("TRAWX", dev_addr, start_addr, read_len);
        return;
    }

    if (strncmp(s, "TREADX ", 7) == 0) {
        uint8_t dev_addr = 0U;
        uint32_t start_addr = 0U;
        uint16_t read_len = 0U;
        int rc = parse_tactile_read_args(s + 7, &dev_addr, &start_addr, &read_len);
        if (rc != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }
        handle_tactile_frame_read("TREADX", dev_addr, start_addr, read_len);
        return;
    }

    if (strncmp(s, "TRAW ", 5) == 0) {
        int dev = atoi(s + 5);
        handle_tactile_raw_read("TRAW", (uint8_t)dev, 1038U, 32U);
        return;
    }

    if (strncmp(s, "TREAD ", 6) == 0) {
        int dev = atoi(s + 6);
        handle_tactile_frame_read("TREAD", (uint8_t)dev, 1038U, 32U);
        return;
    }

    if (strncmp(s, "SPING ", 6) == 0) {
        int id = atoi(s + 6);
        int rc = servo_ping((uint8_t)id);
        (void)tx_send_line((rc == 0) ? "OK\n" : "FAIL\n");
        return;
    }

    if (strncmp(s, "SREADPOS ", 9) == 0) {
        int id = atoi(s + 9);
        uint16_t pos = 0U;
        int rc = servo_read_pos((uint8_t)id, &pos);

        if (rc == 0) {
            snprintf(out, sizeof(out), "POS %d %u\n", id, (unsigned)pos);
        } else {
            snprintf(out, sizeof(out), "FAIL %d\n", id);
        }

        (void)tx_send_line(out);
        return;
    }

    if (strncmp(s, "SMOVE ", 6) == 0) {
        int id = 0;
        unsigned pos = 0U;
        unsigned ms = 0U;

        if (sscanf(s + 6, "%d %u %u", &id, &pos, &ms) == 3) {
            int rc = servo_move((uint8_t)id, (uint16_t)pos, (uint16_t)ms);
            (void)tx_send_line((rc == 0) ? "OK\n" : "FAIL\n");
        } else {
            (void)tx_send_line("ERR bad_args\n");
        }
        return;
    }

    if (strncmp(s, "SREADREG ", 9) == 0) {
        uint8_t id = 0U;
        uint8_t addr = 0U;
        uint8_t read_len = 0U;
        uint8_t data[32];

        if (parse_servo_reg_read_args(s + 9, &id, &addr, &read_len) != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }

        if (servo_read_reg(id, addr, data, read_len) == 0) {
            send_servo_reg_line(id, addr, data, read_len);
        } else {
            snprintf(out, sizeof(out), "FAIL %u 0x%02X\n", (unsigned)id, (unsigned)addr);
            (void)tx_send_line(out);
        }
        return;
    }

    if (strncmp(s, "SWRITE8 ", 8) == 0) {
        uint8_t id = 0U;
        uint8_t addr = 0U;
        uint8_t value = 0U;

        if (parse_servo_reg_write8_args(s + 8, &id, &addr, &value) != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }

        if (servo_write_reg8(id, addr, value) == 0) {
            snprintf(
                out,
                sizeof(out),
                "OK %u 0x%02X 0x%02X\n",
                (unsigned)id,
                (unsigned)addr,
                (unsigned)value
            );
        } else {
            snprintf(out, sizeof(out), "FAIL %u 0x%02X\n", (unsigned)id, (unsigned)addr);
        }
        (void)tx_send_line(out);
        return;
    }

    if (strncmp(s, "SWRITE16 ", 9) == 0) {
        uint8_t id = 0U;
        uint8_t addr = 0U;
        uint16_t value = 0U;

        if (parse_servo_reg_write16_args(s + 9, &id, &addr, &value) != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }

        if (servo_write_reg16(id, addr, value) == 0) {
            snprintf(
                out,
                sizeof(out),
                "OK %u 0x%02X 0x%04X\n",
                (unsigned)id,
                (unsigned)addr,
                (unsigned)value
            );
        } else {
            snprintf(out, sizeof(out), "FAIL %u 0x%02X\n", (unsigned)id, (unsigned)addr);
        }
        (void)tx_send_line(out);
        return;
    }

    if (strncmp(s, "SSETID ", 7) == 0) {
        uint8_t current_id = 0U;
        uint8_t new_id = 0U;

        if (parse_servo_set_id_args(s + 7, &current_id, &new_id) != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }

        if (servo_set_id(current_id, new_id) != 0) {
            snprintf(
                out,
                sizeof(out),
                "FAIL %u -> %u\n",
                (unsigned)current_id,
                (unsigned)new_id
            );
            (void)tx_send_line(out);
            return;
        }

        HAL_Delay(40U);
        if (servo_ping(new_id) == 0) {
            snprintf(
                out,
                sizeof(out),
                "ID %u -> %u OK\n",
                (unsigned)current_id,
                (unsigned)new_id
            );
        } else {
            snprintf(
                out,
                sizeof(out),
                "ID %u -> %u VERIFY_FAIL\n",
                (unsigned)current_id,
                (unsigned)new_id
            );
        }
        (void)tx_send_line(out);
        return;
    }

    if (strcmp(s, "GSTAT") == 0) {
        if (gripper_control_format_status(out, sizeof(out) - 2U) != 0) {
            (void)tx_send_line("FAIL gripper_status\n");
            return;
        }
        size_t out_len = strlen(out);
        if (out_len < sizeof(out) - 1U) {
            out[out_len++] = '\n';
            out[out_len] = '\0';
        }
        (void)tx_send_line(out);
        return;
    }

    if (strcmp(s, "GSTOP") == 0) {
        if (gripper_control_stop() == 0) {
            (void)tx_send_line("OK gripper_idle\n");
        } else {
            (void)tx_send_line("FAIL gripper_idle\n");
        }
        return;
    }

    if (strncmp(s, "GTARE", 5) == 0) {
        uint8_t samples = 5U;
        if (s[5] != '\0') {
            if (!isspace((unsigned char)s[5]) || parse_gripper_u8_arg(s + 6, &samples) != 0) {
                (void)tx_send_line("ERR bad_args\n");
                return;
            }
        }
        if (gripper_control_tare(samples) == 0) {
            (void)tx_send_line("OK tare\n");
        } else {
            (void)tx_send_line("FAIL tare\n");
        }
        return;
    }

    if (strncmp(s, "GID ", 4) == 0) {
        uint8_t servo_id = 0U;
        if (parse_gripper_u8_arg(s + 4, &servo_id) != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }
        if (gripper_control_set_servo_id(servo_id) == 0) {
            snprintf(out, sizeof(out), "OK servo=%u\n", (unsigned)servo_id);
        } else {
            snprintf(out, sizeof(out), "FAIL servo=%u\n", (unsigned)servo_id);
        }
        (void)tx_send_line(out);
        return;
    }

    if (strncmp(s, "GTACT ", 6) == 0) {
        uint8_t dev_addr = 0U;
        if (parse_gripper_u8_arg(s + 6, &dev_addr) != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }
        if (gripper_control_set_tactile_dev_addr(dev_addr) == 0) {
            snprintf(out, sizeof(out), "OK tactile=%u\n", (unsigned)dev_addr);
        } else {
            snprintf(out, sizeof(out), "FAIL tactile=%u\n", (unsigned)dev_addr);
        }
        (void)tx_send_line(out);
        return;
    }

    if (strncmp(s, "GDIR ", 5) == 0) {
        int8_t direction = 0;
        if (parse_gripper_i8_arg(s + 5, &direction) != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }
        if (gripper_control_set_close_direction(direction) == 0) {
            snprintf(out, sizeof(out), "OK dir=%d\n", (int)direction);
        } else {
            snprintf(out, sizeof(out), "FAIL dir=%d\n", (int)direction);
        }
        (void)tx_send_line(out);
        return;
    }

    if (strncmp(s, "GLIM ", 5) == 0) {
        uint16_t open_limit = 0U;
        uint16_t close_limit = 0U;
        if (parse_gripper_u16_pair_args(s + 5, &open_limit, &close_limit) != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }
        if (gripper_control_set_limits(open_limit, close_limit) == 0) {
            snprintf(out, sizeof(out), "OK lim=%u:%u\n", (unsigned)open_limit, (unsigned)close_limit);
        } else {
            snprintf(out, sizeof(out), "FAIL lim=%u:%u\n", (unsigned)open_limit, (unsigned)close_limit);
        }
        (void)tx_send_line(out);
        return;
    }

    if (strncmp(s, "GPID ", 5) == 0) {
        char kp_buf[24];
        char ki_buf[24];
        char kd_buf[24];
        float kp = 0.0f;
        float ki = 0.0f;
        float kd = 0.0f;
        if (parse_gripper_float_triplet_args(s + 5, &kp, &ki, &kd) != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }
        if (gripper_control_set_pid(kp, ki, kd) != 0) {
            (void)tx_send_line("FAIL pid\n");
            return;
        }
        format_fixed_local(kp_buf, sizeof(kp_buf), kp, 3U);
        format_fixed_local(ki_buf, sizeof(ki_buf), ki, 3U);
        format_fixed_local(kd_buf, sizeof(kd_buf), kd, 3U);
        snprintf(out, sizeof(out), "OK pid=%s,%s,%s\n", kp_buf, ki_buf, kd_buf);
        (void)tx_send_line(out);
        return;
    }

    if (strncmp(s, "GCONTACT ", 9) == 0) {
        char threshold_buf[24];
        float threshold = 0.0f;
        if (parse_gripper_float_arg(s + 9, &threshold) != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }
        if (gripper_control_set_contact_threshold(threshold) != 0) {
            (void)tx_send_line("FAIL contact\n");
            return;
        }
        format_fixed_local(threshold_buf, sizeof(threshold_buf), threshold, 2U);
        snprintf(out, sizeof(out), "OK contact=%s\n", threshold_buf);
        (void)tx_send_line(out);
        return;
    }

    if (strncmp(s, "GSAFE ", 6) == 0) {
        char limit_buf[24];
        float limit = 0.0f;
        if (parse_gripper_float_arg(s + 6, &limit) != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }
        if (gripper_control_set_hard_force_limit(limit) != 0) {
            (void)tx_send_line("FAIL hard_limit\n");
            return;
        }
        format_fixed_local(limit_buf, sizeof(limit_buf), limit, 2U);
        snprintf(out, sizeof(out), "OK hard_limit=%s\n", limit_buf);
        (void)tx_send_line(out);
        return;
    }

    if (strncmp(s, "GDEADBAND ", 10) == 0) {
        char deadband_buf[24];
        float deadband = 0.0f;
        if (parse_gripper_float_arg(s + 10, &deadband) != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }
        if (gripper_control_set_deadband(deadband) != 0) {
            (void)tx_send_line("FAIL deadband\n");
            return;
        }
        format_fixed_local(deadband_buf, sizeof(deadband_buf), deadband, 2U);
        snprintf(out, sizeof(out), "OK deadband=%s\n", deadband_buf);
        (void)tx_send_line(out);
        return;
    }

    if (strncmp(s, "GSTEP ", 6) == 0) {
        char step_buf[24];
        float max_step = 0.0f;
        if (parse_gripper_float_arg(s + 6, &max_step) != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }
        if (gripper_control_set_max_step(max_step) != 0) {
            (void)tx_send_line("FAIL step\n");
            return;
        }
        format_fixed_local(step_buf, sizeof(step_buf), max_step, 2U);
        snprintf(out, sizeof(out), "OK step=%s\n", step_buf);
        (void)tx_send_line(out);
        return;
    }

    if (strncmp(s, "GPOLL ", 6) == 0) {
        uint16_t poll_ms = 0U;
        if (parse_gripper_u16_arg(s + 6, &poll_ms) != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }
        if (gripper_control_set_poll_period(poll_ms) == 0) {
            snprintf(out, sizeof(out), "OK poll=%u\n", (unsigned)poll_ms);
        } else {
            (void)tx_send_line("FAIL poll\n");
            return;
        }
        (void)tx_send_line(out);
        return;
    }

    if (strncmp(s, "GMOVE ", 6) == 0) {
        uint16_t move_ms = 0U;
        if (parse_gripper_u16_arg(s + 6, &move_ms) != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }
        if (gripper_control_set_move_time(move_ms) == 0) {
            snprintf(out, sizeof(out), "OK move_ms=%u\n", (unsigned)move_ms);
        } else {
            (void)tx_send_line("FAIL move_ms\n");
            return;
        }
        (void)tx_send_line(out);
        return;
    }

    if (strcmp(s, "GSYNC") == 0) {
        if (gripper_control_sync_command_to_current() == 0) {
            (void)tx_send_line("OK sync\n");
        } else {
            (void)tx_send_line("FAIL sync\n");
        }
        return;
    }

    if (strncmp(s, "GSETPOS ", 8) == 0) {
        uint16_t pos_raw = 0U;
        uint16_t move_ms = 0U;
        if (parse_gripper_move_args(s + 8, &pos_raw, &move_ms) != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }
        if (gripper_control_move_raw(pos_raw, move_ms) == 0) {
            snprintf(out, sizeof(out), "OK pos=%u\n", (unsigned)pos_raw);
        } else {
            snprintf(out, sizeof(out), "FAIL pos=%u\n", (unsigned)pos_raw);
        }
        (void)tx_send_line(out);
        return;
    }

    if (strcmp(s, "GHOLD") == 0) {
        if (gripper_control_start_contact_hold() == 0) {
            (void)tx_send_line("OK contact_hold\n");
        } else {
            (void)tx_send_line("FAIL contact_hold\n");
        }
        return;
    }

    if (strncmp(s, "GHYBRID ", 8) == 0) {
        char target_buf[24];
        float target_force = 0.0f;
        if (parse_gripper_float_arg(s + 8, &target_force) != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }
        if (gripper_control_start_hybrid_hold(target_force) == 0) {
            format_fixed_local(target_buf, sizeof(target_buf), target_force, 2U);
            snprintf(out, sizeof(out), "OK hybrid=%s\n", target_buf);
        } else {
            format_fixed_local(target_buf, sizeof(target_buf), target_force, 2U);
            snprintf(out, sizeof(out), "FAIL hybrid=%s\n", target_buf);
        }
        (void)tx_send_line(out);
        return;
    }

    if (strncmp(s, "GFORCE ", 7) == 0) {
        char force_buf[24];
        float target_force = 0.0f;
        if (parse_gripper_float_arg(s + 7, &target_force) != 0) {
            (void)tx_send_line("ERR bad_args\n");
            return;
        }
        if (gripper_control_start_force_pi(target_force) == 0) {
            format_fixed_local(force_buf, sizeof(force_buf), target_force, 2U);
            snprintf(out, sizeof(out), "OK force=%s\n", force_buf);
        } else {
            format_fixed_local(force_buf, sizeof(force_buf), target_force, 2U);
            snprintf(out, sizeof(out), "FAIL force=%s\n", force_buf);
        }
        (void)tx_send_line(out);
        return;
    }

    (void)tx_send_line("ERR unknown_cmd\n");
}

void host_link_poll(void)
{
    try_flush_tx();

    if (!s_line_ready) {
        return;
    }

    s_line_ready = 0U;
    handle_line(s_line_buf);
}
