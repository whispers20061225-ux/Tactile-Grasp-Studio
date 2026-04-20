#pragma once
#include <stdint.h>

/*
 * 这里提供“业务友好”的接口：
 * - servo_ping：检查是否在线
 * - servo_read_pos：读当前位置（验证读通）
 * - servo_move：写目标位置+时间（验证能动）
 */

int servo_ping(uint8_t id);
int servo_read_pos(uint8_t id, uint16_t* pos_out);
int servo_move(uint8_t id, uint16_t pos, uint16_t ms);
int servo_read_reg(uint8_t id, uint8_t addr, uint8_t* data_out, uint8_t nbytes);
int servo_write_reg8(uint8_t id, uint8_t addr, uint8_t value);
int servo_write_reg16(uint8_t id, uint8_t addr, uint16_t value);
int servo_set_id(uint8_t current_id, uint8_t new_id);
