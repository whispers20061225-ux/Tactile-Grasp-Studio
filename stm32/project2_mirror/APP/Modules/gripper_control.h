#ifndef APP_MODULES_GRIPPER_CONTROL_H_
#define APP_MODULES_GRIPPER_CONTROL_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    GRIPPER_MODE_IDLE = 0,
    GRIPPER_MODE_POSITION = 1,
    GRIPPER_MODE_CONTACT_HOLD = 2,
    GRIPPER_MODE_FORCE_PI = 3,
    GRIPPER_MODE_HYBRID_HOLD = 4,
} gripper_mode_t;

typedef struct
{
    gripper_mode_t mode;
    uint8_t servo_id;
    uint8_t tactile_dev_addr;
    int8_t close_direction;
    uint8_t source_connected;
    uint8_t tare_valid;
    uint8_t contact_active;
    uint16_t open_limit_raw;
    uint16_t close_limit_raw;
    uint16_t commanded_pos_raw;
    uint16_t move_time_ms;
    uint16_t poll_period_ms;
    float target_force;
    float contact_threshold;
    float hard_force_limit;
    float measured_force_raw;
    float measured_force;
    float tare_force;
    float kp;
    float ki;
    float kd;
    float deadband;
    float max_step_per_tick;
    float integrator;
    float integrator_limit;
    float last_error;
    float last_derivative;
    uint32_t update_count;
    uint32_t fault_count;
    uint32_t last_update_ms;
    char status_text[48];
} gripper_control_status_t;

void gripper_control_init(void);
void gripper_control_poll(void);

int gripper_control_status(gripper_control_status_t *out);
int gripper_control_format_status(char *out, size_t out_cap);

int gripper_control_set_servo_id(uint8_t servo_id);
int gripper_control_set_tactile_dev_addr(uint8_t dev_addr);
int gripper_control_set_close_direction(int8_t direction);
int gripper_control_set_limits(uint16_t open_limit_raw, uint16_t close_limit_raw);
int gripper_control_set_pid(float kp, float ki, float kd);
int gripper_control_set_contact_threshold(float threshold);
int gripper_control_set_hard_force_limit(float limit);
int gripper_control_set_deadband(float deadband);
int gripper_control_set_max_step(float max_step_per_tick);
int gripper_control_set_poll_period(uint16_t poll_period_ms);
int gripper_control_set_move_time(uint16_t move_time_ms);
int gripper_control_tare(uint8_t sample_count);
int gripper_control_sync_command_to_current(void);
int gripper_control_move_raw(uint16_t pos_raw, uint16_t move_time_ms);
int gripper_control_start_contact_hold(void);
int gripper_control_start_force_pi(float target_force);
int gripper_control_start_hybrid_hold(float target_force);
int gripper_control_stop(void);

#ifdef __cplusplus
}
#endif

#endif /* APP_MODULES_GRIPPER_CONTROL_H_ */
