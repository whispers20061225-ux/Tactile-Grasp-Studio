#include "gripper_control.h"

#include "servo_api.h"
#include "tactile_api.h"
#include "stm32f1xx_hal.h"

#include <stdio.h>
#include <string.h>

#define GRIPPER_DEFAULT_SERVO_ID         6U
#define GRIPPER_DEFAULT_TACTILE_DEV_ADDR 1U
#define GRIPPER_DEFAULT_OPEN_LIMIT_RAW   900U
#define GRIPPER_DEFAULT_CLOSE_LIMIT_RAW  3100U
#define GRIPPER_DEFAULT_MOVE_TIME_MS     100U
#define GRIPPER_DEFAULT_POLL_PERIOD_MS   40U
#define GRIPPER_HYBRID_SETTLE_MS         160U
#define GRIPPER_HYBRID_CONFIRM_FRAMES    2U
#define GRIPPER_DEFAULT_TARGET_FORCE     8.0f
#define GRIPPER_DEFAULT_KP               0.8f
#define GRIPPER_DEFAULT_KI               0.08f
#define GRIPPER_DEFAULT_KD               0.0f
#define GRIPPER_DEFAULT_CONTACT_TH       4.0f
#define GRIPPER_DEFAULT_HARD_LIMIT       18.0f
#define GRIPPER_DEFAULT_DEADBAND         2.0f
#define GRIPPER_DEFAULT_MAX_STEP         6.0f
#define GRIPPER_DEFAULT_INTEGRAL_LIMIT   25.0f
#define GRIPPER_FORCE_FILTER_ALPHA       0.25f
#define GRIPPER_TACTILE_TX_TIMEOUT_MS    20U
#define GRIPPER_TACTILE_RX_TIMEOUT_MS    30U

typedef struct
{
    gripper_control_status_t status;
    float filtered_force;
    float last_force;
    uint8_t hybrid_contact_latched;
    uint8_t hybrid_close_confirm;
    uint8_t hybrid_release_confirm;
    uint32_t hybrid_settle_until_ms;
} gripper_runtime_t;

static gripper_runtime_t s_gripper;

static float absf_local(float value)
{
    return (value < 0.0f) ? -value : value;
}

static float maxf_local(float a, float b)
{
    return (a > b) ? a : b;
}

static int32_t round_to_i32(float value)
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

    scaled = round_to_i32(value * (float)scale);
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

static const char *mode_name(gripper_mode_t mode)
{
    switch (mode) {
    case GRIPPER_MODE_IDLE:
        return "idle";
    case GRIPPER_MODE_POSITION:
        return "position";
    case GRIPPER_MODE_CONTACT_HOLD:
        return "contact_hold";
    case GRIPPER_MODE_FORCE_PI:
        return "force_pi";
    case GRIPPER_MODE_HYBRID_HOLD:
        return "hybrid_hold";
    default:
        return "unknown";
    }
}

static void set_status_text(const char *text)
{
    if (text == NULL || text[0] == '\0') {
        text = "ok";
    }
    snprintf(
        s_gripper.status.status_text,
        sizeof(s_gripper.status.status_text),
        "%s",
        text
    );
}

static uint16_t clamp_position(uint16_t pos_raw)
{
    uint16_t lower = s_gripper.status.open_limit_raw;
    uint16_t upper = s_gripper.status.close_limit_raw;
    if (lower > upper) {
        uint16_t tmp = lower;
        lower = upper;
        upper = tmp;
    }

    if (pos_raw < lower) {
        return lower;
    }
    if (pos_raw > upper) {
        return upper;
    }
    return pos_raw;
}

static void clear_controller_state(void)
{
    s_gripper.status.integrator = 0.0f;
    s_gripper.status.last_error = 0.0f;
    s_gripper.status.last_derivative = 0.0f;
    s_gripper.filtered_force = 0.0f;
    s_gripper.last_force = 0.0f;
    s_gripper.status.contact_active = 0U;
    s_gripper.status.source_connected = 0U;
    s_gripper.hybrid_contact_latched = 0U;
    s_gripper.hybrid_close_confirm = 0U;
    s_gripper.hybrid_release_confirm = 0U;
    s_gripper.hybrid_settle_until_ms = 0U;
}

static int read_force_sample(float *force_out)
{
    int8_t fx = 0;
    int8_t fy = 0;
    uint8_t fz = 0U;
    int rc = tactile_read_m2020_totals(
        s_gripper.status.tactile_dev_addr,
        &fx,
        &fy,
        &fz,
        GRIPPER_TACTILE_TX_TIMEOUT_MS,
        GRIPPER_TACTILE_RX_TIMEOUT_MS
    );

    (void)fx;
    (void)fy;

    if (rc != 0) {
        return rc;
    }

    *force_out = (float)fz;
    return 0;
}

static int issue_move(uint16_t pos_raw)
{
    uint16_t clamped = clamp_position(pos_raw);
    if (clamped == s_gripper.status.commanded_pos_raw) {
        return 0;
    }
    if (servo_move(s_gripper.status.servo_id, clamped, s_gripper.status.move_time_ms) != 0) {
        s_gripper.status.fault_count++;
        set_status_text("servo_move_fail");
        return -1;
    }

    s_gripper.status.commanded_pos_raw = clamped;
    return 0;
}

static int step_close_amount(float close_delta)
{
    float signed_delta = (float)s_gripper.status.close_direction * close_delta;
    int32_t next = round_to_i32((float)s_gripper.status.commanded_pos_raw + signed_delta);

    if (next < 0) {
        next = 0;
    } else if (next > 4095) {
        next = 4095;
    }

    return issue_move((uint16_t)next);
}

static void update_force_state(float measured_force_raw)
{
    if (!s_gripper.status.tare_valid) {
        s_gripper.status.tare_force = 0.0f;
    }

    s_gripper.status.measured_force_raw = measured_force_raw;
    s_gripper.status.measured_force = measured_force_raw - s_gripper.status.tare_force;
    if (s_gripper.status.measured_force < 0.0f) {
        s_gripper.status.measured_force = 0.0f;
    }

    s_gripper.filtered_force +=
        GRIPPER_FORCE_FILTER_ALPHA * (s_gripper.status.measured_force - s_gripper.filtered_force);
    s_gripper.status.contact_active =
        (s_gripper.filtered_force >= s_gripper.status.contact_threshold) ? 1U : 0U;
}

static void run_contact_hold(void)
{
    if (s_gripper.status.contact_active) {
        set_status_text("contact_hold_stable");
        return;
    }

    if (step_close_amount(maxf_local(1.0f, s_gripper.status.max_step_per_tick)) != 0) {
        return;
    }
    set_status_text("seeking_contact");
}

static void start_hybrid_settle(uint32_t now_ms)
{
    s_gripper.hybrid_settle_until_ms = now_ms + GRIPPER_HYBRID_SETTLE_MS;
    s_gripper.hybrid_close_confirm = 0U;
    s_gripper.hybrid_release_confirm = 0U;
}

static uint8_t hybrid_is_settling(uint32_t now_ms)
{
    if (s_gripper.hybrid_settle_until_ms == 0U) {
        return 0U;
    }
    if ((int32_t)(s_gripper.hybrid_settle_until_ms - now_ms) > 0) {
        return 1U;
    }
    s_gripper.hybrid_settle_until_ms = 0U;
    return 0U;
}

static void run_force_pi(float dt)
{
    float error = 0.0f;
    float derivative = 0.0f;
    float control = 0.0f;
    float step = 0.0f;
    float unclamped_integrator = 0.0f;
    float clamped_integrator = 0.0f;

    if (s_gripper.filtered_force >= s_gripper.status.hard_force_limit) {
        s_gripper.status.integrator = 0.0f;
        (void)step_close_amount(-maxf_local(2.0f, s_gripper.status.max_step_per_tick));
        set_status_text("overforce_backoff");
        return;
    }

    if (!s_gripper.status.contact_active) {
        s_gripper.status.integrator = 0.0f;
        if (step_close_amount(maxf_local(1.0f, s_gripper.status.max_step_per_tick)) != 0) {
            return;
        }
        set_status_text("force_seek_contact");
        return;
    }

    error = s_gripper.status.target_force - s_gripper.filtered_force;
    derivative = (dt > 0.0f) ? ((s_gripper.filtered_force - s_gripper.last_force) / dt) : 0.0f;
    s_gripper.status.last_derivative = derivative;

    if (absf_local(error) <= s_gripper.status.deadband) {
        s_gripper.status.integrator *= 0.92f;
        s_gripper.status.last_error = error;
        set_status_text("force_hold");
        return;
    }

    unclamped_integrator = s_gripper.status.integrator + error * dt;
    clamped_integrator = unclamped_integrator;
    if (clamped_integrator > s_gripper.status.integrator_limit) {
        clamped_integrator = s_gripper.status.integrator_limit;
    } else if (clamped_integrator < -s_gripper.status.integrator_limit) {
        clamped_integrator = -s_gripper.status.integrator_limit;
    }
    s_gripper.status.integrator = clamped_integrator;

    control =
        s_gripper.status.kp * error +
        s_gripper.status.ki * s_gripper.status.integrator -
        s_gripper.status.kd * derivative;

    step = control;
    if (step > s_gripper.status.max_step_per_tick) {
        step = s_gripper.status.max_step_per_tick;
    } else if (step < -s_gripper.status.max_step_per_tick) {
        step = -s_gripper.status.max_step_per_tick;
    }

    if (absf_local(step) < 1.0f) {
        set_status_text("force_trim_small");
        s_gripper.status.last_error = error;
        return;
    }

    if (step_close_amount(step) != 0) {
        return;
    }

    s_gripper.status.last_error = error;
    if (step > 0.0f) {
        set_status_text("force_close");
    } else {
        set_status_text("force_release");
    }
}

static void run_hybrid_hold(float dt, uint32_t now_ms)
{
    float error = 0.0f;
    float derivative = 0.0f;
    float control = 0.0f;
    float step = 0.0f;
    float unclamped_integrator = 0.0f;
    float clamped_integrator = 0.0f;
    float max_close_step = maxf_local(1.0f, s_gripper.status.max_step_per_tick);
    float max_release_step = maxf_local(1.0f, s_gripper.status.max_step_per_tick * 0.5f);

    if (s_gripper.filtered_force >= s_gripper.status.hard_force_limit) {
        s_gripper.status.integrator = 0.0f;
        s_gripper.hybrid_close_confirm = 0U;
        s_gripper.hybrid_release_confirm = 0U;
        (void)step_close_amount(-max_release_step);
        start_hybrid_settle(now_ms);
        set_status_text("hybrid_overforce");
        return;
    }

    if (!s_gripper.status.contact_active) {
        s_gripper.hybrid_contact_latched = 0U;
        s_gripper.hybrid_settle_until_ms = 0U;
        s_gripper.hybrid_close_confirm = 0U;
        s_gripper.hybrid_release_confirm = 0U;
        s_gripper.status.integrator = 0.0f;
        if (step_close_amount(max_close_step) != 0) {
            return;
        }
        set_status_text("hybrid_seek_contact");
        return;
    }

    if (!s_gripper.hybrid_contact_latched) {
        s_gripper.hybrid_contact_latched = 1U;
        s_gripper.status.integrator = 0.0f;
        s_gripper.status.last_error = 0.0f;
        s_gripper.status.last_derivative = 0.0f;
        start_hybrid_settle(now_ms);
        set_status_text("hybrid_contact_latched");
        return;
    }

    if (hybrid_is_settling(now_ms)) {
        set_status_text("hybrid_settle");
        return;
    }

    error = s_gripper.status.target_force - s_gripper.filtered_force;
    derivative = (dt > 0.0f) ? ((s_gripper.filtered_force - s_gripper.last_force) / dt) : 0.0f;
    s_gripper.status.last_derivative = derivative;

    if (absf_local(error) <= s_gripper.status.deadband) {
        s_gripper.hybrid_close_confirm = 0U;
        s_gripper.hybrid_release_confirm = 0U;
        s_gripper.status.integrator *= 0.92f;
        s_gripper.status.last_error = error;
        set_status_text("hybrid_hold");
        return;
    }

    if (error > 0.0f) {
        s_gripper.hybrid_close_confirm++;
        s_gripper.hybrid_release_confirm = 0U;
        if (s_gripper.hybrid_close_confirm < GRIPPER_HYBRID_CONFIRM_FRAMES) {
            s_gripper.status.last_error = error;
            set_status_text("hybrid_wait_close");
            return;
        }
    } else {
        s_gripper.hybrid_release_confirm++;
        s_gripper.hybrid_close_confirm = 0U;
        if (s_gripper.hybrid_release_confirm < GRIPPER_HYBRID_CONFIRM_FRAMES) {
            s_gripper.status.last_error = error;
            set_status_text("hybrid_wait_release");
            return;
        }
    }

    unclamped_integrator = s_gripper.status.integrator + error * dt;
    clamped_integrator = unclamped_integrator;
    if (clamped_integrator > s_gripper.status.integrator_limit) {
        clamped_integrator = s_gripper.status.integrator_limit;
    } else if (clamped_integrator < -s_gripper.status.integrator_limit) {
        clamped_integrator = -s_gripper.status.integrator_limit;
    }
    s_gripper.status.integrator = clamped_integrator;

    control =
        s_gripper.status.kp * error +
        s_gripper.status.ki * s_gripper.status.integrator -
        s_gripper.status.kd * derivative;

    step = control;
    if (step > max_close_step) {
        step = max_close_step;
    } else if (step < -max_release_step) {
        step = -max_release_step;
    }

    if (absf_local(step) < 1.0f) {
        step = (error > 0.0f) ? 1.0f : -1.0f;
    }

    if (step_close_amount(step) != 0) {
        return;
    }

    s_gripper.status.last_error = error;
    s_gripper.hybrid_close_confirm = 0U;
    s_gripper.hybrid_release_confirm = 0U;
    start_hybrid_settle(now_ms);
    if (step > 0.0f) {
        set_status_text("hybrid_trim_close");
    } else {
        set_status_text("hybrid_trim_release");
    }
}

void gripper_control_init(void)
{
    memset(&s_gripper, 0, sizeof(s_gripper));
    s_gripper.status.mode = GRIPPER_MODE_IDLE;
    s_gripper.status.servo_id = GRIPPER_DEFAULT_SERVO_ID;
    s_gripper.status.tactile_dev_addr = GRIPPER_DEFAULT_TACTILE_DEV_ADDR;
    s_gripper.status.close_direction = 1;
    s_gripper.status.open_limit_raw = GRIPPER_DEFAULT_OPEN_LIMIT_RAW;
    s_gripper.status.close_limit_raw = GRIPPER_DEFAULT_CLOSE_LIMIT_RAW;
    s_gripper.status.commanded_pos_raw = 2048U;
    s_gripper.status.move_time_ms = GRIPPER_DEFAULT_MOVE_TIME_MS;
    s_gripper.status.poll_period_ms = GRIPPER_DEFAULT_POLL_PERIOD_MS;
    s_gripper.status.target_force = GRIPPER_DEFAULT_TARGET_FORCE;
    s_gripper.status.contact_threshold = GRIPPER_DEFAULT_CONTACT_TH;
    s_gripper.status.hard_force_limit = GRIPPER_DEFAULT_HARD_LIMIT;
    s_gripper.status.kp = GRIPPER_DEFAULT_KP;
    s_gripper.status.ki = GRIPPER_DEFAULT_KI;
    s_gripper.status.kd = GRIPPER_DEFAULT_KD;
    s_gripper.status.deadband = GRIPPER_DEFAULT_DEADBAND;
    s_gripper.status.max_step_per_tick = GRIPPER_DEFAULT_MAX_STEP;
    s_gripper.status.integrator_limit = GRIPPER_DEFAULT_INTEGRAL_LIMIT;
    set_status_text("idle");
    (void)gripper_control_sync_command_to_current();
}

void gripper_control_poll(void)
{
    uint32_t now_ms = HAL_GetTick();
    uint32_t elapsed_ms = now_ms - s_gripper.status.last_update_ms;
    float measured_force_raw = 0.0f;
    float dt = 0.0f;
    int rc = 0;

    if (s_gripper.status.mode == GRIPPER_MODE_IDLE || s_gripper.status.mode == GRIPPER_MODE_POSITION) {
        return;
    }

    if (elapsed_ms < s_gripper.status.poll_period_ms) {
        return;
    }

    s_gripper.status.last_update_ms = now_ms;
    dt = (elapsed_ms > 0U) ? ((float)elapsed_ms / 1000.0f) : 0.0f;

    rc = read_force_sample(&measured_force_raw);
    if (rc != 0) {
        s_gripper.status.source_connected = 0U;
        s_gripper.status.fault_count++;
        set_status_text("tactile_read_fail");
        return;
    }

    s_gripper.status.source_connected = 1U;
    s_gripper.status.update_count++;
    update_force_state(measured_force_raw);

    if (s_gripper.status.mode == GRIPPER_MODE_CONTACT_HOLD) {
        run_contact_hold();
    } else if (s_gripper.status.mode == GRIPPER_MODE_FORCE_PI) {
        run_force_pi(dt);
    } else if (s_gripper.status.mode == GRIPPER_MODE_HYBRID_HOLD) {
        run_hybrid_hold(dt, now_ms);
    }

    s_gripper.last_force = s_gripper.filtered_force;
}

int gripper_control_status(gripper_control_status_t *out)
{
    if (out == NULL) {
        return -1;
    }
    *out = s_gripper.status;
    return 0;
}

int gripper_control_format_status(char *out, size_t out_cap)
{
    char force_buf[24];
    char raw_buf[24];
    char tare_buf[24];
    char target_buf[24];
    char kp_buf[24];
    char ki_buf[24];
    char kd_buf[24];
    char deadband_buf[24];
    char step_buf[24];

    if (out == NULL || out_cap == 0U) {
        return -1;
    }

    format_fixed_local(force_buf, sizeof(force_buf), s_gripper.filtered_force, 2U);
    format_fixed_local(raw_buf, sizeof(raw_buf), s_gripper.status.measured_force_raw, 2U);
    format_fixed_local(tare_buf, sizeof(tare_buf), s_gripper.status.tare_force, 2U);
    format_fixed_local(target_buf, sizeof(target_buf), s_gripper.status.target_force, 2U);
    format_fixed_local(kp_buf, sizeof(kp_buf), s_gripper.status.kp, 3U);
    format_fixed_local(ki_buf, sizeof(ki_buf), s_gripper.status.ki, 3U);
    format_fixed_local(kd_buf, sizeof(kd_buf), s_gripper.status.kd, 3U);
    format_fixed_local(deadband_buf, sizeof(deadband_buf), s_gripper.status.deadband, 2U);
    format_fixed_local(step_buf, sizeof(step_buf), s_gripper.status.max_step_per_tick, 2U);

    snprintf(
        out,
        out_cap,
        "GSTAT mode=%s servo=%u dev=%u dir=%d pos=%u force=%s raw=%s tare=%s target=%s "
        "contact=%u src=%u kp=%s ki=%s kd=%s db=%s step=%s period=%u status=%s",
        mode_name(s_gripper.status.mode),
        (unsigned)s_gripper.status.servo_id,
        (unsigned)s_gripper.status.tactile_dev_addr,
        (int)s_gripper.status.close_direction,
        (unsigned)s_gripper.status.commanded_pos_raw,
        force_buf,
        raw_buf,
        tare_buf,
        target_buf,
        (unsigned)s_gripper.status.contact_active,
        (unsigned)s_gripper.status.source_connected,
        kp_buf,
        ki_buf,
        kd_buf,
        deadband_buf,
        step_buf,
        (unsigned)s_gripper.status.poll_period_ms,
        s_gripper.status.status_text
    );
    return 0;
}

int gripper_control_set_servo_id(uint8_t servo_id)
{
    if (servo_id == 0U || servo_id > 250U || servo_id == 0xFEU) {
        return -1;
    }
    s_gripper.status.servo_id = servo_id;
    return 0;
}

int gripper_control_set_tactile_dev_addr(uint8_t dev_addr)
{
    if (dev_addr == 0U) {
        return -1;
    }
    s_gripper.status.tactile_dev_addr = dev_addr;
    return 0;
}

int gripper_control_set_close_direction(int8_t direction)
{
    if (direction != -1 && direction != 1) {
        return -1;
    }
    s_gripper.status.close_direction = direction;
    return 0;
}

int gripper_control_set_limits(uint16_t open_limit_raw, uint16_t close_limit_raw)
{
    if (open_limit_raw == close_limit_raw) {
        return -1;
    }
    s_gripper.status.open_limit_raw = open_limit_raw;
    s_gripper.status.close_limit_raw = close_limit_raw;
    s_gripper.status.commanded_pos_raw = clamp_position(s_gripper.status.commanded_pos_raw);
    return 0;
}

int gripper_control_set_pid(float kp, float ki, float kd)
{
    if (kp < 0.0f || ki < 0.0f || kd < 0.0f) {
        return -1;
    }
    s_gripper.status.kp = kp;
    s_gripper.status.ki = ki;
    s_gripper.status.kd = kd;
    return 0;
}

int gripper_control_set_contact_threshold(float threshold)
{
    if (threshold < 0.0f) {
        return -1;
    }
    s_gripper.status.contact_threshold = threshold;
    return 0;
}

int gripper_control_set_hard_force_limit(float limit)
{
    if (limit <= 0.0f) {
        return -1;
    }
    s_gripper.status.hard_force_limit = limit;
    return 0;
}

int gripper_control_set_deadband(float deadband)
{
    if (deadband < 0.0f) {
        return -1;
    }
    s_gripper.status.deadband = deadband;
    return 0;
}

int gripper_control_set_max_step(float max_step_per_tick)
{
    if (max_step_per_tick <= 0.0f) {
        return -1;
    }
    s_gripper.status.max_step_per_tick = max_step_per_tick;
    return 0;
}

int gripper_control_set_poll_period(uint16_t poll_period_ms)
{
    if (poll_period_ms < 10U || poll_period_ms > 1000U) {
        return -1;
    }
    s_gripper.status.poll_period_ms = poll_period_ms;
    return 0;
}

int gripper_control_set_move_time(uint16_t move_time_ms)
{
    if (move_time_ms > 5000U) {
        return -1;
    }
    s_gripper.status.move_time_ms = move_time_ms;
    return 0;
}

int gripper_control_tare(uint8_t sample_count)
{
    uint8_t requested = sample_count == 0U ? 5U : sample_count;
    float sum = 0.0f;
    uint8_t ok_count = 0U;

    for (uint8_t i = 0U; i < requested; i++) {
        float sample = 0.0f;
        if (read_force_sample(&sample) != 0) {
            HAL_Delay(5U);
            continue;
        }
        sum += sample;
        ok_count++;
        HAL_Delay(5U);
    }

    if (ok_count == 0U) {
        set_status_text("tare_failed");
        return -1;
    }

    s_gripper.status.tare_force = sum / (float)ok_count;
    s_gripper.status.tare_valid = 1U;
    s_gripper.status.measured_force = 0.0f;
    s_gripper.status.measured_force_raw = 0.0f;
    s_gripper.filtered_force = 0.0f;
    s_gripper.last_force = 0.0f;
    set_status_text("tare_captured");
    return 0;
}

int gripper_control_sync_command_to_current(void)
{
    uint16_t pos = 0U;
    if (servo_read_pos(s_gripper.status.servo_id, &pos) != 0) {
        s_gripper.status.commanded_pos_raw = clamp_position(s_gripper.status.commanded_pos_raw);
        set_status_text("sync_pos_fail");
        return -1;
    }

    s_gripper.status.commanded_pos_raw = clamp_position(pos);
    return 0;
}

int gripper_control_move_raw(uint16_t pos_raw, uint16_t move_time_ms)
{
    if (move_time_ms > 0U) {
        s_gripper.status.move_time_ms = move_time_ms;
    }
    s_gripper.status.mode = GRIPPER_MODE_POSITION;
    clear_controller_state();
    set_status_text("manual_position");
    return issue_move(pos_raw);
}

int gripper_control_start_contact_hold(void)
{
    (void)gripper_control_sync_command_to_current();
    clear_controller_state();
    s_gripper.status.mode = GRIPPER_MODE_CONTACT_HOLD;
    set_status_text("contact_hold_start");
    return 0;
}

int gripper_control_start_force_pi(float target_force)
{
    if (target_force < 0.0f) {
        return -1;
    }
    s_gripper.status.target_force = target_force;
    (void)gripper_control_sync_command_to_current();
    clear_controller_state();
    s_gripper.status.mode = GRIPPER_MODE_FORCE_PI;
    set_status_text("force_pi_start");
    return 0;
}

int gripper_control_start_hybrid_hold(float target_force)
{
    if (target_force < 0.0f) {
        return -1;
    }
    s_gripper.status.target_force = target_force;
    (void)gripper_control_sync_command_to_current();
    clear_controller_state();
    s_gripper.status.mode = GRIPPER_MODE_HYBRID_HOLD;
    set_status_text("hybrid_start");
    return 0;
}

int gripper_control_stop(void)
{
    s_gripper.status.mode = GRIPPER_MODE_IDLE;
    clear_controller_state();
    set_status_text("idle");
    return 0;
}
