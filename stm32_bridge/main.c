/*
 * STM32 USB CDC <-> UART 半双工桥接（模板）
 *
 * - USB: CDC 类（虚拟串口）
 * - UART: 半双工, 115200 8N1
 *
 * This file is a template. Integrate into a CubeMX project
 * and wire to the generated USB CDC and UART handles.
 */

#include "main.h"
#include "usbd_cdc_if.h"

extern UART_HandleTypeDef huart1; /* change to your UART */

#define UART_RX_TIMEOUT_MS 20
#define USB_TX_TIMEOUT_MS 20

static uint8_t usb_rx_buf[256];
static uint8_t uart_rx_buf[256];

/* USB CDC 收到数据后的回调 */
void CDC_ReceiveCallback(uint8_t *buf, uint32_t len)
{
    if (len == 0) {
        return;
    }

    /* 将数据转发到 UART（半双工） */
    HAL_UART_Transmit(&huart1, buf, len, UART_RX_TIMEOUT_MS);

    /* 读取 UART 响应并回传给 USB */
    uint32_t rx_len = 0;
    uint32_t start = HAL_GetTick();
    while ((HAL_GetTick() - start) < UART_RX_TIMEOUT_MS) {
        uint8_t byte = 0;
        if (HAL_UART_Receive(&huart1, &byte, 1, 1) == HAL_OK) {
            if (rx_len < sizeof(uart_rx_buf)) {
                uart_rx_buf[rx_len++] = byte;
            }
        }
    }

    if (rx_len > 0) {
        CDC_Transmit_FS(uart_rx_buf, rx_len);
    }
}

/*
 * main() 和 USB/UART 初始化由 CubeMX 生成。
 * 需要在 CDC 接收回调里调用 CDC_ReceiveCallback().
 */
int main(void)
{
    HAL_Init();
    SystemClock_Config();

    MX_GPIO_Init();
    MX_USART1_UART_Init();
    MX_USB_DEVICE_Init();

    while (1) {
        /* USB CDC callbacks handle the forwarding. */
        HAL_Delay(1);
    }
}
