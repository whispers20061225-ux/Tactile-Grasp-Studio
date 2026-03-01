STM32 USB CDC <-> UART 半双工桥接（模板）

该模板将 USB CDC 数据转发到 UART 总线（半双工），并把 UART 响应
回传给 PC。请将其集成到 CubeMX 工程中。

步骤
1) 为你的 STM32 建立 CubeMX 工程。
2) 使能 USB Device（CDC）。
3) 使能 USART 半双工（115200 8N1）。
4) 在 CDC 接收回调中调用：
   CDC_ReceiveCallback(buf, len)
5) 编译并烧录。

注意
- 使用舵机总线口，如板子需要方向控制请确保方向脚配置正确（部分板子硬件已处理）。
- 烧录后跳线切到 Servo 模式（不是 Upload）。
- 舵机板需要外接 12V 电源。
