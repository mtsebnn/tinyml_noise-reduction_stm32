#include "nn_data.h"
#include "filters.h"

#include <stdint.h>
#include <stdio.h>

// register definition
// register addresses from technical reference manuals
// access memory cell at the exact address (using pointers)

// Debug Exception and Monitor Control Register used to enable Trace-infrastructure of the CPU
// bit 24 in this register is the TRCENA-Bit (Trance Enable) -> setting this to 1 enables debug tracing and DWT
#define DEMCR (*((volatile uint32_t*)0xE000EDFC))

// DWT (Data watchpoint and Trace Unit) Control Register
#define DWT_CTRL (*((volatile uint32_t*)0xE0001000)) // bit 0 is the CYCCNTENA-Bit (activates the cycle counter)
#define DWT_CYCCNT (*((volatile uint32_t*)0xE0001004)) // DWT Cycle Count Register (32-bit hardware counter)

// RCC (Reset and Clock Control) 0x40023800 - 0x4002 3BFF
#define RCC_AHB1ENR (*((volatile uint32_t*)0x40023830)) // AHB1 periph. clk enable reg. (turn on clock for periph. connected to the AHB1-bus)
#define RCC_APB1ENR (*((volatile uint32_t*)0x40023840)) // APB1 periph. clk enaable reg. (turn on clock for periph. connected to the APB1-bus)

// GPIOA registers (connected to fast AHB1-bus) for the pin configuration 0x40020000 - 0x400203FF
#define GPIOA_MODER (*((volatile uint32_t*)0x40020000)) // GPIO port mode reg. (set the mode of a pin)
#define GPIOA_AFRL (*((volatile uint32_t*)0x40020020)) // GPIO altern. func. low reg. (connect pin to a specific internal peripheral (i.e. USART2))

// USART2 registers (connected to slower APB1-bus) 0x40004400 - 0x400047FF
#define USART2_SR (*((volatile uint32_t*)0x40004400)) // status reg. (check wether the hardware is ready to send/receive the next character)
#define USART2_DR (*((volatile uint32_t*)0x40004404)) // data reg. (buffer reg. to send/receive data bytes serialized over the TX pin)
#define USART2_BRR (*((volatile uint32_t*)0x40004408)) // baud rate reg. (set baudrate so the PC reads at the same clockspeed as the MC sends)
#define USART2_CR1 (*((volatile uint32_t*)0x4000440C)) // control reg. 1 (activate USART2-module and its transmitter (TX))


void init_cyc_cnt(void) {
    DEMCR |= (1 << 24); // set TRCENA-Bit ("a |= b" is bitwise OR operator -> turn on bits in a that are turned on in b)
    DWT_CYCCNT = 0;
    DWT_CTRL |= (1 << 0); // set CYCCNTENA-Bit
}

void init_uart(void) {
    RCC_AHB1ENR |= (1 << 0); // activate clock for GPIOA (bit 0)
    RCC_APB1ENR |= (1 << 17); // activate clock for USART2 (bit 17)

    // Only USART2 connected to st-link -> use PA2 (send) and PA3 (read) (STM32f446re datasheet, table 11) 
    // Configure pin PA2 and PA3 as altern. func. 
    // (bitwise config.: 00 Input, 01 Output, 10 Altern. func., 11 analog)
    GPIOA_MODER &= ~((3 << 4) | (3 << 6)); // set bits 5:4 and 7:6 = 00 (configuration bits for pin 2 and 3 of GPIOA)
    GPIOA_MODER |= ((2 << 4) | (2 << 6));  // set bits 5:4 and 7:6 = 10 => alternative function mode
    GPIOA_AFRL &= ~((0xF << 8) | (0xF << 12)); // set bits 11:8 and 15:12 = 0000 (AF-bits for PA2)
    GPIOA_AFRL |= ((7 << 8) | (7 << 12)); // set bits 11:8 and 15:12 = 0111 => turn on connection AF7 (connect PA2 to transmitter (TX) and PA3 to receiver of USART2)

    // use baudrate 115200 (bits/second) at 16MHz
    // DIV = 16MHz / (16 * 115200) = 8,6805
    USART2_BRR = (8 << 4) | 11; // USART_BRR expects DIV_Mantissa bits 15:4 and DIV_Fraction bits 3:0
    USART2_CR1 = (1 << 13) | (1 << 3) | (1 << 2); // set bit 13 = 1 (USART ENABLE), bit 3 = 1 (TRANSMITTER ENABLE) and bit 2 =1 (RECEIVER ENABLE)
}


// low-level support function used by printf()
int _write (int file, char *p, int len) {
    for (int i = 0; i < len; i++) {
        while(!(USART2_SR & (1 << 7))); // wait until USART2_SR TXE (Transmit Data Register Empty) is empty (bit 7 = 1)
        USART2_DR = p[i];
    }
    return len; // return that "len" chars have been written
}

// wait for and read a char via USART
char uart_read_char(void) {
    while (!(USART2_SR & (1 << 5))); // wait until USART2_SR RXNE (Read Data Register Not Empty) is not empty (bit 5 = 1)
    return (char)(USART2_DR & 0xFF);
}


int main(void) {
    init_cyc_cnt();
    init_uart();

    setvbuf(stdout, NULL, _IONBF, 0);

    // input window
    int8_t input_window[WINDOW_SIZE];

    while (1) {
        char cmd = uart_read_char();

        if (cmd == 'R') {
            reset_filters();
        }
        else if (cmd == 'S') {

            for (int i = 0; i < WINDOW_SIZE; i++) {
                input_window[i] = (int8_t)uart_read_char();
            }

            // 1. neural network
            uint32_t start_cycles_nn = DWT_CYCCNT;
            int32_t predicted_nn = predict_nn(input_window, &data);
            uint32_t cycles_nn = DWT_CYCCNT - start_cycles_nn;

            int8_t current_sample = input_window[8];

            // 2. Kalman filter
            uint32_t start_cycles_kal = DWT_CYCCNT;
            int32_t predicted_kal = predict_kalman(current_sample);
            uint32_t cycles_kal = DWT_CYCCNT - start_cycles_kal;

            // 3. Wiener filter
            uint32_t start_cycles_wie = DWT_CYCCNT;
            int32_t predicted_wie = predict_wiener(input_window);
            uint32_t cycles_wie = DWT_CYCCNT - start_cycles_wie;

            // 4. IIR filter
            uint32_t start_cycles_iir = DWT_CYCCNT;
            int32_t predicted_iir = predict_iir(current_sample);
            uint32_t cycles_iir = DWT_CYCCNT - start_cycles_iir;


            // print out total inference cycles and predicted signals
            printf("%lu, %ld, %lu, %ld, %lu, %ld, %lu, %ld\n", 
                cycles_nn, predicted_nn,
                cycles_kal, predicted_kal,
                cycles_wie, predicted_wie,
                cycles_iir, predicted_iir
            ); // %lu = unsigned long, %ld = long, %.2f = float
        }
    }
}


extern uint32_t _sdata, _edata, _sidata, _sbss, _ebss, _estack;
void Reset_Handler(void) {
    uint32_t *src = &_sidata;
    uint32_t *dst = &_sdata;

    while (dst< &_edata) {
        *dst++ = *src++;
    }

    dst = &_sbss;
    while (dst < &_ebss) {
        *dst++ = 0;
    }
    main();
}
const void *isr_vector_table[] __attribute__((section(".isr_vector"))) = {&_estack, (void*)Reset_Handler};
