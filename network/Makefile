# define compiler and tools
CC = arm-none-eabi-gcc
# objcopy can convert binary files (i.e. compiled programs (.elf) -> binary file (.bin))
OBJCOPY = arm-none-eabi-objcopy

# compiler flags to correctly target the architecture, instruction set, and floating-point hardware
# "-mcpu=cortex-m4" tells the compiler to generate code specifically for the ARM Cortex-M4 processor
# "-mthumb" forces the compiler to generate Thumb-2 instructions (mandatory for Cortex-M devices)
# "-mfloat-abi=hard" enables hardware floating-point instructions (not needed for this project, but standard)
# "-mfpu=fpv4-sp-d16" tells the compiler which floating-point unit (FPU) the CPU has
# "-O2" optimize generated code for performance without using too aggressive/risky optimizations
# "-Wall" enables the most useful/common warning messages
CFLAGS = -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -O2 -Wall

# linker flags
# "-T STM32F446XX_FLASH.ld" use this linker script
# "-nostartfiles" prevent automatic usage of standard startup files
# "--specs=nano.specs" use a small optimized C standard library designed for embedded microcontrollers
LINKFLAGS = -T stm32f446re.ld -nostartfiles --specs=nano.specs --specs=nosys.specs

# source files
SOURCE = main.c inference.c

# target file
TARGET = neural_network


# standard target (when using the "make" command, Make automatically builds the target "all")
# => generates neural_network.bin
all: $(TARGET).bin

$(TARGET).elf: $(SOURCE)
# build command for neural_network.elf 
# $@: automatic make-variable, file name of the target of the rule "neural_network.elf"
	$(CC) $(CFLAGS) $(SOURCE) $(LINKFLAGS) -o $@

# binary file can only be generated if the elf-file exists
$(TARGET).bin: $(TARGET).elf
# build command for neural_network.elf 
# -O binary: binary as output format
# $<: automatic make-variable, name of the first prerequisite "neural_network.elf"
# $@: automatic make-variable, file name of the target of the rule "neural_network.bin"
	$(OBJCOPY) -O binary $< $@


# remove generated files ("make clean" command)
clean:
# rm: remove files
# -f: ignore error messages if files do not exist
	rm -f $(TARGET).elf $(TARGET).bin
