################################################################################
# Automatically-generated file. Do not edit!
################################################################################

SHELL = cmd.exe

# Each subdirectory must supply rules for building sources it contributes
build-1885960561:
	@$(MAKE) --no-print-directory -Onone -f subdir_rules.mk build-1885960561-inproc

build-1885960561-inproc: ../IMU_Stepper.ino
	@echo 'Building file: "$<"'
	@echo 'Invoking: Resource Custom Build Step'
	
	@echo 'Finished building: "$<"'
	@echo ' '

IMU_Stepper.cpp: build-1885960561 ../IMU_Stepper.ino
main.cpp: build-1885960561

%.o: ./%.cpp $(GEN_OPTS) | $(GEN_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: GNU Compiler'
	"C:/ti/energia_tool_arm-none-eabi-gcc_6.3.1-20170620/bin/arm-none-eabi-gcc.exe" -c -mcpu=cortex-m4 -march=armv7e-m -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -Dxdc_target_types__=gnu/targets/arm/std.h -Dxdc_target_name__=M4F -Dxdc_cfg__xheader__=\"configPkg/package/cfg/energia_pm4fg.h\" -DBOARD_MSP_EXP432P401R -Dxdc__nolocalstring=1 -D__MSP432P401R__ -DCORE_VERSION=5251 -DF_CPU=48000000L -DENERGIA_MSP_EXP432P401R -DENERGIA_ARCH_MSP432R -D_DEFAULT_SOURCE -DENERGIA=21 -DARDUINO=10610 -I"C:/Users/scott/AppData/Local/Energia15/packages/energia/hardware/msp432r/5.25.1/cores/msp432r/ti/runtime/wiring/" -I"C:/Users/scott/AppData/Local/Energia15/packages/energia/hardware/msp432r/5.25.1/cores/msp432r/ti/runtime/wiring/msp432" -I"C:/Users/scott/AppData/Local/Energia15/packages/energia/hardware/msp432r/5.25.1/system/kernel/tirtos/packages/ti/sysbios/posix" -I"C:/Users/scott/AppData/Local/Energia15/packages/energia/hardware/msp432r/5.25.1/system/kernel/tirtos/packages" -I"C:/Users/scott/AppData/Local/Energia15/packages/energia/hardware/msp432r/5.25.1/system/source" -I"C:/Users/scott/AppData/Local/Energia15/packages/energia/hardware/msp432r/5.25.1/system/energia" -I"C:/Users/scott/AppData/Local/Energia15/packages/energia/hardware/msp432r/5.25.1/system/source/ti/devices/msp432p4xx/driverlib" -I"C:/Users/scott/AppData/Local/Energia15/packages/energia/hardware/msp432r/5.25.1/system/source/ti/devices/msp432p4xx/inc" -I"C:/Users/scott/AppData/Local/Energia15/packages/energia/hardware/msp432r/5.25.1/system/source/ti/devices/msp432p4xx/" -I"C:/Users/scott/AppData/Local/Energia15/packages/energia/hardware/msp432r/5.25.1/system/source/third_party/CMSIS/Include" -I"C:/Users/scott/AppData/Local/Energia15/packages/energia/hardware/msp432r/5.25.1/cores/msp432r" -I"C:/Users/scott/AppData/Local/Energia15/packages/energia/hardware/msp432r/5.25.1/variants/MSP_EXP432P401R" -I"C:/Users/scott/OneDrive/Documents/SJSU/19B_FA/ENGR160SL/MC workspace/IMU_Stepper" -I"C:/Users/scott/OneDrive/Documents/energia-1.8.7E21/libraries/Adafruit_Unified_Sensor-1.0.3" -I"C:/Users/scott/OneDrive/Documents/energia-1.8.7E21/libraries/Adafruit_BNO055" -I"C:/Users/scott/OneDrive/Documents/energia-1.8.7E21/libraries/Stepper" -I"C:/Users/scott/AppData/Local/Energia15/packages/energia/hardware/msp432r/5.25.1/libraries/Servo" -I"C:/ti/energia_tool_arm-none-eabi-gcc_6.3.1-20170620/arm-none-eabi/include" -Os -ffunction-sections -fdata-sections -g -gdwarf-3 -gstrict-dwarf -w -Wall -mabi=aapcs -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)"  -fno-rtti -fno-threadsafe-statics -fno-exceptions $(GEN_OPTS__FLAG) -o"$@" "$<"
	@echo 'Finished building: "$<"'
	@echo ' '


