/*!
 * @file setCurrent.ino
 * @brief This I2C 0-25mA DAC module can be used to output a current of 0-25mA.
 * @note The carrying capacity of this module is related to the external power supply voltage: range 18-24V, maximum 450R carrying capacity for 18V power supply, and 650R for 24V.
 * @n This demo is used for demonstration. Control the module to output the current of 10mA and save the config to make sure it is not lost when the module is powered up again.
 * @n The hardware connection in this demo:
 * @n 1. Disable  Macro #define I2C_PIN_REMAP_ENABLE I2C uses the default I2C pins, i.e., using the pin corresponding to the MCU hardware I2C.
 * @n 2. Enable  Macro #define I2C_PIN_REMAP_ENABLE I2C pin remapping macro, the way of connecting SCL and SDA pins is as shown in the following table:
 * @n ---------------------------------------------------------------------------------
 * @n |Module | UNO     | Leonardo  |  Mega2560 |     ESP32    |     M0    | microbit |
 * @n ---------------------------------------------------------------------------------
 * @n |SCL    |  3      |     3     |     3     |   SCL(26)(D3)|   SCL(3)  |  SCL(P8) |
 * @n |SDA    |  5      |     5     |     5     |   SDA(27)(D4)|   SDA(5)  |  SDA(P9) |
 * @n ---------------------------------------------------------------------------------
 * @note When using I2C pin remapping, besides the pins listed above, other IO pins of the MCU can also be selected
 * @note esp8266 does not support I2C pin remapping
 *
 * @copyright   Copyright (c) 2010 DFRobot Co.Ltd (http://www.dfrobot.com)
 * @license     The MIT License (MIT)
 * @author [Arya](xue.peng@dfrobot.com)
 * @version  V1.0
 * @date  2022-03-02
 * @url https://github.com/DFRobot/DFRobot_GP8302
 */

 #include "DFRobot_GP8302.h"
#include <Arduino.h>
#include <math.h>


 #define I2C_PIN_REMAP_ENABLE  //Enable I2C pin remapping, remap it to the pins defined by I2C_SCL_PIN and I2C_SDA_PIN, if it's not enabled, the pins used by the hardware Wire object will be used by default.
 
 #ifdef I2C_PIN_REMAP_ENABLE
 #define I2C_SCL_PIN 8
 #define I2C_SDA_PIN 7
 #endif
 
 DFRobot_GP8302 module;
 
 // --- Control and serial command state ---
 static float targetCurrent_mA = 0.0f;   // Desired current set via serial, 0-25 mA
 static float currentCurrent_mA = 0.0f;  // Current output value we are ramping
 static const float kMaxCurrent_mA = 25.0f;
 static const float kMinCurrent_mA = 0.0f;
 static const float kRampStep_mA = 0.1f; // 0.1 mA per second
 static const unsigned long kRampIntervalMs = 1000UL; // step every 1s
 static const unsigned long kPrintIntervalMs = 1000UL; // print at most once per second
 
 static unsigned long lastRampMs = 0;
 static unsigned long lastPrintMs = 0;
 
 // Simple serial command buffer
 static String cmdBuf;
 static unsigned long lastSerialByteMs = 0;
 static const unsigned long kCmdTimeoutMs = 250UL; // if no new byte for 250ms, try to parse
 
 void setup(){
  Serial.begin(115200);
  
  while(!Serial){
    //Wait for USB serial port to connect. Needed for native USB port only
  }

  Serial.print("I2C to 0-25 mA analog current moudle initialization ... ");
 
 #ifdef I2C_PIN_REMAP_ENABLE
   uint8_t status = module.begin(/* scl = */I2C_SCL_PIN, /*sda = */I2C_SDA_PIN);  //I2C scl and sda pins redefine
 #else
   uint8_t status = module.begin(); // Default to use the pins used by the MCU hardware I2C Wire object
 #endif
 
   if(status != 0){
     Serial.print("failed. Error code: ");
     Serial.println(status);
     Serial.println("Error Code: ");
     Serial.println("\t1: _scl or _sda pin is invaild.");
     Serial.println("\t2: Device not found, please check if the device is connected.");
     while(1) yield();
   }
   Serial.println("done!");
   
   /**
    * @fn output
    * @brief Set the device to output the current of 0-25mA
    * @param current_mA  The output current value, range: 0-25mA
    * @return The DAC value corresponding to the output current value
    * @note calibration4_20mA After calibration, the output function will output the calibrated current value and return the calibrated DAC value
    */
   // Initialize output at 0 mA
  uint16_t dac = module.output(/*current_mA =*/0);
  currentCurrent_mA = 0.0f;
  targetCurrent_mA = 0.0f;
  Serial.print("DAC value: "); Serial.println(dac);
 
   /**
    * @fn output_mA
    * @brief Set DAC value to control the device to output the current of 0-25mA.
    * @param dac  Specific DAC value, range 0-0xFFF
    * @note DAC value range is 0-0xFFF, 0-0xFFF DAC value corresponds to the output current of 0-25mA, the formula of DAC value converting to actual current: Iout = (DAC/0xFFF)*25mA
    * @return The actual current, unit mA
    */
   //float current = module.output_mA(/*dac =*/0x666);  //Control the DAC module to output the current corresponding to a DAC value of 0x666 and return the current corresponding to the value, unit mA
   //Serial.print("Output current :"); Serial.print(current); Serial.println(" mA.");
   
   //module.store(); Serial.println("Save current configuration."); //Uncomment this line of code, and the current config above will be saved and will not be lost after power down.
  pinMode(10, INPUT_PULLUP);
  pinMode(5,OUTPUT);
 
  Serial.println();
  Serial.println("Commands: a<n> to set current in mA (0-25). Example: a1 for 1mA, a10 for 10mA.");
  Serial.println("Ramping at 0.1 mA/sec to target.");
}


  void loop() {
  // Load detect passthrough LED/indicator
  bool DetectLoad = digitalRead(10);
  digitalWrite(5, !DetectLoad);

  // --- Serial command handling ---
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    lastSerialByteMs = millis();
    if (c == '\r' || c == '\n') {
      // parse on newline
      if (cmdBuf.length() > 0) {
        // process
        if (cmdBuf.charAt(0) == 'a' || cmdBuf.charAt(0) == 'A') {
          String num = cmdBuf.substring(1);
          long val = num.toInt();
          float t = (float)val;
          if (t < kMinCurrent_mA) t = kMinCurrent_mA;
          if (t > kMaxCurrent_mA) t = kMaxCurrent_mA;
          targetCurrent_mA = t;
          Serial.print("Target set to "); Serial.print(targetCurrent_mA, 1); Serial.println(" mA");
        }
      }
      cmdBuf = "";
    } else {
      // accumulate, keep it short
      if (cmdBuf.length() < 16) cmdBuf += c;
    }
  }
  // Optional timeout-based parse if user didn't send newline
  if (cmdBuf.length() >= 2 && (millis() - lastSerialByteMs) > kCmdTimeoutMs) {
    if (cmdBuf.charAt(0) == 'a' || cmdBuf.charAt(0) == 'A') {
      String num = cmdBuf.substring(1);
      bool digitsOnly = true;
      for (size_t i = 0; i < num.length(); ++i) {
        if (!isDigit(num.charAt(i))) { digitsOnly = false; break; }
      }
      if (digitsOnly) {
        long val = num.toInt();
        float t = (float)val;
        if (t < kMinCurrent_mA) t = kMinCurrent_mA;
        if (t > kMaxCurrent_mA) t = kMaxCurrent_mA;
        targetCurrent_mA = t;
        Serial.print("Target set to "); Serial.print(targetCurrent_mA, 1); Serial.println(" mA");
        cmdBuf = "";
      }
    }
  }

  // --- Ramping control at 0.1 mA per second ---
  unsigned long now = millis();
  if (now - lastRampMs >= kRampIntervalMs) {
    lastRampMs = now;
    if (fabsf(targetCurrent_mA - currentCurrent_mA) >= (kRampStep_mA - 1e-6)) {
      if (currentCurrent_mA < targetCurrent_mA) {
        currentCurrent_mA += kRampStep_mA;
        if (currentCurrent_mA > targetCurrent_mA) currentCurrent_mA = targetCurrent_mA;
      } else if (currentCurrent_mA > targetCurrent_mA) {
        currentCurrent_mA -= kRampStep_mA;
        if (currentCurrent_mA < targetCurrent_mA) currentCurrent_mA = targetCurrent_mA;
      }
    }
    // Apply output every step
    module.output(currentCurrent_mA);
  }

  // --- Rate-limited status print ---
  if (now - lastPrintMs >= kPrintIntervalMs) {
    lastPrintMs = now;
    Serial.print("I= "); Serial.print(currentCurrent_mA, 1); Serial.print(" mA  ");
    Serial.print("Target= "); Serial.print(targetCurrent_mA, 1); Serial.println(" mA");
  }

  delay(5);
}
 
   
 
 