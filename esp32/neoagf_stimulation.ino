/*
 * GP8413 DAC EEG Signal Generator - Memory Optimized
 */

 #include <Wire.h>
 #include <DFRobot_GP8XXX.h>
 
 // GP8413 Configuration
 #define GP8413_ADDRESS 0x59
 #define GP8413_CHANNEL_A 0x00
 
 // Signal parameters
 float freq = 10.0;        // Default 10Hz (Alpha)
 float amp = 0.005;        // 5mV amplitude
 float dcOffset = 2.5;     // Center around 2.5V
 unsigned long lastUpdate = 0;
 int sampleRate = 2000;     // Reduced sample rate
 
 // EEG frequency presets (stored in PROGMEM to save RAM)
 const float PROGMEM eegFreqs[] = {2.0, 6.0, 10.0, 20.0, 40.0};
 const char* const PROGMEM eegNames[] = {"Delta", "Theta", "Alpha", "Beta", "Gamma"};
 
 DFRobot_GP8XXX_IIC GP8413(RESOLUTION_15_BIT, 0x59, &Wire);
 
 void setup() {
   Serial.begin(115200);
   //Wire.begin();
   Wire.setPins(7, 8);
   Wire.setClock(100000);
   delay(500);
   
   Serial.println(F("=== GP8413 EEG Generator ==="));
   Serial.println(F("Commands: 0-4(bands), f<val>, a<val>, +/-, help"));
   
   while (GP8413.begin() != 0) {
     Serial.println("❌ Communication with the device failed. Please check if the connection is correct or if the device address is set correctly.");
     delay(1000);
   }
   Serial.println(F("✅ GP8413 OK"));
 
   GP8413.setDACOutRange(GP8413.eOutputRange5V);
 
   printStatus();
 }
 
 void loop() {
   if (Serial.available()) {
     handleCommand();
   }
   
   if (micros() - lastUpdate >= (1000000 / sampleRate)) {
     generateSine();
     lastUpdate = micros();
   }
 }
 
 void writeDAC(uint16_t value) {
   GP8413.setDACOutVoltage(value, 2);
 }
 
 void generateSine() {
   static float phase = 0.0; // Track phase
   static unsigned long lastTime = 0;
   unsigned long currentTime = micros();
 
   if (currentTime - lastTime >= (1000000 / sampleRate)) {
     float sineVal = sin(phase);
     float voltage = dcOffset + (amp * sineVal);
     uint16_t dacVal = (uint16_t)((voltage / 5.0) * 32767); // 15-bit DAC range is 0-32767
     Serial.print("V=");
     Serial.print(voltage);
     Serial.print(" DAC=");
     Serial.println(dacVal);
     writeDAC(dacVal);
 
     phase += 2 * PI * freq / sampleRate;
     if (phase >= 2 * PI) phase -= 2 * PI; // Wrap phase
     lastTime = currentTime;
   }
 }
 
 void handleCommand() {
   String cmd = Serial.readStringUntil('\n');
   cmd.trim();
   
   if (cmd.length() == 0) return;
   
   Serial.print(F("Cmd: "));
   Serial.println(cmd);
   
   if (cmd.length() == 1 && cmd[0] >= '0' && cmd[0] <= '4') {
     int band = cmd[0] - '0';
     freq = pgm_read_float(&eegFreqs[band]);
     Serial.print(F("✅ "));
     Serial.print((__FlashStringHelper*)pgm_read_word(&eegNames[band]));
     Serial.print(F(": "));
     Serial.print(freq);
     Serial.println(F(" Hz"));
   }
   else if (cmd.startsWith("f")) {
     float newFreq = cmd.substring(1).toFloat();
     if (newFreq > 0 && newFreq <= 100) {
       freq = newFreq;
       Serial.print(F("✅ Freq: "));
       Serial.println(freq);
     } else {
       Serial.println(F("❌ Use 0.1-100 Hz"));
     }
   }
   else if (cmd.startsWith("a")) {
     float newAmp = cmd.substring(1).toFloat();
     if (newAmp >= 0.1 && newAmp <= 100.0) {
       amp = newAmp / 1000.0;
       Serial.print(F("✅ Amp: "));
       Serial.print(newAmp);
       Serial.println(F(" mV"));
     } else {
       Serial.println(F("❌ Use 0.1-100 mV"));
     }
   }
   else if (cmd == "+") {
     amp *= 1.5;
     if (amp > 0.1) amp = 0.1;
     Serial.print(F("✅ Amp: "));
     Serial.print(amp * 1000);
     Serial.println(F(" mV"));
   }
   else if (cmd == "-") {
     amp /= 1.5;
     if (amp < 0.0001) amp = 0.0001;
     Serial.print(F("✅ Amp: "));
     Serial.print(amp * 1000);
     Serial.println(F(" mV"));
   }
   else if (cmd == "help") {
     printHelp();
   }
   else if (cmd == "test") {
     testDAC();
   }
   else {
     Serial.println(F("❌ Unknown. Type 'help'"));
   }
   
   if (cmd != "help" && cmd != "test") {
     printStatus();
   }
 }
 
 void printStatus() {
   Serial.println();
   Serial.print(F("Freq: "));
   Serial.print(freq);
   Serial.print(F(" Hz, Amp: "));
   Serial.print(amp * 1000);
   Serial.println(F(" mV"));
   Serial.println();
 }
 
 void printHelp() {
   Serial.println(F("\n=== Commands ==="));
   Serial.println(F("0: Delta (2Hz)"));
   Serial.println(F("1: Theta (6Hz)"));
   Serial.println(F("2: Alpha (10Hz)"));
   Serial.println(F("3: Beta (20Hz)"));
   Serial.println(F("4: Gamma (40Hz)"));
   Serial.println(F("f<val>: Frequency (f12.5)"));
   Serial.println(F("a<val>: Amplitude mV (a5.0)"));
   Serial.println(F("+/-: Inc/Dec amplitude"));
   Serial.println(F("test: Run DAC test"));
   Serial.println(F("================\n"));
 }
 
 void testDAC() {
   Serial.println(F("Testing DAC..."));
   uint16_t vals[] = {0, 1024, 2048, 3072, 4095};
   
   for (int i = 0; i < 5; i++) {
     Serial.print(F("Setting "));
     Serial.print(vals[i] * 10.0 / 4095.0);
     Serial.println(F("V"));
     writeDAC(vals[i]);
     delay(1000);
   }
   
   Serial.println(F("Test done. Back to sine wave."));
 }