/*
 * ESP32C3 BLE Server for EEG Data Transmission with ADS1015
 * 
 * Hardware Setup:
 * - ESP32C3 (XIAO ESP32C3)
 * - ADS1015 (I2C address 0x48) connected to SDA/SCL pins
 * - BioAmp EXG Pill connected to ADS1015 A0
 * - EEG electrodes connected to BioAmp EXG Pill
 * 
 * This code:
 * 1. Reads EEG data from ADS1015
 * 2. Applies EEG filtering (0.5-29.5 Hz bandpass)
 * 3. Transmits filtered data via BLE notifications
 */

 #include <BLEDevice.h>
 #include <BLEServer.h>
 #include <BLEUtils.h>
 #include <BLE2902.h>
 #include <Wire.h>
 #include <Adafruit_ADS1X15.h> // Adafruit ADS1X15 library
 
 // BLE Configuration
 #define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
 #define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"
 #define ADS1015_REG_CONFIG_DR_250SPS
 
 // EEG Configuration
 #define SAMPLE_RATE 250     // Hz
 #define BAUD_RATE 115200
 #define LED_STATUS_PIN 5           // LED for status indication
 #define LED_PWR_PIN 4
 
 // ADS1015 Configuration
 Adafruit_ADS1015 ads; // ADS1015 object
 const float ADS_GAIN = 2.0 / 3.0; // ±6.144V range
 const float ADS_MAX_VOLTAGE = 6.144; // Max voltage for gain 2/3
 const int16_t ADS_MAX_VALUE = 32767; // 16-bit signed max value
 
 // Global variables
 BLEServer* pServer = NULL;
 BLECharacteristic* pCharacteristic = NULL;
 bool deviceConnected = false;
 bool oldDeviceConnected = false;
 
 float filterBuffer[6] = {0}; // Buffer for filter states
 
 class MyServerCallbacks: public BLEServerCallbacks {
     void onConnect(BLEServer* pServer) {
       deviceConnected = true;
       digitalWrite(LED_STATUS_PIN, HIGH); // LED on when connected
       Serial.println("Client connected");
     };
 
     void onDisconnect(BLEServer* pServer) {
       deviceConnected = false;
       digitalWrite(LED_STATUS_PIN, LOW); // LED off when disconnected
       Serial.println("Client disconnected");
     }
 };
 
// ---------------- Filtering utilities ----------------
struct Biquad {
  // Direct Form I
  float b0{1}, b1{0}, b2{0}, a0{1}, a1{0}, a2{0};
  float x1{0}, x2{0}, y1{0}, y2{0};
  inline float process(float x) {
    // y = (b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2)/a0
    float y = (b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
    x2 = x1; x1 = x;
    y2 = y1; y1 = y;
    return y;
  }
  inline void reset() { x1 = x2 = y1 = y2 = 0.0f; }
};

// RBJ biquad coefficient calculators
void make_notch(Biquad &q, float fs, float f0, float Q) {
  const float w0 = 2.0f * PI * f0 / fs;
  const float cosw0 = cosf(w0);
  const float sinw0 = sinf(w0);
  const float alpha = sinw0 / (2.0f * Q);
  q.b0 = 1.0f;
  q.b1 = -2.0f * cosw0;
  q.b2 = 1.0f;
  q.a0 = 1.0f + alpha;
  q.a1 = -2.0f * cosw0;
  q.a2 = 1.0f - alpha;
}

void make_lowpass(Biquad &q, float fs, float f0, float Q) {
  const float w0 = 2.0f * PI * f0 / fs;
  const float cosw0 = cosf(w0);
  const float sinw0 = sinf(w0);
  const float alpha = sinw0 / (2.0f * Q);
  const float one_minus_cos = 1.0f - cosw0;
  const float a0 = 1.0f + alpha;
  q.b0 = 0.5f * one_minus_cos;
  q.b1 = one_minus_cos;
  q.b2 = 0.5f * one_minus_cos;
  q.a0 = a0;
  q.a1 = -2.0f * cosw0;
  q.a2 = 1.0f - alpha;
}

// One-pole DC blocker (first-order high-pass)
struct DCBlocker {
  float alpha{0.9874f}; // ~0.5 Hz at fs=250
  float x1{0.0f}, y1{0.0f};
  inline float process(float x) {
    float y = alpha * (y1 + x - x1);
    x1 = x; y1 = y; return y;
  }
  inline void reset() { x1 = y1 = 0.0f; }
};

// Global filter instances
static Biquad notch50, notch60, lp1, lp2;
static DCBlocker dcblk;

#define PWR_IO 0
#define PWR_BTN 1


 
 void setup() {
   pinMode(PWR_IO, OUTPUT);
   pinMode(PWR_BTN, INPUT_PULLUP);
   digitalWrite(PWR_IO, HIGH);

   delay(100);

   Serial.begin(BAUD_RATE);
   
   pinMode(LED_STATUS_PIN, OUTPUT);
   pinMode(LED_PWR_PIN, OUTPUT);

   digitalWrite(LED_PWR_PIN, LOW); 
   digitalWrite(LED_STATUS_PIN, LOW); 
   
   delay(200);

   Serial.println("Starting EEG BLE Server...");
   Wire.setPins(7, 8);
   // Initialize I2C and ADS1015
   Wire.begin();
   if (!ads.begin(0x48)) {
     Serial.println("Failed to initialize ADS1015!");
     while (1); // Halt if ADS1015 not found
   }
   ads.setGain(GAIN_TWOTHIRDS); 
   // Ensure data rate is at least the sample rate
#ifdef ADS1015_REG_CONFIG_DR_250SPS
   ads.setDataRate(RATE_ADS1015_250SPS);
#endif
   Serial.println("ADS1015 initialized");
 
   // Create BLE Device
   BLEDevice::init("ESP32_EEG");
   
   // Create BLE Server
   pServer = BLEDevice::createServer();
   pServer->setCallbacks(new MyServerCallbacks());
 
   // Create BLE Service
   BLEService *pService = pServer->createService(SERVICE_UUID);
 
   // Create BLE Characteristic
   pCharacteristic = pService->createCharacteristic(
                       CHARACTERISTIC_UUID,
                       BLECharacteristic::PROPERTY_READ   |
                       BLECharacteristic::PROPERTY_WRITE  |
                       BLECharacteristic::PROPERTY_NOTIFY |
                       BLECharacteristic::PROPERTY_INDICATE
                     );
 
   // Add descriptor for notifications
   pCharacteristic->addDescriptor(new BLE2902());
 
   // Start the service
   pService->start();
 
   // Start advertising
   BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
   pAdvertising->addServiceUUID(SERVICE_UUID);
   pAdvertising->setScanResponse(false);
   pAdvertising->setMinPreferred(0x0);
   BLEDevice::startAdvertising();
   
   Serial.println("EEG BLE Server ready!");
   Serial.println("Waiting for client connection...");

   delay(200);
   digitalWrite(LED_PWR_PIN, HIGH); 

   // Initialize filters
   // DC blocker already has alpha for ~0.5 Hz at 250 Hz
   make_notch(notch50, SAMPLE_RATE, 50.0f, 30.0f);
   make_notch(notch60, SAMPLE_RATE, 60.0f, 30.0f);
   // 4th-order low-pass at 45 Hz via two cascaded biquads with Butterworth-equivalent Qs
   make_lowpass(lp1, SAMPLE_RATE, 45.0f, 0.5412f);
   make_lowpass(lp2, SAMPLE_RATE, 45.0f, 1.3065f);
 }
 
 void loop() {
   // Timing control for consistent sampling rate
   static unsigned long lastSample = 0;
   unsigned long currentTime = micros();
   unsigned long sampleInterval = 1000000 / SAMPLE_RATE; // microseconds
   
   if (currentTime - lastSample >= sampleInterval) {
     lastSample = currentTime;
     
     // Read EEG data from ADS1015 (single-ended, channel 3)
     int16_t rawValue = ads.readADC_SingleEnded(3);
     // Use library scaling for correct volts; avoids hardcoded 5V assumption
     float voltage = ads.computeVolts(rawValue);

     // Filtering pipeline: DC block -> 50 Hz notch -> 60 Hz notch -> 4th-order LP @45 Hz
     float x = dcblk.process(voltage);
     x = notch50.process(x);
     x = notch60.process(x);
     x = lp1.process(x);
     float filteredEEG = lp2.process(x);
     //float filteredEEG = offsetAdjustedVoltage;
     
     // Send data via BLE if connected
     if (deviceConnected) {
       // Convert to string and send
       String dataString = String(filteredEEG, 6); // 6 decimal places
       pCharacteristic->setValue(dataString.c_str());
       pCharacteristic->notify();
       
       // Debug output every 50 samples
       static int sampleCount = 0;
       if (++sampleCount >= 50) {
         Serial.printf("EEG: %.6f V (Raw: %d)\n", filteredEEG, rawValue);
         sampleCount = 0;
       }
     }
   }
   
   // Handle BLE connection changes
   if (!deviceConnected && oldDeviceConnected) {
     delay(500); // Give time for BLE stack to prepare
     pServer->startAdvertising(); // Restart advertising
     Serial.println("Restarting advertising...");
     oldDeviceConnected = deviceConnected;
   }
   
   if (deviceConnected && !oldDeviceConnected) {
     oldDeviceConnected = deviceConnected;
   }
 }