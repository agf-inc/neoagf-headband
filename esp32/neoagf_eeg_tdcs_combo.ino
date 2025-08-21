/*
 * neoagf_eeg_tdcs_combo.ino
 * Combined ESP32C3 sketch: BLE EEG streaming + tDCS (GP8302) current control
 *
 * - BLE EEG notifications at 250 Hz from ADS1015 (channel 3)
 * - BLE Control characteristic for mode and tDCS current control
 * - LEDs: PWR (BT status), MODE (operation mode)
 *   - PWR LED blinks while advertising, steady ON when connected
 *   - MODE LED steady in EEG mode, slow pulse in STIM mode
 */

 #include <Arduino.h>
 #include <Wire.h>
 #include <Adafruit_ADS1X15.h>
 #include <DFRobot_GP8302.h>
 #include <BLEDevice.h>
 #include <BLEServer.h>
 #include <BLEUtils.h>
 #include <BLE2902.h>
 #include <nvs_flash.h>
 
 // ===== Pins =====
 #define PWR_IO 0         // Power rail control (optional)
 #define PWR_BTN 1        // Power button (INPUT_PULLUP)
 #define LED_PWR_PIN 5    // Power + Bluetooth status LED (matches original LED_STATUS_PIN)
 #define LED_MODE_PIN 4   // Mode LED (EEG vs STIM)
 
 // ===== I2C (shared) =====
 #define I2C_SDA_PIN 7
 #define I2C_SCL_PIN 8
 
 // ===== BLE UUIDs =====
#define SERVICE_UUID            "f47ac10b-58cc-4372-a567-0e02b2c3d479"
#define EEG_CHAR_UUID           "f47ac10b-58cc-4372-a567-0e02b2c3d480"
#define CONTROL_CHAR_UUID       "f47ac10b-58cc-4372-a567-0e02b2c3d481"
 #define BLE_ADVERTISING_NAME    "NEOAGF"
 
 // ===== EEG Config =====
 #define SAMPLE_RATE 250
 Adafruit_ADS1015 ads;
 
 // ===== tDCS (GP8302) =====
 DFRobot_GP8302 tdcs;
 static float targetCurrent_mA = 0.0f;
 static float currentCurrent_mA = 0.0f;
 static float rampStep_mA = 0.1f;               // default increment/decrement
 static const float kMaxCurrent_mA = 25.0f;
 static const float kMinCurrent_mA = 0.0f;
 static const unsigned long kRampIntervalMs = 1000UL; // every 1s
 static unsigned long lastRampMs = 0;
 
 // ===== BLE Globals =====
 BLEServer* pServer = nullptr;
 BLECharacteristic* pEEGChar = nullptr;
 BLECharacteristic* pCtrlChar = nullptr;
 bool deviceConnected = false;
 bool oldDeviceConnected = false;
 
 // ===== Modes =====
 enum class OpMode { NO_OP = 0, EEG = 1, STIM = 2 };
 static OpMode mode = OpMode::NO_OP;
static bool adsReady = false;
static bool tdcsReady = false;
 
 // ===== Filtering utilities (from neoagf_bt_eeg_reader.ino) =====
 struct Biquad {
   float b0{1}, b1{0}, b2{0}, a0{1}, a1{0}, a2{0};
   float x1{0}, x2{0}, y1{0}, y2{0};
   inline float process(float x) {
     float y = (b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
     x2 = x1; x1 = x;
     y2 = y1; y1 = y;
     return y;
   }
   inline void reset() { x1 = x2 = y1 = y2 = 0.0f; }
 };
 
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
 
 struct DCBlocker {
   float alpha{0.9874f}; // ~0.5 Hz at fs=250
   float x1{0.0f}, y1{0.0f};
   inline float process(float x) {
     float y = alpha * (y1 + x - x1);
     x1 = x; y1 = y; return y;
   }
   inline void reset() { x1 = y1 = 0.0f; }
 };
 
 static Biquad notch50, notch60, lp1, lp2;
static DCBlocker dcblk;
 
 // ===== LED timing =====
 static unsigned long lastBTBlinkMs = 0;
 static bool btBlinkState = false;
 static unsigned long lastModePulseMs = 0;
 static bool modePulseState = false;
 
 // ===== Staged initialization =====
 enum InitStage {
   INIT_PINS = 0,
   INIT_I2C,
   INIT_ADS1015,
   INIT_FILTERS,
   INIT_TDCS,
   INIT_BLE,
   INIT_COMPLETE
 };
 static volatile bool systemReady = false;
static InitStage initStage = INIT_BLE; // Only bring up BLE at boot; peripherals init on mode switch
 
 // ===== BLE callbacks =====
 class ServerCallbacks : public BLEServerCallbacks {
   void onConnect(BLEServer* s) override {
     deviceConnected = true;
     Serial.println("Client connected");
   }
   void onDisconnect(BLEServer* s) override {
     deviceConnected = false;
     Serial.println("Client disconnected");
   }
 };

static float clampf(float v, float lo, float hi) { return (v < lo) ? lo : (v > hi) ? hi : v; }

// ===== Peripheral init/deinit helpers =====
static bool initADS() {
  Serial.println("[EEG] Initializing ADS1015 path...");
  // Ensure I2C pins and bus are configured before sensor init
  Wire.setPins(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.begin();
  if (!ads.begin(0x48)) {
    Serial.println("[EEG] ADS1015 not found @0x48");
    adsReady = false;
    return false;
  }
  ads.setDataRate(RATE_ADS1015_250SPS);
  ads.setGain(GAIN_TWOTHIRDS);
  dcblk.reset();
  make_notch(notch50, SAMPLE_RATE, 50.0f, 30.0f);
  make_notch(notch60, SAMPLE_RATE, 60.0f, 30.0f);
  make_lowpass(lp1, SAMPLE_RATE, 45.0f, 0.5412f);
  make_lowpass(lp2, SAMPLE_RATE, 45.0f, 1.3065f);
  adsReady = true;
  Serial.println("[EEG] ADS1015 ready (250 SPS, GAIN 2/3)");
  return true;
}

static void deinitADS() {
  if (adsReady) {
    Serial.println("[EEG] Deinitializing ADS1015 path...");
  }
  // No explicit deinit in Adafruit lib; ensure flags off
  adsReady = false;
}

static bool initTDCS() {
  Serial.println("[STIM] Initializing GP8302 path...");
  // Ensure I2C pins stay correct before GP8302 init
  Wire.setPins(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.begin();
  uint8_t st = tdcs.begin(I2C_SCL_PIN, I2C_SDA_PIN);
  if (st != 0) {
    Serial.println("[STIM] GP8302 not ready");
    tdcsReady = false;
    return false;
  }
  tdcs.output(0);
  currentCurrent_mA = 0.0f;
  tdcsReady = true;
  Serial.println("[STIM] GP8302 ready, output=0.0 mA");
  return true;
}

static void deinitTDCS() {
  if (tdcsReady) {
    Serial.println("[STIM] Shutting down GP8302 output");
    tdcs.output(0);
  }
  tdcsReady = false;
}

static void setMode(OpMode m) {
  if (m == mode) {
    Serial.println("[MODE] No change");
    return;
  }
  Serial.print("[MODE] Switching to ");
  Serial.println(m == OpMode::STIM ? "STIM" : (m == OpMode::EEG ? "EEG" : "NO_OP"));

  // Tear down current peripherals first
  if (mode == OpMode::EEG) {
    deinitADS();
  } else if (mode == OpMode::STIM) {
    // Stop stimulation cleanly
    targetCurrent_mA = 0.0f;
    deinitTDCS();
  }

  // Apply new mode and bring up corresponding peripherals
  mode = m;
  if (mode == OpMode::EEG) {
    targetCurrent_mA = 0.0f; // ensure stim target zeroed
    initADS();
  } else if (mode == OpMode::STIM) {
    initTDCS();
  } else { // NO_OP
    targetCurrent_mA = 0.0f;
  }
}
 
 static void handleCommand(const String &cmdRaw) {
   // Normalize: remove non-printables, trim, uppercase for keyword checks
   String clean;
   clean.reserve(cmdRaw.length());
   for (size_t i = 0; i < cmdRaw.length(); ++i) {
     char c = cmdRaw[i];
     if (c >= 32 && c <= 126) clean += c; // printable ASCII
   }
   clean.trim();
   if (clean.length() == 0) return;
 
   // Build an upper-case copy for comparisons (values still parsed from original where needed)
   String up = clean;
   up.toUpperCase();
 
   // MODE handling (accept spaces and case-insensitive)
   if (up.startsWith("MODE")) {
     if (up.indexOf("STIM") >= 0) {
       setMode(OpMode::STIM);
       pCtrlChar->setValue("OK MODE STIM");
       pCtrlChar->notify();
       return;
     } else if (up.indexOf("EEG") >= 0) {
       setMode(OpMode::EEG);
       pCtrlChar->setValue("OK MODE EEG");
       pCtrlChar->notify();
       return;
     } else if (up.indexOf("NO_OP") >= 0) {
       setMode(OpMode::NO_OP);
       pCtrlChar->setValue("OK MODE NO_OP");
       pCtrlChar->notify();
       return;
     } else {
       pCtrlChar->setValue("ERR MODE?");
       pCtrlChar->notify();
       return;
     }
   }
 
   // Current nudges
   if (up == "I+") {
     targetCurrent_mA = clampf(targetCurrent_mA + rampStep_mA, kMinCurrent_mA, kMaxCurrent_mA);
     pCtrlChar->setValue((String("OK I=") + String(targetCurrent_mA, 1)).c_str());
     pCtrlChar->notify();
     return;
   }
   if (up == "I-") {
     targetCurrent_mA = clampf(targetCurrent_mA - rampStep_mA, kMinCurrent_mA, kMaxCurrent_mA);
     pCtrlChar->setValue((String("OK I=") + String(targetCurrent_mA, 1)).c_str());
     pCtrlChar->notify();
     return;
   }
 
   // Absolute current set (case-insensitive, tolerate spaces)
   if (up.startsWith("I=")) {
     // Parse numeric from the clean string after the first '='
     int eq = clean.indexOf('=');
     float v = (eq >= 0) ? clean.substring(eq + 1).toFloat() : NAN;
     if (!isnan(v)) {
       targetCurrent_mA = clampf(v, kMinCurrent_mA, kMaxCurrent_mA);
       pCtrlChar->setValue((String("OK I=") + String(targetCurrent_mA, 1)).c_str());
       pCtrlChar->notify();
       return;
     }
   }
 
   // Step set
   if (up.startsWith("STEP=")) {
     int eq = clean.indexOf('=');
     float v = (eq >= 0) ? clean.substring(eq + 1).toFloat() : 0.0f;
     if (v <= 0) v = 0.1f;
     rampStep_mA = clampf(v, 0.1f, 5.0f);
     pCtrlChar->setValue((String("OK STEP=") + String(rampStep_mA, 2)).c_str());
     pCtrlChar->notify();
     return;
   }
 
   // Status (accept STATUS and STATUS?)
   if (up == "STATUS" || up == "STATUS?") {
     String s = String("{");
     s += String("\"bt\":\"") + (deviceConnected ? "connected" : "advertising") + "\",";
     s += String("\"mode\":\"") + (mode == OpMode::STIM ? "STIM" : (mode == OpMode::EEG ? "EEG" : "NO_OP")) + "\",";
     s += String("\"I\":") + String(currentCurrent_mA, 2) + ",";
     s += String("\"target\":") + String(targetCurrent_mA, 2) + "}";
     pCtrlChar->setValue(s.c_str());
     pCtrlChar->notify();
     return;
   }

   // Ignore echoes or non-commands that look like JSON or already formatted responses
   if (clean.length() > 0) {
     if (clean[0] == '{' || clean.startsWith("OK ") || clean.startsWith("ERR ")) {
       return;
     }
   }

   pCtrlChar->setValue("ERR UNKNOWN");
   pCtrlChar->notify();
 }
 
 class ControlCallbacks : public BLECharacteristicCallbacks {
   void onWrite(BLECharacteristic* c) override {
     // Be robust across BLE libraries: getValue may return std::string or Arduino String.
     // Convert to Arduino String via c_str() to avoid type issues.
     String v = String(c->getValue().c_str());
     if (v.length() > 0) {
       Serial.print("[BLE CMD] ");
       Serial.println(v);
       handleCommand(v);
     }
   }
   void onRead(BLECharacteristic* c) override {
     // Always return current STATUS snapshot on read
     String s = String("{");
     s += String("\"bt\":\"") + (deviceConnected ? "connected" : "advertising") + "\",";
     s += String("\"mode\":\"") + (mode == OpMode::STIM ? "STIM" : (mode == OpMode::EEG ? "EEG" : "NO_OP")) + "\",";
     s += String("\"I\":") + String(currentCurrent_mA, 2) + ",";
     s += String("\"target\":") + String(targetCurrent_mA, 2) + "}";
     c->setValue(s.c_str());
   }
 };
 
 // ===== Setup =====
 void setup() {
   Serial.begin(115200);
 
   delay(100);
   pinMode(PWR_IO, OUTPUT);
   pinMode(PWR_BTN, INPUT_PULLUP);
   digitalWrite(PWR_IO, HIGH);
 
   pinMode(LED_PWR_PIN, OUTPUT);
   pinMode(LED_MODE_PIN, OUTPUT);
   digitalWrite(LED_PWR_PIN, HIGH);
   digitalWrite(LED_MODE_PIN, LOW);

   nvs_flash_erase();
   nvs_flash_init();
 
   Wire.setPins(I2C_SDA_PIN, I2C_SCL_PIN);
   Wire.begin();
 
   Serial.println("\n=== Booting neoagf_eeg_tdcs_combo ===");
   Serial.println("[SER] Commands: DISC (disconnect BLE), RESET (restart MCU)");
 
   // End of Setup
 }
 
 // Handle staged initialization
 static void handleInitialization() {
  static unsigned long lastInitMs = 0;
  unsigned long nowMs = millis();

  // Rate-limit init steps to avoid busy looping
  if (lastInitMs != 0 && (nowMs - lastInitMs) < 50) {
    return;
  }
  lastInitMs = nowMs;

  switch (initStage) {
    case INIT_BLE:
      Serial.println("[INIT] Starting BLE...");
      BLEDevice::init(BLE_ADVERTISING_NAME);
      Serial.println("[INIT] Creating BLE Server");
      pServer = BLEDevice::createServer();
      Serial.println("[INIT] BLE Server created");
      pServer->setCallbacks(new ServerCallbacks());
      delay(100);
      Serial.println("[INIT] Creating BLE Services and characteristics");
      {
        BLEService* svc = pServer->createService(SERVICE_UUID);
        pEEGChar = svc->createCharacteristic(
            EEG_CHAR_UUID,
            BLECharacteristic::PROPERTY_READ |
            BLECharacteristic::PROPERTY_NOTIFY |
            BLECharacteristic::PROPERTY_INDICATE);
        pEEGChar->addDescriptor(new BLE2902());

        pCtrlChar = svc->createCharacteristic(
            CONTROL_CHAR_UUID,
            BLECharacteristic::PROPERTY_READ |
            BLECharacteristic::PROPERTY_WRITE |
            BLECharacteristic::PROPERTY_NOTIFY);
        pCtrlChar->setCallbacks(new ControlCallbacks());
        pCtrlChar->addDescriptor(new BLE2902());

        svc->start();
      }
      delay(50);
      Serial.println("[INIT] Starting to advertise...");
      {
        BLEAdvertising *adv = BLEDevice::getAdvertising();
        adv->addServiceUUID(SERVICE_UUID);
        adv->setScanResponse(false);
        adv->setMinPreferred(0x0);
        BLEDevice::startAdvertising();
      }

      Serial.print("[BLE] Service started, advertising as ");
      Serial.println(BLE_ADVERTISING_NAME);
      btBlinkState = true;
      lastBTBlinkMs = millis();
      initStage = INIT_COMPLETE;
      break;

    case INIT_COMPLETE:
      Serial.println("[INIT] System ready!");
      systemReady = true;
      break;
  }

  Serial.printf("Free heap after init step %d: %d bytes\n", initStage, ESP.getFreeHeap());
}
 
 // ===== Loop =====
 void loop() {
   // Handle staged initialization first
   if (!systemReady) {
     handleInitialization();
     return; // Don't run main loop until init complete
   }
 
   unsigned long nowMs = millis();
 
   // 0) Serial command handler (non-blocking)
   static String serBuf;
   while (Serial.available() > 0) {
     char ch = (char)Serial.read();
     if (ch == '\r' || ch == '\n') {
       serBuf.trim();
       if (serBuf.length() > 0) {
         if (serBuf.equalsIgnoreCase("DISC")) {
           Serial.println("[SER] Disconnecting BLE and stopping advertising...");
           // De-initialize BLE stack: drops connections and releases memory
           BLEDevice::deinit(true);
           deviceConnected = false;
           oldDeviceConnected = false;
           digitalWrite(LED_PWR_PIN, LOW);
         } else if (serBuf.equalsIgnoreCase("RESET")) {
           Serial.println("[SER] Restarting MCU...");
           delay(100);
           ESP.restart();
         } else {
           Serial.print("[SER] Unknown cmd: "); Serial.println(serBuf);
         }
       }
       serBuf = ""; // clear buffer
     } else {
       // Basic line buffer, guard size
       if (serBuf.length() < 64) serBuf += ch;
     }
   }

   // 1) EEG sampling at 250 Hz (only when adsReady)
   if (mode == OpMode::EEG) {
     if (!adsReady) {
       // Lazy init if not ready yet
       initADS();
     } else {
       static unsigned long lastSample = 0;
       unsigned long currentTime = micros();
       unsigned long sampleInterval = 1000000UL / SAMPLE_RATE;
       if (currentTime - lastSample >= sampleInterval) {
         lastSample = currentTime;

         int16_t raw = ads.readADC_SingleEnded(3);
         float v = ads.computeVolts(raw);
         float x = dcblk.process(v);
         x = notch50.process(x);
         x = notch60.process(x);
         x = lp1.process(x);
         float filtered = lp2.process(x);         

         if (deviceConnected && pEEGChar) {
           String s = String(filtered, 6);
           pEEGChar->setValue(s.c_str());
           pEEGChar->notify();
         }
         static int eegSampleCount = 0;
         if (++eegSampleCount >= 50) {
           eegSampleCount = 0;
           Serial.print("[EEG] V=");
           Serial.println(filtered, 6);
         }
       }
     }
   }

   // 2) BLE connection transitions
   if (!deviceConnected && oldDeviceConnected) {
     delay(500);
     pServer->startAdvertising();
     oldDeviceConnected = deviceConnected;
     Serial.println("[BLE] Restarting advertising...");
   }
   if (deviceConnected && !oldDeviceConnected) {
     oldDeviceConnected = deviceConnected;
     Serial.println("[BLE] Connected");
   }

   // 3) tDCS ramp logic (1 Hz) when in STIM and tdcsReady
   if (mode == OpMode::STIM) {
     if (!tdcsReady) {
       initTDCS();
     } else if ((nowMs - lastRampMs >= kRampIntervalMs) || lastRampMs == 0) {
       lastRampMs = nowMs;
       if (fabsf(targetCurrent_mA - currentCurrent_mA) >= (rampStep_mA - 1e-6f)) {
         if (currentCurrent_mA < targetCurrent_mA) {
           currentCurrent_mA += rampStep_mA;
           if (currentCurrent_mA > targetCurrent_mA) currentCurrent_mA = targetCurrent_mA;
         } else if (currentCurrent_mA > targetCurrent_mA) {
           currentCurrent_mA -= rampStep_mA;
           if (currentCurrent_mA < targetCurrent_mA) currentCurrent_mA = targetCurrent_mA;
         }
       }
       if (tdcsReady) {
         tdcs.output(currentCurrent_mA);
       }
       // Print ramp status once per second
       Serial.print("[tDCS] I=");
       Serial.print(currentCurrent_mA, 2);
       Serial.print(" mA target=");
       Serial.print(targetCurrent_mA, 2);
       Serial.print(" mode=");
       Serial.println(mode == OpMode::STIM ? "STIM" : (mode == OpMode::EEG ? "EEG" : "NO_OP"));
     }
   }

  // 4) LEDs
  // PWR LED: blink at ~1 Hz when not connected; steady ON when connected
  if (!deviceConnected) {
    if (nowMs - lastBTBlinkMs >= 500) { // 1 Hz (toggle every 500ms)
      lastBTBlinkMs = nowMs;
      btBlinkState = !btBlinkState;
      digitalWrite(LED_PWR_PIN, btBlinkState ? HIGH : LOW);
    }
  } else {
    digitalWrite(LED_PWR_PIN, HIGH);
  }

  // MODE LED: steady ON in EEG; slow pulse (1 Hz blink) in STIM
  if (mode == OpMode::EEG) {
    digitalWrite(LED_MODE_PIN, HIGH);
  } else if (mode == OpMode::STIM) { // STIM
    if ((nowMs - lastModePulseMs >= 500) || (lastModePulseMs == 0)) { // 1 Hz blink
      lastModePulseMs = nowMs;
      modePulseState = !modePulseState;
      digitalWrite(LED_MODE_PIN, modePulseState ? HIGH : LOW);
    }
  } else {
    digitalWrite(LED_MODE_PIN, LOW);
  }

  // Optional: simple button scaffold (no action bound yet)
  // int btn = digitalRead(PWR_BTN);
  // (future) short/long press actions
}
 