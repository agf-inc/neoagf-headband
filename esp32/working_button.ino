#define PWR_IO 0
#define PWR_BTN 1
#define LED1 4
#define LED2 5

#define LONG_PRESS_DURATION 2000 // 2 seconds in milliseconds

void setup() {
  pinMode(PWR_IO, OUTPUT);
  pinMode(PWR_BTN, INPUT_PULLUP); // Assuming button pulls to ground when pressed
  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  
  digitalWrite(PWR_IO, HIGH); // Set PWR_IO HIGH on startup
  digitalWrite(LED1, LOW);    // Initialize LED1 OFF
  digitalWrite(LED2, LOW);    // Initialize LED2 OFF
}

void loop() {
  static unsigned long pressStartTime = 0;
  static bool buttonWasPressed = false;

  // Check if button is pressed (LOW because of INPUT_PULLUP)
  if (digitalRead(PWR_BTN) == LOW) {
    if (!buttonWasPressed) {
      // Button just pressed
      buttonWasPressed = true;
      pressStartTime = millis();
      digitalWrite(LED1, HIGH); // Turn on LED1 when button is pressed
    }
    
    // Check for long press
    if (millis() - pressStartTime >= LONG_PRESS_DURATION) {
      digitalWrite(PWR_IO, LOW); // Set PWR_IO LOW after 2 seconds
    }
  } else {
    // Button released
    if (buttonWasPressed) {
      buttonWasPressed = false;
      digitalWrite(LED1, LOW); // Turn off LED1 when button is released
    }
  }
}