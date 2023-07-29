import RPi.GPIO as GPIO
import time

# Setup for the GPIO
GPIO.setmode(GPIO.BCM)  # Use Broadcom chip's pin numbers
GPIO.setup(18, GPIO.OUT)  # Set pin 18 as an output pin

# Set the PWM (Pulse Width Modulation)
pwm = GPIO.PWM(18, 50)  # 50Hz frequency

# Start the PWM with a 0% duty cycle (off)
pwm.start(0)

try:
    while True:
        # Move the servo back and forth
        pwm.ChangeDutyCycle(2.5)  # 0 degree
        time.sleep(1)
        pwm.ChangeDutyCycle(7.5)  # 90 degree
        time.sleep(1)
        pwm.ChangeDutyCycle(12.5)  # 180 degree
        time.sleep(1)
except KeyboardInterrupt:
    # If a keyboard interrupt is detected then cleanup and stop
    pwm.stop()
    GPIO.cleanup()
