# Input inference classifier

We build an input inference classifier based on Axolotl (https://github.com/tomasreimers/axolotl) and we modified it to work with a Google Pixel 4 device by changing different settings (e.g., display resolution, ppi density, etc.). Additionally, we have developed a component for mapping the predicted coordinates into key labels.

** Dependencies **

Python 2.7.12, keras 2.3.1, sklearn 0.20.4, matplotlib 2.1.0, numpy 1.16.4, tensorflow 1.14.0

** Dataset **

We created two datasets for training our classifiers. A mock app is used for loading a webpage that calls the HTML5 functions that access motion sensors
and outputs sensors values to logcat. Additionally, apart from the accelerometer and gyroscope values, we log the coordinates (i.e., x,y) while touching the screen, which are then normalized between -1 and 1. A value of -2 is used to indicate that no touch occurred at that time. Using this setup we created two different typing datasets. One dataset contains samples created using two-handed typing, while the other contains samples created using one-handed typing. In both datasets keys were pressed randomly for one hour.

One-handed typing dataset files: ???

Two-handed typing dataset files: ???

** How to run the code: **

???

**Paper**

For technical details please refer to our publication:

This Sneaky Piggy Went to the Android Ad Market: Misusing Mobile Sensors for Stealthy Data Exfiltration
