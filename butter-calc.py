from scipy import signal

fs = 250
lowcut = 0.5
highcut = 90.0
order = 2  # 2 here -> overall 4th-order bandpass

b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=fs)
print("V2: Numerator  (b):", b)
print("V2: Denominator (a):", a)