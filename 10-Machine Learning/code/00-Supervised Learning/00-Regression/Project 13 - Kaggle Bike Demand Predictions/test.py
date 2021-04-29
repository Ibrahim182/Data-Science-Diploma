import joblib
import numpy as np 

model = joblib.load('model.h5')
scaler = joblib.load('scaler.h5')


custom_data = np.array([27, 50, 16, 2013, 5, 0, 0, 1, 0, 0, 0])

custom_data = scaler.transform([custom_data])

prediction = model.predict(custom_data)

print(f'the bikes count is {prediction[0]}')