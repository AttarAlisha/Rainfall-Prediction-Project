from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('rainfall_model.pkl', 'rb'))

# Create a Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        pressure = float(request.form['pressure'])

        # Make prediction
        features = np.array([[temperature, humidity, pressure]])
        prediction = model.predict(features)[0]
        prediction = max(0, prediction)  # Ensure non-negative prediction

        # Render the result on the webpage with form values pre-filled
        return render_template('index.html', 
                               prediction_text=f'Predicted Rainfall: {prediction:.2f} mm',
                               temperature=temperature,
                               humidity=humidity,
                               pressure=pressure)

    except ValueError:
        return render_template('index.html', 
                               prediction_text='Invalid Input. Please enter valid numbers.',
                               temperature=request.form.get('temperature', ''),
                               humidity=request.form.get('humidity', ''),
                               pressure=request.form.get('pressure', ''))

if __name__ == '__main__':
    app.run(debug=True)
