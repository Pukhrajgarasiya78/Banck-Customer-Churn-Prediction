from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your saved model (replace with your actual model path)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Helper function to encode categorical variables
def encode_features(data):
    geography_map = {'France': 0, 'Germany': 1, 'Spain': 2}
    gender_map = {'Male': 0, 'Female': 1}
    
    data['Geography'] = geography_map.get(data['Geography'], -1)
    data['Gender'] = gender_map.get(data['Gender'], -1)
    
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    geography = request.form['geography']
    gender = request.form['gender']
    age = int(request.form['age'])
    balance = float(request.form['balance'])
    num_of_products = int(request.form['num_of_products'])
    is_active_member = int(request.form['is_active_member'])
    complain = int(request.form['complain'])

    # Prepare data for the model
    input_data = {
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'IsActiveMember': is_active_member,
        'Complain': complain
    }
    
    # Encode the data
    encoded_data = encode_features(input_data)
    
    # Prepare features for model input
    features = [
        encoded_data['Geography'],
        encoded_data['Gender'],
        encoded_data['Age'],
        encoded_data['Balance'],
        encoded_data['NumOfProducts'],
        encoded_data['IsActiveMember'],
        encoded_data['Complain']
    ]
    
    input_array = np.array(features).reshape(1, -1)
    
    # Predict using the loaded model
    try:
        prediction = model.predict(input_array)
        prediction_result = "Churn" if int(prediction[0]) == 1 else "No Churn"
        return render_template('index.html', prediction=prediction_result)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
