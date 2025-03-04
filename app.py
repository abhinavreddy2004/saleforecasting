from flask import Flask, request, jsonify
import pandas as pd
import os
from prophet import Prophet

app = Flask(__name__)

import pandas as pd

dataset_url = "https://your-cloud-link.com/Walmart.csv"
df = pd.read_csv(dataset_url)

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df_filtered = df[['Date', 'Weekly_Sales']].rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})

# Train Prophet model (only once)
model = Prophet()
model.fit(df_filtered)


@app.route('/')
def home():
    return """ 
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sales Forecasting API</title>
    </head>
    <body>
        <h1>Welcome to the Sales Forecasting API</h1>
        <p>Use the following endpoint to get predictions:</p>
        <p><strong><a href='/predict?weeks=12'>Click Here</a></strong></p>
    </body>
    </html>
    """


@app.route('/predict', methods=['GET'])
def predict_sales():
    try:
        weeks = int(request.args.get('weeks', 12))  # Default: 12 weeks
        
        # Create future dates
        future = model.make_future_dataframe(periods=weeks, freq='W')
        forecast = model.predict(future)
        
        # Get required columns
        result = forecast[['ds', 'yhat']].tail(weeks)
        result = result.rename(columns={'ds': 'Date', 'yhat': 'Predicted_Sales'})
        
        return jsonify(result.to_dict(orient='records'))
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
