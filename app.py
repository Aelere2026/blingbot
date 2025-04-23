from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Define your custom transformer classes
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X[((X[['x','y','z']] > 0).all(axis=1))].copy()
        X['volume'] = X['x'] * X['y'] * X['z']
        return X.reset_index(drop=True)

class VolumeOutlierRemover(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        Q1 = X['volume'].quantile(0.25)
        Q3 = X['volume'].quantile(0.75)
        self.lower_ = Q1 - 1.5*(Q3-Q1)
        self.upper_ = Q3 + 1.5*(Q3-Q1)
        return self
    
    def transform(self, X):
        return X[(X['volume'] >= self.lower_) & (X['volume'] <= self.upper_)].reset_index(drop=True)

# Initialize Flask app
app = Flask(__name__)

# Load model and preprocessing components (now that classes are defined)
try:
    model = joblib.load('diamond_rf_model.joblib')
    pipeline = joblib.load('diamond_preprocessing_pipeline.joblib')
    fe = joblib.load('feature_engineering.joblib')
    orr = joblib.load('outlier_remover.joblib')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    # Create new instances as fallback
    fe = FeatureEngineering()
    orr = VolumeOutlierRemover()
    print("Using newly instantiated transformers instead")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.json
        
        # Create DataFrame
        input_data = pd.DataFrame({
            'carat': [float(data['carat'])],
            'cut': [data['cut']],
            'color': [data['color']],
            'clarity': [data['clarity']],
            'depth': [float(data['depth'])],
            'table': [float(data['table'])],
            'x': [float(data['x'])],
            'y': [float(data['y'])],
            'z': [float(data['z'])]
        })
        
        # Apply transformations
        try:
            input_fe = fe.transform(input_data)
            input_clean = orr.transform(input_fe)
            input_prepared = pipeline.transform(input_clean)
            
            # Get prediction
            log_price_pred = model.predict(input_prepared)[0]
            price_pred = np.exp(log_price_pred)
            
            # Return prediction
            return jsonify({
                'predicted_price': round(price_pred, 2),
                'success': True
            })
        except Exception as e:
            return jsonify({
                'error': f"Prediction failed: {str(e)}",
                'success': False
            })
    
    except Exception as e:
        return jsonify({
            'error': f"Request processing failed: {str(e)}",
            'success': False
        })

if __name__ == '__main__':
    app.run(debug=True)