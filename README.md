# Diamond Price Prediction Web Application

This web application predicts diamond prices based on a Random Forest regression model trained on diamond characteristics. The application provides an interface where users can input diamond attributes and receive an estimated price.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Information](#model-information)
- [Technologies Used](#technologies-used)

## Features
- Intuitive web interface for diamond price prediction
- High accuracy predictions with < 6% mean absolute percentage error
- Supports all standard diamond attributes (carat, cut, color, clarity, dimensions)
- Responsive design for both desktop and mobile use

## Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup
1. Clone the repository:
```bash
git clone https://github.com/Aelere2026/blingbot.git
cd blingbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Generate the model files:
**IMPORTANT:** Due to GitHub size constraints, the trained model files are not included in the repository. You must generate them by running the preprocessing and training notebook.

```bash
jupyter notebook preprocess_train.ipynb
```

Execute all cells in the notebook to:
- Load and preprocess the diamond dataset
- Train the Random Forest model
- Save the required model files:
  - diamond_rf_model.joblib
  - diamond_preprocessing_pipeline.joblib
  - feature_engineering.joblib
  - outlier_remover.joblib

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

3. Enter diamond characteristics in the web interface:
   - Carat weight
   - Cut quality (Fair, Good, Very Good, Premium, Ideal)
   - Color grade (D-J)
   - Clarity grade (IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1)
   - Depth percentage
   - Table percentage
   - Dimensions (x, y, z in mm)

4. Click "Get Price Estimate" to receive the predicted diamond price

## Project Structure
```
blingbot/
├── app.py                        # Flask web application
├── templates/                    # HTML templates
│   └── index.html                # Main application interface
├── preprocess_train.ipynb        # Notebook for data preprocessing and model training
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Model Information

### Data Preprocessing
- Feature engineering to create volume measurement
- Log transformation of numerical features
- Outlier removal using IQR method
- Ordinal encoding of categorical features (cut, color, clarity)

### Model Details
- Algorithm: Random Forest Regression
- Hyperparameters:
  - n_estimators: 200
  - max_depth: 20
  - min_samples_split: 2
  - min_samples_leaf: 1
- Performance:
  - Validation MAPE: 5.84%
  - Validation R²: 0.9899

## Technologies Used
- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Joblib
- HTML/CSS/JavaScript
- Bootstrap
