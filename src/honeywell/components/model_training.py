import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

class ModelTrainer:
    """
    Trains an XGBoost classifier on the engineered features and saves the model.
    """
    def __init__(self, config):
        """
        Initializes the component with configuration.
        Args:
            config (object): Configuration object with data path and model path.
        """
        self.config = config
        # Ensure the output directory exists
        output_dir = os.path.dirname(self.config.model_path)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Model training output directory created at: {output_dir}")

    def train(self):
        """
        Main method to load data, train the XGBoost model, and save it.
        """
        # --- 1. Load Data ---
        try:
            data = pd.read_csv(self.config.data_path)
            logging.info("Model-ready data loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Error loading data file: {e}. Make sure the feature engineering step was run.")
            raise

        # --- 2. Prepare Data for Training ---
        features = [
            'Oven Temp (C)_mean', 'Oven Temp (C)_std',
            'Mixer Speed (RPM)_mean', 'Mixer Speed (RPM)_std',
            'Mixing Time (min)_mean', 'hydration_deviation'
        ]
        target = 'Is_Anomaly'

        X = data[features]
        y = data[target]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        logging.info("Data split into training and testing sets.")

        # --- 3. Train the XGBoost Model ---
        model = XGBClassifier(eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train)
        logging.info("XGBoost model training complete.")

        # --- 4. Save the Trained Model ---
        joblib.dump(model, self.config.model_path)
        logging.info(f"Model saved successfully to: {self.config.model_path}")

# For direct testing of this component
if __name__ == '__main__':
    class ConfigMock:
        data_path = "artifacts/feature_engineering/model_ready_data.csv"
        model_path = "artifacts/model_training/model.pkl"

    try:
        logging.info(">>>>> Component: Model Training (XGBoost) started <<<<<")
        config = ConfigMock()
        model_trainer = ModelTrainer(config=config)
        model_trainer.train()
        logging.info(">>>>> Component: Model Training (XGBoost) finished <<<<<")
    except Exception as e:
        logging.exception(e)
        raise e