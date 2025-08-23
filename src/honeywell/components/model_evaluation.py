import os
import pandas as pd
import logging
import joblib
import json
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


class ModelEvaluation:
    """
    Evaluates the trained model on the test set and saves metrics and plots.
    """
    def __init__(self, config):
        """
        Initializes the component with configuration.
        Args:
            config (object): Configuration object with necessary paths.
        """
        self.config = config
        # Ensure the output directory exists
        os.makedirs(self.config.root_dir, exist_ok=True)
        logging.info(f"Model evaluation output directory created at: {self.config.root_dir}")

    def evaluate(self):
        """
        Main method to load model, evaluate, and save results.
        """
        # --- 1. Load Data and Model ---
        data = pd.read_csv(self.config.data_path)
        model = joblib.load(self.config.model_path)
        logging.info("Model and data loaded successfully.")

        # --- 2. Recreate the Test Set ---
        # It's CRITICAL to use the exact same parameters as in the training step
        features = [
            'Oven Temp (C)_mean', 'Oven Temp (C)_std',
            'Mixer Speed (RPM)_mean', 'Mixer Speed (RPM)_std',
            'Mixing Time (min)_mean', 'hydration_deviation'
        ]
        target = 'Is_Anomaly'
        X = data[features]
        y = data[target]
        
        # We only need the test set for evaluation
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        logging.info("Test set recreated successfully.")

        # --- 3. Generate Predictions and Save Metrics ---
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save metrics as a JSON file for easy access later
        with open(self.config.metrics_path, 'w') as f:
            json.dump(report, f, indent=4)
        logging.info(f"Classification report saved to: {self.config.metrics_path}")

        # --- 4. Generate and Save SHAP Summary Plot ---
        # This plot is our "X-Factor" for explainability
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        plt.figure()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        # Save the plot to a file
        plt.savefig(self.config.shap_plot_path, bbox_inches='tight')
        plt.close()
        logging.info(f"SHAP summary plot saved to: {self.config.shap_plot_path}")

# For direct testing of this component
if __name__ == '__main__':
    class ConfigMock:
        data_path = "artifacts/feature_engineering/model_ready_data.csv"
        model_path = "artifacts/model_training/model.pkl"
        root_dir = "artifacts/model_evaluation"
        metrics_path = "artifacts/model_evaluation/metrics.json"
        shap_plot_path = "artifacts/model_evaluation/shap_summary_plot.png"

    try:
        logging.info(">>>>> Component: Model Evaluation started <<<<<")
        config = ConfigMock()
        model_evaluation = ModelEvaluation(config=config)
        model_evaluation.evaluate()
        logging.info(">>>>> Component: Model Evaluation finished <<<<<")
    except Exception as e:
        logging.exception(e)
        raise e