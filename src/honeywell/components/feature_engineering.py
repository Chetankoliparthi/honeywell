import os
import pandas as pd
import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


class FeatureEngineering:
    """
    Transforms raw time-series process data into a model-ready feature set.
    Each row in the output represents one batch with its aggregated features.
    """
    def __init__(self, config):
        """
        Initializes the component with configuration.
        Args:
            config (object): Configuration object with input and output paths.
        """
        self.config = config
        # Ensure the output directory exists
        output_dir = os.path.dirname(self.config.output_data_path)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Feature engineering output directory created at: {output_dir}")

    def engineer_features(self):
        """
        Main method to load, process, and save the engineered features.
        """
        # --- 1. Load Data ---
        try:
            process_df = pd.read_csv(self.config.process_data_path)
            quality_df = pd.read_csv(self.config.quality_data_path)
            logging.info("Raw data loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Error loading data files: {e}. Make sure the data generation step was run.")
            raise

        # --- 2. Aggregate Time-Series Data ---
        # We calculate the mean and standard deviation (stability) for key parameters
        aggregations = {
            'Oven Temp (C)': ['mean', 'std'],
            'Mixer Speed (RPM)': ['mean', 'std'],
            'Mixing Time (min)': ['mean'],
            'Flour Protein (%)': ['mean']
        }
        
        batch_features = process_df.groupby('Batch_ID').agg(aggregations)
        # Flatten the multi-level column names (e.g., ('Oven Temp (C)', 'mean') -> 'Oven Temp (C)_mean')
        batch_features.columns = ['_'.join(col).strip() for col in batch_features.columns.values]
        logging.info("Time-series data aggregated per batch.")

        # --- 3. Engineer the "Golden" Feature: Hydration Deviation ---
        # Get the mean water and flour amounts for each batch to calculate actual hydration
        water_mean = process_df.groupby('Batch_ID')['Water (kg)'].mean()
        flour_mean = process_df.groupby('Batch_ID')['Flour (kg)'].mean()
        
        # Merge these back to the main features dataframe
        batch_features = batch_features.merge(water_mean.rename('Water (kg)_mean'), on='Batch_ID')
        batch_features = batch_features.merge(flour_mean.rename('Flour (kg)_mean'), on='Batch_ID')

        batch_features['actual_hydration'] = (batch_features['Water (kg)_mean'] / batch_features['Flour (kg)_mean']) * 100
        batch_features['ideal_hydration'] = batch_features['Flour Protein (%)_mean'] * 5
        batch_features['hydration_deviation'] = batch_features['actual_hydration'] - batch_features['ideal_hydration']
        logging.info("Custom feature 'hydration_deviation' engineered.")

        # --- 4. Combine Features with Quality Labels ---
        # Merge the engineered features with the quality data
        model_ready_df = batch_features.merge(quality_df, on='Batch_ID')

        # Drop intermediate columns used for calculation
        model_ready_df = model_ready_df.drop(columns=['Water (kg)_mean', 'Flour (kg)_mean', 'actual_hydration', 'ideal_hydration'])
        
        # --- 5. Save the Final Dataset ---
        model_ready_df.to_csv(self.config.output_data_path, index=False)
        logging.info(f"Feature engineering complete. Model-ready data saved to: {self.config.output_data_path}")


# The following code is just for testing this component directly
if __name__ == '__main__':
    # A simple config mock for direct execution
    class ConfigMock:
        process_data_path = "artifacts/data_generation/process_data.csv"
        quality_data_path = "artifacts/data_generation/batch_quality.csv"
        output_data_path = "artifacts/feature_engineering/model_ready_data.csv"

    try:
        logging.info(">>>>> Component: Feature Engineering started <<<<<")
        config = ConfigMock()
        feature_eng = FeatureEngineering(config=config)
        feature_eng.engineer_features()
        logging.info(">>>>> Component: Feature Engineering finished <<<<<")
    except Exception as e:
        logging.exception(e)
        raise e