import os
import pandas as pd
import numpy as np
import random
import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


class DataGeneration:
    """
    Generates the final, balanced synthetic dataset based on the successful initial script.
    """
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config.root_dir, exist_ok=True)
        logging.info("Data generation output directory created.")

    def generate_data(self):
        # --- Using the exact parameters from the successful initial script ---
        IDEAL_CONDITIONS = {
            'Flour (kg)': 10.0, 'Sugar (kg)': 5.0, 'Yeast (kg)': 2.0,
            'Water (kg)': 5.5, 'Salt (kg)': 1.0, 'Mixer Speed (RPM)': 150.0,
            'Mixing Time (min)': 35.0, 'Fermentation Temp (C)': 35.0,
            'Oven Temp (C)': 180.0, 'Oven Humidity (%)': 45.0,
            'Flour Protein (%)': 11.0
        }
        NUM_BATCHES = 100
        TIME_STEPS_PER_BATCH = 25

        all_process_data = []
        batch_quality_list = []

        anomaly_functions = [self._simulate_oven_failure, self._simulate_material_mismatch, self._simulate_short_mix]

        for i in range(1, NUM_BATCHES + 1):
            batch_conditions = IDEAL_CONDITIONS.copy()
            
            if random.random() < 0.7:
                batch_df = self._generate_good_batch(batch_conditions, TIME_STEPS_PER_BATCH)
                is_anomaly = 0
            else:
                anomaly_func = random.choice(anomaly_functions)
                batch_df = anomaly_func(batch_conditions, TIME_STEPS_PER_BATCH)
                is_anomaly = 1

            quality_score = self._calculate_quality_score(batch_df)
            final_weight = 50.0 - (100 - quality_score) * 0.2 + np.random.normal(0, 0.5)

            batch_df.insert(0, 'Time', range(TIME_STEPS_PER_BATCH))
            batch_df.insert(0, 'Batch_ID', i)

            all_process_data.append(batch_df)
            batch_quality_list.append({
                'Batch_ID': i, 'Final Weight (kg)': final_weight,
                'Quality Score (%)': quality_score, 'Is_Anomaly': is_anomaly
            })

        final_process_df = pd.concat(all_process_data, ignore_index=True)
        final_quality_df = pd.DataFrame(batch_quality_list)

        final_process_df.to_csv(self.config.process_data_path, index=False)
        final_quality_df.to_csv(self.config.quality_data_path, index=False)
        logging.info(f"Successfully generated final balanced data to {self.config.root_dir}")

    # --- Helper methods using the original, successful parameters ---
    def _generate_good_batch(self, conditions, steps):
        data = {}
        for key, value in conditions.items():
            data[key] = np.random.normal(value, value * 0.01, steps)
        return pd.DataFrame(data)

    def _simulate_oven_failure(self, conditions, steps):
        data = {}
        for key, value in conditions.items():
            if key == 'Oven Temp (C)':
                data[key] = np.random.normal(value, 5.0, steps) # Severe anomaly
            else:
                data[key] = np.random.normal(value, value * 0.01, steps)
        return pd.DataFrame(data)

    def _simulate_material_mismatch(self, conditions, steps):
        data = {}
        conditions['Flour Protein (%)'] = 14.0 # Severe anomaly
        for key, value in conditions.items():
            data[key] = np.random.normal(value, value * 0.01, steps)
        return pd.DataFrame(data)

    def _simulate_short_mix(self, conditions, steps):
        data = {}
        conditions['Mixing Time (min)'] = 25.0 # Severe anomaly (10 min deviation)
        for key, value in conditions.items():
            data[key] = np.random.normal(value, value * 0.01, steps)
        return pd.DataFrame(data)

    def _calculate_quality_score(self, df_batch):
        score = 100.0
        oven_stability = df_batch['Oven Temp (C)'].std()
        score -= (oven_stability - 0.5) * 5
        mixing_time_error = abs(df_batch['Mixing Time (min)'].mean() - 35.0)
        score -= mixing_time_error * 1.5 # Original penalty
        protein = df_batch['Flour Protein (%)'].mean()
        water = df_batch['Water (kg)'].mean()
        flour = df_batch['Flour (kg)'].mean()
        actual_hydration = (water / flour) * 100
        ideal_hydration = protein * 5
        hydration_dev = abs(actual_hydration - ideal_hydration)
        score -= hydration_dev * 3 # Original penalty
        return max(min(score, 99.0), 70.0)

# For direct testing
if __name__ == '__main__':
    class ConfigMock:
        root_dir = "artifacts/data_generation"
        process_data_path = "artifacts/data_generation/process_data.csv"
        quality_data_path = "artifacts/data_generation/batch_quality.csv"
    try:
        logging.info(">>>>> Component: Final Data Generation started <<<<<")
        config = ConfigMock()
        data_gen = DataGeneration(config=config)
        data_gen.generate_data()
        logging.info(">>>>> Component: Final Data Generation finished <<<<<")
    except Exception as e:
        logging.exception(e)
        raise e