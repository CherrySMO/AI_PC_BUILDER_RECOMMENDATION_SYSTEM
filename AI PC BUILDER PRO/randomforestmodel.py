import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.metrics import accuracy_score
# Adjust path to your actual CSV location
BASE_PATH = r"C:\Users\Cantt Computer\project ai"
CSV_PATH = os.path.join(BASE_PATH, "scenario_pc_builds.csv")
SCENARIOS = ["Gaming", "Workstation", "Content Creation", "Home Office", "General Use"]

# Load dataset
df = pd.read_csv(CSV_PATH)

# Features used by all models
FEATURE_COLUMNS = ["cpu_speed", "cpu_cores", "cpu_threads", "gpu_vram", "ram_size", "storage_space", "psu_power", "total_price"]

# Train and save one model per scenario
for scenario in SCENARIOS:
    scenario_df = df[df["scenario"] == scenario]

    if scenario_df.empty:
        print(f"[!] No data found for scenario: {scenario}")
        continue

    X = scenario_df[FEATURE_COLUMNS]
    y = scenario_df["performance_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n✅ Trained model for {scenario}")
    print(f"   🔹 R² Score      : {r2:.4f}")
    print(f"   🔹 MSE           : {mse:.4f}")
    print(f"   🔹 RMSE          : {rmse:.4f}")
    print(f"   🔹 MAE           : {mae:.4f}")

    # Save model
    model_path = os.path.join(BASE_PATH, f"rf_{scenario.lower().replace(' ', '_')}.pkl")
    joblib.dump(model, model_path)
    print(f"📦 Model saved to: {model_path}")
