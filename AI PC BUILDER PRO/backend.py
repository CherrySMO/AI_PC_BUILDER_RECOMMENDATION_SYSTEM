import pandas as pd
import numpy as np
import os
import joblib

# === Configuration (kept in backend as they are intrinsic to data loading and processing) ===
# IMPORTANT: This BASE_PATH should point to where your CSV data files and ML models are stored.
# If they are in a subfolder named 'data' relative to where this backend.py file is,
# then 'data' is correct. Otherwise, adjust this path.
BASE_PATH = os.path.join(os.path.dirname(__file__), "data") # Ensure this path is correct
PART_FILES = {
    "CPU": "CPU.csv",
    "GPU": "GPU.csv",
    "Motherboard": "Motherboard.csv",
    "RAM": "Memory.csv",
    "Storage": "Storage.csv",
    "PSU": "PSU.csv",
    "Case": "Case.csv",
    "Cooler": "Cooler.csv"
}

# Dummy features for Motherboard if not present in CSVs (as used in previous versions)
DUMMY_MOTHERBOARD_FEATURES = {
    "sataPorts": lambda: np.random.randint(4, 8),
    "pcieSlots": lambda: np.random.randint(2, 5),
    "usbPorts": lambda: np.random.randint(6, 12)
}

class Backend:
    """
    Backend class to handle all data loading, processing, and recommendation logic
    for the PC Recommendation App. This class is UI-agnostic.
    """
    def __init__(self, base_path=BASE_PATH, part_files=PART_FILES):
        """
        Initializes the Backend with paths to data files and loads all necessary data
        and machine learning models.

        Args:
            base_path (str): The base directory where CSV and model files are located.
            part_files (dict): A dictionary mapping part types to their respective CSV filenames.
        """
        self.BASE_PATH = base_path
        self.PART_FILES = part_files
        # Load all parts data from CSVs
        self.parts_data = self._load_all_parts()
        # Precompute normalization ranges for consistent scoring and plotting
        self.normalization_ranges = self._precompute_normalization_ranges()
        # Load all machine learning models
        self.ml_models = self._load_ml_models()
        # Define scenarios supported by the application
        self.scenarios = ["Gaming", "Workstation", "Content Creation", "Home Office", "General Use"]

    def _load_all_parts(self):
        """
        Loads all part data from CSV files specified in PART_FILES.
        Ensures 'price' column is numeric and adds dummy data for Motherboard if needed.
        Handles file not found and reading errors gracefully.
        """
        parts_data = {}
        for part, filename in self.PART_FILES.items():
            path = os.path.join(self.BASE_PATH, filename)
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    # Convert 'price' column to numeric, coercing invalid parsing to NaN
                    df['price'] = pd.to_numeric(df['price'], errors='coerce')
                    # Remove rows where price is NaN after conversion
                    df.dropna(subset=['price'], inplace=True) 

                    # Add dummy numerical data for Motherboard if necessary columns are missing
                    if part == "Motherboard":
                        for feature, generator in DUMMY_MOTHERBOARD_FEATURES.items():
                            if feature not in df.columns:
                                df[feature] = [generator() for _ in range(len(df))]

                    # Check if essential columns are present before adding to data
                    if {"name", "price", "image"}.issubset(df.columns):
                        # Sort by price for consistent selection (e.g., cheapest/most expensive)
                        parts_data[part] = df.sort_values(by="price")
                    else:
                        print(f"[!] '{filename}' is missing required columns (name, price, image). Skipping.")
                except Exception as e:
                    print(f"[!] Failed to read '{filename}' at '{path}': {e}")
            else:
                print(f"[!] File not found: '{filename}' at expected path: '{path}'")
        return parts_data

    def _precompute_normalization_ranges(self):
        """
        Calculates and stores min/max for all numerical features across all parts.
        These ranges are used for consistent normalization (0-1 scale) in scoring
        and for charting purposes.
        """
        normalization_ranges = {}
        for part_type, df in self.parts_data.items():
            for col in df.columns:
                # Check if column is numeric and contains at least one non-NaN value
                if pd.api.types.is_numeric_dtype(df[col]):
                    valid_values = df[col].dropna()
                    if not valid_values.empty:
                        normalization_ranges[f"{part_type}_{col}"] = {
                            'min': valid_values.min(),
                            'max': valid_values.max()
                        }
        return normalization_ranges

    def _load_ml_models(self):
        """
        Loads pre-trained machine learning models for different usage scenarios.
        Models are expected to be .pkl files in the BASE_PATH.
        """
        ml_models = {}
        # Define the specific scenarios for which models are expected
        scenarios_to_load = ["Gaming", "Workstation", "Content Creation", "Home Office", "General Use"]
        for scenario in scenarios_to_load:
            file_name = f"rf_{scenario.lower().replace(' ', '_')}.pkl" # Construct model filename
            model_path = os.path.join(self.BASE_PATH, file_name)
            try:
                # Load the model using joblib
                ml_models[scenario] = joblib.load(model_path)
                print(f"[✓] Loaded ML model for {scenario}: {file_name}")
            except FileNotFoundError:
                print(f"[!] ML model not found for scenario: '{scenario}' at '{model_path}'")
            except Exception as e:
                print(f"[!] Error loading ML model '{file_name}': {e}")
        return ml_models

    def recommend_builds_rule_based(self, budget):
        """
        Generates 3 different rule-based PC build recommendations:
        Budget, Balanced, and Performance builds, all within the given budget.

        Args:
            budget (float): The maximum budget for the PC build.

        Returns:
            list: A list of dictionaries, where each dictionary represents a recommended
                  build including its parts, total cost, remaining budget, and type.
                  Returns an empty list if no complete builds can be assembled.
        """
        recommendations = []

        # Ensure all required part data is loaded before attempting to recommend
        if not all(part_type in self.parts_data for part_type in self.PART_FILES.keys()):
            print("Missing required CSV files for rule-based recommendation. Cannot generate builds.")
            return []

        # Iterate through different build strategies (Budget, Balanced, Performance)
        for i in range(3):
            current_build = {}
            current_total_cost = 0
            temp_budget = budget # Use a temporary budget that decreases as parts are added

            all_parts_found = True
            for part_type, df in self.parts_data.items():
                # Filter for parts within the current temporary budget
                affordable_parts = df[df['price'] <= temp_budget].copy()

                if affordable_parts.empty:
                    # If no affordable parts for a type, this build is not possible
                    all_parts_found = False
                    break

                # Select a part based on the current strategy (Budget, Balanced, Performance)
                if i == 0:  # Budget build: select the cheapest available part
                    selected_part = affordable_parts.iloc[0]
                elif i == 1:  # Balanced build: select a mid-range part
                    if len(affordable_parts) > 1:
                        mid_index = len(affordable_parts) // 2
                        selected_part = affordable_parts.iloc[mid_index]
                    else:
                        selected_part = affordable_parts.iloc[0] # Fallback if only one part
                else:  # Performance build: select the most expensive affordable part
                    selected_part = affordable_parts.iloc[-1]

                # Add selected part to the current build and update costs
                current_build[part_type] = selected_part.to_dict()
                current_total_cost += selected_part['price']
                temp_budget -= selected_part['price'] # Reduce budget for next part

            # If a complete build was successfully assembled within the budget
            if all_parts_found:
                build_type = ["💰 Budget Build", "⚖️ Balanced Build", "🚀 Performance Build"][i]
                recommendations.append({
                    "parts": current_build,
                    "cost": current_total_cost,
                    "remaining": budget - current_total_cost,
                    "type": build_type
                })
        return recommendations

    def generate_smart_recommendation_rule_based(self, scenario, budget):
        """
        Generates a smart PC recommendation based on a specified usage scenario and budget.
        This uses a rule-based system with feature weights tailored to the scenario.

        Args:
            scenario (str): The primary usage scenario (e.g., "Gaming", "Workstation").
            budget (float): The maximum budget for the PC build.

        Returns:
            dict: A dictionary representing the recommended build, or None if a complete
                  and compatible build cannot be generated.
        """
        recommended_build = {}
        current_cost = 0

        # Define feature weights for different scenarios.
        # Negative weights indicate that a lower value is better (e.g., price).
        feature_weights = {
            "Gaming": {"CPU": {"speed": 0.3, "coreCount": 0.2, "power": 0.1, "price": -0.4},
                       "GPU": {"VRAM": 0.4, "power": 0.2, "price": -0.4},
                       "RAM": {"size": 0.3, "price": -0.7},
                       "Storage": {"space": 0.1, "price": -0.9},
                       "PSU": {"power": 0.3, "price": -0.7},
                       "Motherboard": {"price": -1.0}, # Less emphasis on specific motherboard features, prioritize compatibility/price
                       "Case": {"price": -1.0},
                       "Cooler": {"price": -1.0}},
            
            "Workstation": {"CPU": {"coreCount": 0.4, "threadCount": 0.3, "speed": 0.2, "power": 0.1, "price": -0.5},
                            "GPU": {"VRAM": 0.2, "price": -0.8},
                            "RAM": {"size": 0.6, "price": -0.4},
                            "Storage": {"space": 0.5, "price": -0.5},
                            "PSU": {"power": 0.2, "price": -0.8},
                            "Motherboard": {"price": -1.0},
                            "Case": {"price": -1.0},
                            "Cooler": {"price": -1.0}},
            
            "Content Creation": {"CPU": {"coreCount": 0.35, "threadCount": 0.35, "speed": 0.1, "price": -0.4},
                                 "GPU": {"VRAM": 0.3, "price": -0.7},
                                 "RAM": {"size": 0.5, "price": -0.5},
                                 "Storage": {"space": 0.4, "price": -0.6},
                                 "PSU": {"power": 0.2, "price": -0.8},
                                 "Motherboard": {"price": -1.0},
                                 "Case": {"price": -1.0},
                                 "Cooler": {"price": -1.0}},

            "Home Office": {"CPU": {"speed": 0.2, "coreCount": 0.1, "price": -0.7},
                            "GPU": {"price": -1.0},
                            "RAM": {"size": 0.3, "price": -0.7},
                            "Storage": {"space": 0.2, "price": -0.8},
                            "PSU": {"power": 0.1, "price": -0.9},
                            "Motherboard": {"price": -1.0},
                            "Case": {"price": -1.0},
                            "Cooler": {"price": -1.0}},
            
            "General Use": {"CPU": {"speed": 0.2, "coreCount": 0.1, "price": -0.7},
                            "GPU": {"price": -1.0},
                            "RAM": {"size": 0.3, "price": -0.7},
                            "Storage": {"space": 0.2, "price": -0.8},
                            "PSU": {"power": 0.1, "price": -0.9},
                            "Motherboard": {"price": -1.0},
                            "Case": {"price": -1.0},
                            "Cooler": {"price": -1.0}},
        }

        # Get weights for the selected scenario, default to "General Use" if not found
        weights = feature_weights.get(scenario, feature_weights["General Use"])

        selected_cpu = None
        selected_motherboard = None

        cpu_df = self.parts_data.get("CPU")
        motherboard_df = self.parts_data.get("Motherboard")

        if cpu_df is None or motherboard_df is None:
            print("CPU or Motherboard data missing for smart recommendation. Cannot generate build.")
            return None

        # --- CPU Selection ---
        # Allocate a portion of the budget for the CPU
        cpu_budget_allocation = budget * 0.25 
        affordable_cpus = cpu_df[cpu_df['price'] <= cpu_budget_allocation].copy() 
        
        if affordable_cpus.empty and budget > 0:
             # If initial allocation too strict, try a larger portion of the total budget
             affordable_cpus = cpu_df[cpu_df['price'] <= budget * 0.4].copy() 
             if affordable_cpus.empty and not cpu_df.empty:
                 # As a last resort, pick the cheapest CPU if no other fits
                 affordable_cpus = cpu_df.sort_values(by='price').head(1).copy() 
                 print("Warning: No CPU found within allocated budget. Selecting cheapest available CPU.")

        if affordable_cpus.empty:
            return None # Cannot proceed without a CPU

        # Select the best CPU based on scenario-specific scoring
        selected_cpu = self._select_best_part_by_score(affordable_cpus, weights.get("CPU", {}), self.normalization_ranges, "CPU")
        if selected_cpu:
            recommended_build["CPU"] = selected_cpu
            current_cost += selected_cpu['price']
            remaining_budget = budget - current_cost
            
            # --- Motherboard Selection (compatible with selected CPU) ---
            motherboard_budget_allocation = remaining_budget * 0.25 
            # Filter motherboards by socket compatibility and budget
            compatible_motherboards = motherboard_df[
                (motherboard_df['socket'] == selected_cpu['socket']) &
                (motherboard_df['price'] <= motherboard_budget_allocation)
            ].copy()

            if compatible_motherboards.empty and remaining_budget > 0:
                # If initial allocation too strict, try a larger portion of remaining budget
                compatible_motherboards = motherboard_df[
                    (motherboard_df['socket'] == selected_cpu['socket']) &
                    (motherboard_df['price'] <= remaining_budget * 0.4)
                ].copy()
                if compatible_motherboards.empty and not motherboard_df.empty:
                    # As a last resort, pick the cheapest compatible motherboard
                    compatible_motherboards = motherboard_df[
                        (motherboard_df['socket'] == selected_cpu['socket'])
                    ].sort_values(by='price').head(1).copy() 
                    if not compatible_motherboards.empty:
                        print("Warning: No compatible Motherboard found within allocated budget. Selecting cheapest compatible Motherboard.")
                    else:
                        return None # No compatible motherboard at all
            if compatible_motherboards.empty:
                return None # Cannot proceed without a compatible motherboard

            # Select the best motherboard based on scoring (though motherboard weights are minimal in example)
            selected_motherboard = self._select_best_part_by_score(compatible_motherboards, weights.get("Motherboard", {}), self.normalization_ranges, "Motherboard")
            if selected_motherboard:
                recommended_build["Motherboard"] = selected_motherboard
                current_cost += selected_motherboard['price']
                remaining_budget = budget - current_cost
        
        # If CPU or Motherboard selection failed, return None
        if not selected_cpu or not selected_motherboard:
            return None

        # --- Selection of Remaining Parts (GPU, RAM, Storage, PSU, Case, Cooler) ---
        # Prioritize parts based on scenario (e.g., GPU for gaming, RAM/Storage for workstation)
        part_types_order = ["GPU", "RAM", "Storage", "PSU", "Case", "Cooler"]
        
        budget_allocation_priority = {
            "Gaming": {"GPU": 0.5, "RAM": 0.2, "Storage": 0.15, "PSU": 0.1, "Case": 0.05, "Cooler": 0.0},
            "Workstation": {"GPU": 0.15, "RAM": 0.3, "Storage": 0.3, "PSU": 0.15, "Case": 0.05, "Cooler": 0.05},
            "Content Creation": {"GPU": 0.25, "RAM": 0.25, "Storage": 0.25, "PSU": 0.15, "Case": 0.05, "Cooler": 0.05},
            "Home Office": {"GPU": 0.05, "RAM": 0.2, "Storage": 0.2, "PSU": 0.1, "Case": 0.1, "Cooler": 0.05},
            "General Use": {"GPU": 0.1, "RAM": 0.25, "Storage": 0.25, "PSU": 0.15, "Case": 0.1, "Cooler": 0.05},
        }
        
        current_allocations = budget_allocation_priority.get(scenario, budget_allocation_priority["General Use"])
        
        for part_type in part_types_order:
            if part_type in self.parts_data:
                # Allocate a portion of the *remaining* budget for the current part type
                allocated_part_budget = remaining_budget * current_allocations.get(part_type, 0.1)
                
                affordable_parts = self.parts_data[part_type][
                    self.parts_data[part_type]['price'] <= allocated_part_budget
                ].copy()

                if affordable_parts.empty:
                    # If no parts fit allocated budget, try within the entire remaining budget
                    affordable_parts = self.parts_data[part_type][
                        self.parts_data[part_type]['price'] <= remaining_budget
                    ].copy()
                    if affordable_parts.empty:
                        # If still no parts, and the original dataframe for this part isn't empty, pick the cheapest
                        if not self.parts_data[part_type].empty:
                            selected_part = self.parts_data[part_type].iloc[0].to_dict()
                            print(f"Warning: Selected cheapest {part_type} as no part fit allocated budget.")
                        else:
                            return None # Cannot complete build if critical part data is missing or empty
                
                if not affordable_parts.empty:
                    # Select the best part from the affordable options using scoring
                    selected_part = self._select_best_part_by_score(affordable_parts, weights.get(part_type, {}), self.normalization_ranges, part_type)
                    if selected_part:
                        recommended_build[part_type] = selected_part
                        current_cost += selected_part['price']
                        remaining_budget = budget - current_cost # Update remaining budget
                    else:
                        print(f"Could not select a {part_type} even from affordable parts. Skipping this build path.")
                        return None # If a part can't be selected, the build is invalid
                else:
                    return None # Cannot complete build if no affordable parts are found for a type
            else:
                return None # Cannot complete build if part type data is missing

        # Final check to ensure all required core parts are in the build
        required_parts = ["CPU", "Motherboard", "RAM", "GPU", "Storage", "PSU"]
        if not all(part in recommended_build for part in required_parts):
            print("Required parts are missing in the final smart build recommendation.")
            return None
        
        return recommended_build

    def generate_ml_based_recommendations(self, budget, scenario="General Use", num_samples=50, top_n=3):
        """
        Generates smart PC recommendations by sampling a number of random builds (within budget)
        and then selecting the top N builds with the highest predicted scores from the ML model.

        Args:
            budget (float): The maximum budget for the PC build.
            scenario (str, optional): The usage scenario for which to predict performance. Defaults to "General Use".
            num_samples (int, optional): The number of random builds to sample and evaluate. Defaults to 50.
            top_n (int, optional): The number of top-scoring builds to return. Defaults to 3.

        Returns:
            list: A list of the best N ML-optimized builds (dictionaries), each augmented with
                  an '_ml_score' key. Returns an empty list if no valid ML-based builds are found.
        """
        # Ensure the ML model for the specified scenario is loaded
        if scenario not in self.ml_models or self.ml_models[scenario] is None:
            print(f"ML model not loaded for scenario '{scenario}'. Cannot generate ML-based recommendations.")
            return []

        all_valid_ml_builds = [] # List to store all successfully sampled builds with their scores

        print(f"\n--- Starting ML-based recommendation for {scenario} with budget ${budget} (samples: {num_samples}, top_n: {top_n}) ---")

        for i in range(num_samples):
            current_sampled_build = {}
            current_total_cost = 0
            temp_budget = budget

            all_parts_found = True
            selected_cpu_socket = None

            try:
                # --- CPU Selection (random, within budget) ---
                cpu_df = self.parts_data.get("CPU")
                if cpu_df is None or cpu_df.empty: raise ValueError("CPU data missing or empty.")
                
                # Try to pick a CPU within 40% of the budget
                affordable_cpus = cpu_df[cpu_df['price'] <= temp_budget * 0.4].copy()
                if affordable_cpus.empty:
                    # Fallback to the cheapest CPU if initial filter is too strict
                    affordable_cpus = cpu_df.sort_values(by='price').head(1).copy()
                if affordable_cpus.empty: raise ValueError("No affordable CPUs found for sampling.")

                selected_cpu_part = affordable_cpus.sample(1).iloc[0].to_dict() # Randomly sample one CPU
                
                current_sampled_build["CPU"] = selected_cpu_part
                current_total_cost += selected_cpu_part['price']
                temp_budget -= selected_cpu_part['price']
                selected_cpu_socket = selected_cpu_part.get('socket')

                # --- Motherboard Selection (compatible with CPU, random, within budget) ---
                motherboard_df = self.parts_data.get("Motherboard")
                if motherboard_df is None or motherboard_df.empty: raise ValueError("Motherboard data missing or empty.")
                
                if selected_cpu_socket is None:
                    raise ValueError("Selected CPU has no socket information for motherboard compatibility.")

                # Try to pick a compatible motherboard within 50% of the remaining budget
                compatible_motherboards = motherboard_df[
                    (motherboard_df['socket'] == selected_cpu_socket) &
                    (motherboard_df['price'] <= temp_budget * 0.5) 
                ].copy()
                if compatible_motherboards.empty:
                     # Fallback to the cheapest compatible motherboard
                     compatible_motherboards = motherboard_df[
                        (motherboard_df['socket'] == selected_cpu_socket)
                    ].sort_values(by='price').head(1).copy()
                if compatible_motherboards.empty: raise ValueError("No compatible Motherboards found for sampling.")

                selected_mb_part = compatible_motherboards.sample(1).iloc[0].to_dict() # Randomly sample one Motherboard
                
                current_sampled_build["Motherboard"] = selected_mb_part
                current_total_cost += selected_mb_part['price']
                temp_budget -= selected_mb_part['price']

                # --- Remaining Parts Selection (random, within remaining budget) ---
                remaining_part_types = ["GPU", "RAM", "Storage", "PSU", "Case", "Cooler"]
                for part_type in remaining_part_types:
                    df = self.parts_data.get(part_type)
                    if df is None or df.empty: raise ValueError(f"Data for {part_type} missing or empty.")

                    affordable_parts = df[df['price'] <= temp_budget].copy()
                    if affordable_parts.empty:
                        if not df.empty:
                            selected_part = df.iloc[0].to_dict() # Fallback to cheapest if no part fits remaining budget
                        else:
                            raise ValueError(f"No affordable {part_type} available.")
                    else:
                        selected_part = affordable_parts.sample(1).iloc[0].to_dict() # Randomly sample one part

                    current_sampled_build[part_type] = selected_part
                    current_total_cost += selected_part['price']
                    temp_budget -= selected_part['price']

            except ValueError as e:
                # If any part selection fails for a sample, skip this sample
                print(f"Sample {i+1} failed to create a valid build due to: {e}")
                all_parts_found = False
                continue

            # Check if the sampled build is complete and within budget
            if all_parts_found and current_total_cost <= budget:
                print(f"Sample {i+1}: Successfully assembled build. Cost: ${current_total_cost:.2f}")
                
                # Predict the performance score for this complete build using the ML model
                predicted_score = self.predict_build_score(current_sampled_build, scenario)
                
                if predicted_score is not None:
                    print(f"Sample {i+1}: Predicted ML score: {predicted_score:.2f}")
                    # Store the ML score directly in the build dictionary
                    current_sampled_build['_ml_score'] = predicted_score
                    all_valid_ml_builds.append(current_sampled_build)
                    
                    # Optimization: If an excellent build is found early, we might stop
                    if predicted_score > 98 and len(all_valid_ml_builds) >= top_n:
                        print(f"Found {top_n} excellent builds with high scores (one > 98), potentially stopping early.")
                        # This could be a break, but continuing ensures we get exactly top_n if more high scores exist
                        # and ensures we have sufficient diversity for 'top_n'
                else:
                    print(f"Sample {i+1}: ML prediction failed for this build, skipping.")
            else:
                print(f"Sample {i+1}: Build failed validation (parts not found or over budget).")
        
        # Sort all successfully generated and scored builds by their ML score in descending order
        all_valid_ml_builds.sort(key=lambda x: x.get('_ml_score', -np.inf), reverse=True)

        # Return only the top N builds based on the ML score
        top_ml_builds = all_valid_ml_builds[:top_n]

        if top_ml_builds:
            print(f"--- ML-based recommendation finished. Found {len(top_ml_builds)} best builds. Highest score: {top_ml_builds[0].get('_ml_score', 'N/A'):.2f} ---")
        else:
            print("--- ML-based recommendation finished. No valid ML-based build found. ---")
        
        return top_ml_builds

    def predict_build_score(self, build_dict, scenario):
        """
        Predicts the performance score of a given PC build using the scenario-specific
        machine learning model.

        Args:
            build_dict (dict): A dictionary representing the PC build with its components.
            scenario (str): The usage scenario (e.g., "Gaming") for which to use the model.

        Returns:
            float: The predicted score of the build (e.g., 0-100), or None if prediction fails.
        """
        model = self.ml_models.get(scenario)
        if model is None:
            print(f"[!] No ML model loaded for scenario '{scenario}'. Cannot predict score.")
            return None

        # Prepare the features from the build dictionary into a format expected by the ML model.
        # Ensure all expected features for your model are present, even if with default 0.
        features = {
            "cpu_speed": build_dict.get('CPU', {}).get('speed', 0),
            "cpu_cores": build_dict.get('CPU', {}).get('coreCount', 0),
            "cpu_threads": build_dict.get('CPU', {}).get('threadCount', 0),
            "gpu_vram": build_dict.get('GPU', {}).get('VRAM', 0),
            "ram_size": build_dict.get('RAM', {}).get('size', 0),
            "storage_space": build_dict.get('Storage', {}).get('space', 0),
            "psu_power": build_dict.get('PSU', {}).get('power', 0),
            # Calculate total price from all parts in the build
            "total_price": sum(part.get('price', 0) for part in build_dict.values() if isinstance(part, dict))
        }
        
        # Convert features into a pandas DataFrame, as expected by scikit-learn models
        df = pd.DataFrame([features])
        # Uncomment for debugging feature input:
        # print(f"  [Predictor] Features DataFrame for prediction:\n{df}") 

        try:
            prediction = model.predict(df)[0]
            # Ensure predictions are non-negative, as scores usually are not.
            if prediction < 0:
                print(f"  [Predictor] Warning: Predicted score was negative ({prediction:.2f}). Clamping to 0.")
                prediction = max(0, prediction) # Clamp to minimum of 0
            return prediction
        except Exception as e:
            print(f"[!] Error during ML model prediction for scenario '{scenario}': {e}")
            print(f"  [Predictor] Problematic features for prediction: {features}")
            return None # Indicate prediction failure

    def _select_best_part_by_score(self, parts_df, part_weights, normalization_ranges, part_type):
        """
        Selects the single best part from a given DataFrame of parts based on a weighted
        scoring system. Features are normalized to a 0-1 scale using global min/max ranges.
        Price is inherently treated as a negative factor (lower price is better).

        Args:
            parts_df (pd.DataFrame): DataFrame of parts to select from.
            part_weights (dict): A dictionary mapping feature names to their importance weights
                                 for the current part type and scenario.
            normalization_ranges (dict): A dictionary containing precomputed min/max ranges
                                         for all numerical features across all part types,
                                         used for consistent normalization.
            part_type (str): The type of part (e.g., "CPU", "GPU") currently being selected.

        Returns:
            dict: The dictionary representation of the selected best part, or None if the
                  input DataFrame is empty or no suitable part can be scored.
        """
        if parts_df.empty:
            return None

        scores = []
        for index, part in parts_df.iterrows():
            score = 0
            for feature, weight in part_weights.items():
                val = part.get(feature) # Safely get feature value, defaults to None if not present
                
                # Check if the feature value is valid for scoring (exists, is numeric, and not NaN)
                if val is not None and pd.api.types.is_numeric_dtype(type(val)) and pd.notna(val):
                    full_feature_name = f"{part_type}_{feature}" # Construct unique key for normalization range
                    
                    if full_feature_name in normalization_ranges:
                        min_val = normalization_ranges[full_feature_name]['min']
                        max_val = normalization_ranges[full_feature_name]['max']
                        
                        if max_val - min_val > 0:
                            # Normalize feature value to a 0-1 scale
                            normalized_val = (val - min_val) / (max_val - min_val)
                        else:
                            normalized_val = 0.5 # Default to mid-range if all values are identical (no range)
                    else: 
                        # If no normalization range is found (e.g., non-critical feature), use raw value
                        # This is less ideal but prevents errors. Best practice is to ensure all relevant
                        # features have normalization ranges.
                        normalized_val = val

                    score += normalized_val * weight # Add weighted normalized value to total score
            scores.append(score)
        
        if scores:
            # Add the computed scores as a new column to the DataFrame (using .loc for safe assignment)
            parts_df.loc[:, 'score'] = scores 
            # Return the part (as a dictionary) with the highest computed score
            return parts_df.loc[parts_df['score'].idxmax()].to_dict()
        else:
            # If no features were applicable for scoring, or all scores are 0,
            # fallback to selecting the cheapest part (price is a universal factor)
            print(f"Warning: No applicable numeric features for scoring {part_type} parts. Falling back to cheapest.")
            return parts_df.sort_values(by='price').iloc[0].to_dict()

    def get_parts_data(self):
        """
        Provides access to the loaded parts data from the backend.
        Used by the frontend to populate dropdowns and display information.
        """
        return self.parts_data
    
    def get_normalization_ranges(self):
        """
        Provides access to the precomputed normalization ranges.
        Used by the frontend for consistent plotting and comparison.
        """
        return self.normalization_ranges

    def get_scenarios(self):
        """
        Provides access to the list of defined usage scenarios.
        Used by the frontend to populate scenario selection dropdowns.
        """
        return self.scenarios

