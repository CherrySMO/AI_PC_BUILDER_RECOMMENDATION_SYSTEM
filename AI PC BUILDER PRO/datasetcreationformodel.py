import pandas as pd
import random
import os
# Assuming you have the compatibility logic from backend.py available or refactored.
# For demonstration, I'll include a simplified compatibility check here,
# but ideally, you'd import or reuse the robust _check_compatibility from your Backend class.

# --- Re-using your Backend's DUMMY_PART_FEATURES for consistency ---
# (Copy this from your backend.py if it's not globally available)
DUMMY_PART_FEATURES = {
    "CPU": {
        "socket": lambda: random.choice(["LGA1700", "AM5", "LGA1200", "AM4"]),
        "TDP_W": lambda: random.randint(65, 250),
        "performance_tier": lambda: random.choice(["Low", "Mid", "High", "Enthusiast"])
    },
    "Motherboard": {
        "socket": lambda: random.choice(["LGA1700", "AM5", "LGA1200", "AM4"]),
        "ramType": lambda: random.choice(["DDR4", "DDR5"]),
        "pcieX16Slots": lambda: random.randint(1, 2),
        "formFactor": lambda: random.choice(["ATX", "Micro-ATX", "Mini-ITX"]),
        "ramSlots": lambda: random.choice([2, 4]),
        "sataPorts": lambda: random.randint(4, 8),
        "pcieSlots": lambda: random.randint(2, 5),
        "usbPorts": lambda: random.randint(6, 12)
    },
    "RAM": {
        "type": lambda: random.choice(["DDR4", "DDR5"]),
        "modules": lambda: random.choice([1, 2, 4])
    },
    "GPU": {
        "TDP_W": lambda: random.randint(75, 450),
        "length": lambda: random.randint(180, 350),
        "power_connectors_required": lambda: random.choice(["1x8pin", "2x8pin", "0x0pin"]),
        "performance_tier": lambda: random.choice(["Low", "Mid", "High", "Enthusiast"])
    },
    "PSU": {
        "power": lambda: random.randint(450, 1200),
        "pcie_connectors_available": lambda: random.randint(2, 6)
    },
    "Case": {
        "supported_form_factors": lambda: random.choice([
            ["ATX", "Micro-ATX", "Mini-ITX"],
            ["ATX", "Micro-ATX"],
            ["Mini-ITX"]
        ]),
        "max_gpu_length": lambda: random.randint(300, 450),
        "max_cooler_height": lambda: random.randint(150, 180)
    },
    "Cooler": {
        "compatible_sockets": lambda: random.sample(["LGA1700", "AM5", "LGA1200", "AM4", "TR4"], k=random.randint(1, 3)),
        "height": lambda: random.randint(120, 170)
    }
}
# --- End DUMMY_PART_FEATURES copy ---


BASE_PATH = r"C:\Users\Cantt Computer\project ai"
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

SCENARIOS = ["Gaming", "Workstation", "Content Creation", "Home Office", "General Use"]

# Scenario-based scoring weights (as in your provided script)
SCENARIO_WEIGHTS = {
    "Gaming": {"cpu_speed": 15, "cpu_cores": 10, "gpu_vram": 20, "ram_size": 5, "storage_space": 2},
    "Workstation": {"cpu_cores": 15, "ram_size": 20, "storage_space": 10, "psu_power": 5},
    "Content Creation": {"cpu_cores": 12, "ram_size": 20, "storage_space": 10, "gpu_vram": 8},
    "Home Office": {"cpu_speed": 10, "ram_size": 10, "storage_space": 5},
    "General Use": {"cpu_speed": 8, "ram_size": 8, "storage_space": 5},
}

def load_parts():
    data = {}
    for part_type, filename in PART_FILES.items():
        path = os.path.join(BASE_PATH, filename)
        try:
            df = pd.read_csv(path)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df.dropna(subset=['price'], inplace=True)

            # Add dummy data for missing columns (similar to Backend's _load_all_parts)
            if part_type in DUMMY_PART_FEATURES:
                for feature, generator in DUMMY_PART_FEATURES[part_type].items():
                    if feature not in df.columns:
                        df[feature] = [generator() for _ in range(len(df))]
                        # print(f"  [Data Gen Auto-fill] Added dummy '{feature}' column to {part_type} data.") # Debugging

            if {"name", "price", "image"}.issubset(df.columns):
                data[part_type] = df.reset_index(drop=True)
            else:
                print(f"[!] '{filename}' for {part_type} is missing required columns (name, price, image). Skipping.")
        except FileNotFoundError:
            print(f"[!] File not found: '{filename}' at expected path: '{path}'")
        except Exception as e:
            print(f"[!] Failed to read '{filename}' at '{path}': {e}")
    return data

# --- Simplified _check_compatibility for data generation ---
# This should ideally be robust and match your Backend's _check_compatibility logic.
# For a quick fix, this simulates basic checks.
def _check_compatibility_for_data_gen(proposed_build):
    cpu = proposed_build.get("CPU")
    motherboard = proposed_build.get("Motherboard")
    ram = proposed_build.get("RAM")
    gpu = proposed_build.get("GPU")
    psu = proposed_build.get("PSU")
    case = proposed_build.get("Case")
    cooler = proposed_build.get("Cooler")

    # CPU & Motherboard Socket
    if cpu and motherboard:
        if cpu.get('socket') != motherboard.get('socket'): return False
    
    # RAM & Motherboard Type
    if ram and motherboard:
        if ram.get('type') != motherboard.get('ramType'): return False

    # RAM Modules vs Motherboard Slots (simple check)
    if ram and motherboard:
        if ram.get('modules', 1) > motherboard.get('ramSlots', 4): return False

    # Case & Motherboard Form Factor
    if case and motherboard:
        mb_form_factor = motherboard.get('formFactor')
        case_supported_form_factors = case.get('supported_form_factors', [])
        if isinstance(case_supported_form_factors, str): # Handle string representation of list
            try: case_supported_form_factors = eval(case_supported_form_factors)
            except: pass
        if mb_form_factor and mb_form_factor not in case_supported_form_factors: return False

    # Case & GPU Length
    if case and gpu:
        if gpu.get('length', 0) > case.get('max_gpu_length', float('inf')): return False

    # Case & Cooler Height
    if case and cooler:
        if cooler.get('height', 0) > case.get('max_cooler_height', float('inf')): return False

    # Cooler & CPU Socket
    if cooler and cpu:
        cooler_compatible_sockets = cooler.get('compatible_sockets', [])
        if isinstance(cooler_compatible_sockets, str): # Handle string representation of list
            try: cooler_compatible_sockets = eval(cooler_compatible_sockets)
            except: pass
        if cpu.get('socket') and cpu.get('socket') not in cooler_compatible_sockets: return False

    # PSU Power (simple heuristic - ensure enough overall power for core components)
    if cpu and gpu and psu:
        estimated_min_power_needed = cpu.get('TDP_W', 0) + gpu.get('TDP_W', 0) + 50 # Base + buffer
        if psu.get('power', 0) < estimated_min_power_needed: return False

    return True # If all checks pass


def generate_build(parts_data, scenario, max_attempts=50):
    build = {}
    total_price = 0
    
    part_selection_order = ["CPU", "Motherboard", "RAM", "GPU", "Storage", "PSU", "Case", "Cooler"]

    for attempt in range(max_attempts):
        current_build_candidate = {}
        current_total_price_candidate = 0
        all_parts_selected = True

        for part_type in part_selection_order:
            df = parts_data.get(part_type)
            if df is None or df.empty:
                all_parts_selected = False
                break # Cannot complete build if part data is missing

            # Try to pick a random part for this slot
            selected_part = df.sample(1).iloc[0].to_dict()
            current_build_candidate[part_type] = selected_part
            current_total_price_candidate += selected_part['price']

            # --- Intermediate Compatibility Check ---
            # As parts are added, check compatibility with already selected parts
            if not _check_compatibility_for_data_gen(current_build_candidate):
                all_parts_selected = False
                break # This build candidate is incompatible, break and try new attempt
        
        if all_parts_selected and _check_compatibility_for_data_gen(current_build_candidate):
            # Build is complete and compatible, proceed to score
            build = current_build_candidate
            total_price = current_total_price_candidate
            break # Found a valid build, exit attempt loop
        else:
            # print(f"  Attempt {attempt+1}: Incompatible build, retrying.") # Debugging incompatible builds
            continue # Try another random combination

    if not build:
        # print("Failed to generate a compatible build after multiple attempts.") # Debugging
        return None # Could not generate a compatible build within max_attempts

    # Extract build-level features (only from the final, compatible build)
    features = {
        "scenario": scenario,
        "cpu_speed": build["CPU"].get("speed", 0),
        "cpu_cores": build["CPU"].get("coreCount", 0),
        "cpu_threads": build["CPU"].get("threadCount", 0),
        "gpu_vram": build["GPU"].get("VRAM", 0),
        "ram_size": build["RAM"].get("size", 0),
        "storage_space": build["Storage"].get("space", 0),
        "psu_power": build["PSU"].get("power", 0),
        "total_price": total_price
    }

    # Score calculation (scenario-specific)
    weights = SCENARIO_WEIGHTS[scenario]
    score = sum(features.get(k, 0) * w for k, w in weights.items())
    features["performance_score"] = score

    return features

def generate_dataset(num_builds=5000): # Increased num_builds for better ML training
    parts_data = load_parts()
    rows = []
    
    # Track attempts vs successful builds to gauge efficiency
    successful_builds_count = 0
    total_attempts_made = 0

    while successful_builds_count < num_builds and total_attempts_made < num_builds * 10: # Limit total attempts
        scenario = random.choice(SCENARIOS)
        build = generate_build(parts_data, scenario)
        total_attempts_made += 1
        if build:
            rows.append(build)
            successful_builds_count += 1
            if successful_builds_count % 100 == 0:
                print(f"Generated {successful_builds_count}/{num_builds} compatible builds...")

    df = pd.DataFrame(rows)
    output_path = os.path.join(BASE_PATH, "scenario_pc_builds.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved {successful_builds_count} compatible scenario-based builds to {output_path}")
    print(f"Total attempts made: {total_attempts_made}")


if __name__ == "__main__":
    generate_dataset()