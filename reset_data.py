import os
import csv
import pandas as pd

# === Path configuration ===
SCREENSHOT_DIR = r'D:\screenshots'
CSV_PATH = 'record.csv'
MODELS_DIR = r'public/models'   # <--- folder where your .glb files are stored


# === Clear screenshots folder ===
def clear_screenshots():
    if os.path.exists(SCREENSHOT_DIR):
        for filename in os.listdir(SCREENSHOT_DIR):
            file_path = os.path.join(SCREENSHOT_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f'✅ Cleared {SCREENSHOT_DIR}/')
    else:
        print(f'⚠️ Directory {SCREENSHOT_DIR}/ not found.')


# === Reset CSV (keep headers) ===
def reset_csv():
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sessionId',
            'model',
            'actionId',
            's_t_img',
            's_t1_img',
            'after_yaw', 'after_pitch',
            'delta_yaw', 'delta_pitch',
            'init_yaw', 'init_pitch',
            'reward'
        ])
    print(f'✅ Reset {CSV_PATH} (headers only).')


# === NEW: Clear .glb models that appear in record.csv ===
def clear_recorded_models():
    if not os.path.exists(CSV_PATH):
        print(f'⚠️ CSV file {CSV_PATH} not found.')
        return

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f'❌ Failed to read CSV: {e}')
        return

    if 'model' not in df.columns:
        print("⚠️ Column 'model' not found in record.csv.")
        return

    # Get unique list of model filenames (strings like "Set_11_elong.glb")
    model_names = df['model'].dropna().unique()

    deleted = []
    missing = []

    for name in model_names:
        glb_path = os.path.join(MODELS_DIR, name)
        if os.path.isfile(glb_path):
            os.remove(glb_path)
            deleted.append(name)
        else:
            missing.append(name)

    print(f'🧹 Cleared {len(deleted)} models from {MODELS_DIR}:')
    for d in deleted:
        print(f'   - deleted: {d}')

    if missing:
        print(f'⚠️ {len(missing)} models listed in CSV were not found:')
        for m in missing:
            print(f'   - missing: {m}')


# === Run all resets ===
if __name__ == '__main__':
    clear_screenshots()
    clear_recorded_models()   # <--- NEW
    reset_csv()
