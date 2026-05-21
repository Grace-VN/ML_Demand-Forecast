# -*- coding: utf-8 -*-
import subprocess
import sys
from pathlib import Path

# ── STEP 1: LOCATE PROJECT ROOT ──────────────────────────────────────────────
# This calculates 'D:\Job\Portfolio\Machine Learning\Demand Forecast'
ROOT_DIR = Path(__file__).resolve().parent

def execute_feature_importance():
    # Target the exact script path using pathlib
    script_path = ROOT_DIR / "model" / "output interpretation" / "feature_importance.py"
    
    if not script_path.exists():
        print(f"❌ Error: Script not found at target location:\n   {script_path}")
        return False
    
    print("=" * 70)
    print(f"🚀 Launching Feature Importance Analysis Script")
    print(f"📂 Execution Path: {script_path.parent}")
    print("=" * 70)
    
    # Run the script in its own isolated process. 
    # Passing a list ['python', str(script_path)] tells the OS exactly where the file boundaries
    # are, completely neutralizing the space in the 'output interpretion' folder name.
    result = subprocess.run(
        [sys.executable, str(script_path)], 
        cwd=ROOT_DIR  # Keeps the root folder as the working directory context
    )
    
    print("\n" + "=" * 70)
    if result.returncode == 0:
        print("✅ Feature Importance script completed successfully!")
        return True
    else:
        print(f"💥 Script execution failed with exit code: {result.returncode}")
        return False

if __name__ == "__main__":
    execute_feature_importance()