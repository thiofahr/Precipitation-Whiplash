# master_script.py
# Python script to automate the data processing
import subprocess
import pandas as pd 
import time
import numpy as np 
from datetime import datetime


model = "ACCESS-CM2"
current_datetime = datetime.now()

# order of scripts to run
scripts_to_run = [
    "1-calculate_anomalies.py", 
    "2-calculate_threshold.py",
    "3-detect_extremes.py",
    "4-identify_whiplash.py",
    "5-calculate_frequency",
    "6-calculate_duration.py",
    "7-calculate_intensity.py"
]

data = []
for i, script in enumerate(scripts_to_run):
    try:
        # Use subprocess.run to execute each script
        print(f"{i+1}/{len(scripts_to_run)} Running {script} script")
        start_time = time.time()
        subprocess.run(["python", script], check=True)
        end_time = time.time()
        elapsed_time = end_time - end_time 
        data.append([model, script, elapsed_time])
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")
        data.append([model, script, 'error'])
        break  # Stop execution
data = np.array(data)

df = pd.DataFrame({
    "model" : data[:,0],
    "file"  : data[:,1],
    "time"  : data[:,2]
})

df.to_csv(rf"Log\process.{current_datetime}.csv", index=False)
