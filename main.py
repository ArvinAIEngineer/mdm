import pandas as pd
import random
import names
from datetime import datetime
import os

# Configuration
DATA_FILE = "customers.csv"
NUM_RECORDS = 100

def generate_synthetic_data():
    # Check if file already exists to avoid overwriting
    if os.path.exists(DATA_FILE):
        print(f"Database file '{DATA_FILE}' already exists. Skipping generation.")
        return

    # Generate synthetic data
    data = []
    for _ in range(NUM_RECORDS):
        # Generate realistic name
        name = names.get_full_name()
        
        # Generate Indian phone number (+91 prefix)
        phone = f"+91{random.randint(6000000000, 9999999999)}"
        
        # Generate random DOB between 1970 and 2000
        dob = datetime.strftime(
            datetime(1970 + random.randint(0, 30), random.randint(1, 12), random.randint(1, 28)),
            "%Y-%m-%d"
        )
        
        # Generate random address
        address = f"{random.randint(100, 999)} {random.choice(['Main St', 'Park Ave', 'Oak Rd', 'MG Road', 'Church St'])}, {random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata'])}"
        
        # Append record
        data.append({
            "name": name,
            "phone": phone,
            "dob": dob,
            "address": address
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(DATA_FILE, index=False)
    print(f"Sample database created with {NUM_RECORDS} records at '{DATA_FILE}'.")

if __name__ == "__main__":
    generate_synthetic_data()
