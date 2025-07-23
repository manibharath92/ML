import pandas as pd
import numpy as np

num_rows = 100

customer_ids = [f"CUST{str(i)}" for i in range(1, num_rows + 1)]

# Generate random data
data = {
    "customer_id": customer_ids,
    "annual income": np.random.randint(10000, 100000, size=num_rows),
    "spending score": np.random.randint(1, 101, size=num_rows),
    "browsing time": np.round(np.random.uniform(0.5, 10, size=num_rows), 2),
    "frequency purchase": np.random.randint(1, 50, size=num_rows)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel file
filename = "customers.csv"
df.to_csv(filename, index=False)

print(f"Spreadsheet '{filename}' created successfully.")