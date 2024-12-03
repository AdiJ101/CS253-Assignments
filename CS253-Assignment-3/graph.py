import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV
df = pd.read_csv('train.csv')  # Replace 'your_file.csv' with the path to your CSV file

# Grouping by party and calculating the total criminal cases
party_criminal_cases = df.groupby('Party')['Criminal Case'].sum()

# Sorting parties based on the total criminal cases
party_criminal_cases = party_criminal_cases.sort_values(ascending=False)

# Calculating the percentage distribution
total_cases = party_criminal_cases.sum()
party_percentage = (party_criminal_cases / total_cases) * 100

# Plotting the graph
plt.figure(figsize=(10, 6))
party_percentage.plot(kind='bar', color='skyblue')
plt.title('Percentage Distribution of Parties with Candidates Having Most Criminal Records')
plt.xlabel('Party')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
