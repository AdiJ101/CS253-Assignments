import csv
import matplotlib.pyplot as plt

def convert_to_numeric(amount):
    if 'Crore' in amount:
        return float(amount.replace(' Crore+', '')) * 10000  # 1 Crore = 10000 lacs
    elif 'Lac' in amount:
        return float(amount.replace(' Lac+', ''))   # Value is already in lacs
    elif 'Thou' in amount:
        return float(amount.replace(' Thou+', '')) / 100  # Convert thousands to lacs
    elif 'Hund' in amount:
        return float(amount.replace(' Hund+', '')) / 10000   # Convert hundreds to lacs
    else:
        return float(amount)

# Initialize an empty dictionary to store party assets
party_assets = {}

# Read data from CSV file
with open('train.csv', mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        party = row["Party"]
        assets = convert_to_numeric(row["Total Assets"])  # Convert assets to numeric value
        if party in party_assets:
            party_assets[party] += assets
        else:
            party_assets[party] = assets

# Calculate total wealth across all parties
total_wealth = sum(party_assets.values())

# Calculate the percentage of wealth each party holds
party_percentages = {party: (assets / total_wealth) * 100 for party, assets in party_assets.items()}

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.bar(party_percentages.keys(), party_percentages.values(), color='skyblue')
plt.title('Percentage Distribution of Parties with Wealthiest Candidates')
plt.xlabel('Party')
plt.ylabel('Percentage of Wealth')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()