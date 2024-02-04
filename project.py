import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from shapely.geometry import Point
from joblib import dump, load  # for model persistence
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Function to generate random coordinates
def generate_coordinates(num_points):
    locations = []

    for _ in range(num_points):
        latitude = 24 + (random.random() * (49 - 24))
        longitude = -125 + (random.random() * (-66 + 125))
        locations.append(Point(longitude, latitude))

    return locations

# Generate random data
random.seed(42)
num_rows = 5

data = {
    'current': [random.randint(100, 150) for _ in range(num_rows)],
    'voltage': [random.randint(200, 250) for _ in range(num_rows)],
    'direction': [random.choice(['North', 'South', 'East', 'West']) for _ in range(num_rows)],
    'fault_history': [random.choice([0, 1]) for _ in range(num_rows)],
    'latitude': [coord.y for coord in generate_coordinates(num_rows)],
    'longitude': [coord.x for coord in generate_coordinates(num_rows)],
    'weather': [random.choice(['Sunny', 'Cloudy', 'Rainy', 'Windy']) for _ in range(num_rows)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('random_data.csv', index=False)

# Load your actual dataset
# Assuming you have a CSV file named 'electrical_distribution_data.csv'
# Adjust the file path accordingly
file_path = 'random_data.csv'

df = pd.read_csv(file_path)

# Data Preprocessing
df.fillna(0, inplace=True)

# Encode categorical data
df = pd.get_dummies(df, columns=['direction', 'weather'])

# Normalize numerical features
scaler = StandardScaler()
numerical_features = ['current', 'voltage', 'latitude', 'longitude']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Convert latitude and longitude to a GeoDataFrame
geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# Machine Learning Model
X = df.drop(['fault_history'], axis=1)
y = df['fault_history']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Feature Importance
feature_importance = model.feature_importances_
print("Feature Importance:")
for feature, importance in zip(X.columns, feature_importance):
    print(f"{feature}: {importance}")

# Save the trained model
model_path = 'model.joblib'
dump(model, model_path)

# Tkinter GUI
root = tk.Tk()
root.title("Fault Detection System")

# GIS Integration
# Plotting the geographical distribution of faults
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Create a Matplotlib Figure and embed it in the Tkinter window
gis_figure, ax = plt.subplots(figsize=(10, 6))
canvas = FigureCanvasTkAgg(gis_figure, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

# Navigation toolbar for zoom functionality
toolbar = NavigationToolbar2Tk(canvas, root)
canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
toolbar.update()

# Function to update GIS plot
def update_gis_plot():
    ax.clear()
    world.plot(ax=ax)
    gdf.plot(ax=ax, marker='o', color='red', markersize=15, alpha=0.5)
    ax.set_title('Geographical Distribution of Faults')
    canvas.draw()

# Button to update GIS plot
update_plot_button = tk.Button(root, text="Update GIS Plot", command=update_gis_plot)
update_plot_button.pack(pady=10)

# Function to open the file dialog
def open_file_dialog():
    global gdf
    file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        df.fillna(0, inplace=True)
        df = pd.get_dummies(df, columns=['direction', 'weather'])
        df[numerical_features] = scaler.transform(df[numerical_features])
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        update_gis_plot()

# Button to open file dialog
file_dialog_button = tk.Button(root, text="Open File Dialog", command=open_file_dialog)
file_dialog_button.pack(pady=10)

# Initial GIS plot
update_gis_plot()

# Run Tkinter event loop
root.mainloop()


