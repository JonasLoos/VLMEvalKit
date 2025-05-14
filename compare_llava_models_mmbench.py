import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path('outputs')

def load_model_data(model_name):
    folder_path = OUTPUT_DIR / model_name
    if not folder_path.exists(): return None
    subfolder = [f for f in folder_path.iterdir() if f.is_dir()][0]
    csv_path = subfolder / f'{model_name}_MMBench_DEV_EN_V11_acc.csv'
    if not csv_path.exists(): return None
    df = pd.read_csv(csv_path)
    # Get the first row (since there's only one row of data)
    return df.iloc[0].to_dict()

# Load data for both models
data = {}
for model_name in OUTPUT_DIR.iterdir():
    if not model_name.is_dir(): continue
    x = load_model_data(model_name.name)
    if x: data[model_name.name] = x

# Extract categories and scores
categories = list(list(data.values())[0].keys())[1:]

# Set up the plot
plt.figure(figsize=(len(data) * 4, 6))
x = np.arange(len(categories))
width = 1/(len(data)+1)

# Create bars
for i, model_name in enumerate(data.keys()):
    scores = [float(data[model_name][cat]) for cat in categories]
    plt.bar(x - (len(data)-1)*width/2 + i * width, scores, width, label=model_name)

# Customize the plot
plt.ylabel('Accuracy Score')
plt.title('Comparison of LLaVA Models on MMBench Dataset')
plt.xticks(x, categories, rotation=90)
plt.ylim(0, 1.1)  # Set y-axis limit to show full range
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')

if len(categories) < 8:
    add_labels(plt.gca().patches)

# Add legend
plt.legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig('llava_comparison_mmbench.png', dpi=300, bbox_inches='tight')
plt.close()
