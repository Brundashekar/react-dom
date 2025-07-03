import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set up the style
plt.style.use('default')
sns.set_palette("husl")

# Your correlation data
correlation_data = [
    {'x': 'Communication', 'y': 'Communication', 'value': 1.0, 'label': '1.00'},
    {'x': 'Communication', 'y': 'Aptitude', 'value': 0.99, 'label': '0.99'},
    {'x': 'Communication', 'y': 'Technical', 'value': -0.23, 'label': '-0.23'},
    {'x': 'Communication', 'y': 'Interview_Score', 'value': -0.042, 'label': '-0.042'},
    {'x': 'Aptitude', 'y': 'Communication', 'value': 0.99, 'label': '0.99'},
    {'x': 'Aptitude', 'y': 'Aptitude', 'value': 1.0, 'label': '1.00'},
    {'x': 'Aptitude', 'y': 'Technical', 'value': -0.2, 'label': '-0.20'},
    {'x': 'Aptitude', 'y': 'Interview_Score', 'value': 0.03, 'label': '0.03'},
    {'x': 'Technical', 'y': 'Communication', 'value': -0.23, 'label': '-0.23'},
    {'x': 'Technical', 'y': 'Aptitude', 'value': -0.2, 'label': '-0.20'},
    {'x': 'Technical', 'y': 'Technical', 'value': 1.0, 'label': '1.00'},
    {'x': 'Technical', 'y': 'Interview_Score', 'value': -0.7, 'label': '-0.70'},
    {'x': 'Interview_Score', 'y': 'Communication', 'value': -0.042, 'label': '-0.042'},
    {'x': 'Interview_Score', 'y': 'Aptitude', 'value': 0.03, 'label': '0.03'},
    {'x': 'Interview_Score', 'y': 'Technical', 'value': -0.7, 'label': '-0.70'},
    {'x': 'Interview_Score', 'y': 'Interview_Score', 'value': 1.0, 'label': '1.00'}
]

# Sample raw data
sample_scores = [
    {'id': 1, 'Communication': 85, 'Aptitude': 88, 'Technical': 45, 'Interview_Score': 72},
    {'id': 2, 'Communication': 92, 'Aptitude': 90, 'Technical': 38, 'Interview_Score': 68},
    {'id': 3, 'Communication': 78, 'Aptitude': 82, 'Technical': 55, 'Interview_Score': 75},
    {'id': 4, 'Communication': 88, 'Aptitude': 85, 'Technical': 42, 'Interview_Score': 70},
    {'id': 5, 'Communication': 95, 'Aptitude': 93, 'Technical': 35, 'Interview_Score': 65}
]

# Convert to DataFrames
df_corr = pd.DataFrame(correlation_data)
df_scores = pd.DataFrame(sample_scores)

# Create correlation matrix for heatmap
variables = ['Communication', 'Aptitude', 'Technical', 'Interview_Score']
corr_matrix = np.zeros((4, 4))

for i, var1 in enumerate(variables):
    for j, var2 in enumerate(variables):
        value = df_corr[(df_corr['x'] == var1) & (df_corr['y'] == var2)]['value'].iloc[0]
        corr_matrix[i, j] = value

# Create the correlation matrix DataFrame
corr_df = pd.DataFrame(corr_matrix, index=variables, columns=variables)

# Create the dashboard
fig = plt.figure(figsize=(16, 12))

# Main title
fig.suptitle('Candidate Assessment Scores Correlation Dashboard', fontsize=20, fontweight='bold', y=0.95)

# 1. Correlation Heatmap (main chart)
ax1 = plt.subplot(2, 3, (1, 2))
sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0,
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8},
            linewidths=0.5)
ax1.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('')
ax1.set_ylabel('')

# 2. Average Scores Bar Chart
ax2 = plt.subplot(2, 3, 3)
avg_scores = df_scores[variables].mean()
bars = ax2.bar(range(len(variables)), avg_scores.values, color=['#3b82f6', '#10b981', '#f59e0b', '#ef4444'])
ax2.set_title('Average Scores', fontsize=12, fontweight='bold')
ax2.set_xlabel('Assessment Type')
ax2.set_ylabel('Average Score')
ax2.set_xticks(range(len(variables)))
ax2.set_xticklabels([var.replace('_', ' ') for var in variables], rotation=45, ha='right')

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

# 3. Score Distribution
ax3 = plt.subplot(2, 3, 4)
df_scores_melted = df_scores.melt(id_vars=['id'], value_vars=variables,
                                  var_name='Assessment', value_name='Score')
sns.boxplot(data=df_scores_melted, x='Assessment', y='Score', ax=ax3)
ax3.set_title('Score Distribution', fontsize=12, fontweight='bold')
ax3.set_xticklabels([var.replace('_', ' ') for var in variables], rotation=45, ha='right')

# 4. Key Insights Text Box
ax4 = plt.subplot(2, 3, 5)
ax4.axis('off')
insights_text = """
KEY INSIGHTS

Strong Positive Correlation:
• Communication & Aptitude: 0.99
  Very high correlation indicates these
  skills often go together

Strong Negative Correlation:
• Technical & Interview Score: -0.70
  Higher technical scores associated
  with lower interview performance

Weak Correlations:
• Most other pairs show weak
  relationships (-0.2 to 0.2 range)
• Communication & Technical: -0.23
• Aptitude & Technical: -0.20

Pattern Analysis:
• Communication and Aptitude skills
  are closely related
• Technical skills seem independent
  of soft skills
"""

ax4.text(0.05, 0.95, insights_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# 5. Individual Scores Line Plot
ax5 = plt.subplot(2, 3, 6)
for idx, row in df_scores.iterrows():
    ax5.plot(variables, [row[var] for var in variables],
             marker='o', linewidth=2, label=f'Candidate {row["id"]}')

ax5.set_title('Individual Score Profiles', fontsize=12, fontweight='bold')
ax5.set_xlabel('Assessment Type')
ax5.set_ylabel('Score')
ax5.set_xticks(range(len(variables)))
ax5.set_xticklabels([var.replace('_', ' ') for var in variables], rotation=45, ha='right')
ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("CORRELATION SUMMARY")
print("="*60)

print(f"\nCorrelation Matrix:")
print(corr_df.round(3))

print(f"\nAverage Scores:")
for var in variables:
    avg = df_scores[var].mean()
    std = df_scores[var].std()
    print(f"{var.replace('_', ' ')}: {avg:.1f} ± {std:.1f}")

print(f"\nStrongest Correlations:")
# Get upper triangle of correlation matrix (excluding diagonal)
mask = np.triu(np.ones_like(corr_df), k=1).astype(bool)
correlations = corr_df.where(mask).stack().reset_index()
correlations.columns = ['Variable1', 'Variable2', 'Correlation']
correlations = correlations.sort_values('Correlation', key=abs, ascending=False)

for _, row in correlations.head(5).iterrows():
    print(f"{row['Variable1']} vs {row['Variable2']}: {row['Correlation']:.3f}")

# Save the data to CSV if needed
print(f"\nSaving data to CSV files...")
df_corr.to_csv('correlation_data.csv', index=False)
df_scores.to_csv('sample_scores.csv', index=False)
print("Files saved: correlation_data.csv, sample_scores.csv")
