import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Load the analysis results
# If the Excel file doesn't exist, run the first code before this one
results_file = 'results/rehabilitation_robotics_analysis.xlsx'

# Check if results file exists
if not os.path.exists(results_file):
    print("Error: Could not find the results file. Please run the basic analysis code first.")
    exit()

# Read data from the Excel file
yearly_counts = pd.read_excel(results_file, sheet_name='Yearly Trends')
main_category_counts = pd.read_excel(results_file, sheet_name='Main Categories')
subcategory_counts = pd.read_excel(results_file, sheet_name='Subcategories')
main_cat_by_year = pd.read_excel(results_file, sheet_name='Main Cat by Year', index_col=0)
subcategories_by_year = pd.read_excel(results_file, sheet_name='Subcats by Year', index_col=0)
subcategory_by_main_df = pd.read_excel(results_file, sheet_name='Subcats by Main Cat', index_col=0)
df_results = pd.read_excel(results_file, sheet_name='Classified Papers')

# Read the original data to get citation data if available
df = pd.read_excel('physical therap robot OR rehab robot REVIEW 1-1281.xlsx')

# Set up a professional seaborn style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Create custom colors
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
palette = sns.color_palette(colors)
sns.set_palette(palette)

# Create directory for report REVIEWS_figures if it doesn't exist
if not os.path.exists('report_figures'):
    os.makedirs('report_figures')

# Extract subcategory names from the data
subcategory_list = [col for col in df_results.columns
                    if col not in ['Article Title', 'Publication Year', 'Main_Category']]

print(f"Creating high-quality visualization REVIEWS_figures for {len(yearly_counts)} years of data...")
print(f"Found {len(main_category_counts)} main categories and {len(subcategory_list)} subcategories.")

# ---------------------------------------------------------------------------------
# 1. Line plot with trend line - Number of publications over time
# ---------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
ax = sns.lineplot(x='Publication Year', y='Number of Papers', data=yearly_counts,
                  marker='o', linewidth=3, markersize=8, color=colors[0])

# Add trend line
x = yearly_counts['Publication Year']
y = yearly_counts['Number of Papers']
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "--", linewidth=2, alpha=0.7, color=colors[2])

# Calculate average annual growth rate
if len(y) > 1 and y.iloc[0] > 0:
    growth_rate = ((y.iloc[-1] / y.iloc[0]) ** (1 / (len(y) - 1)) - 1) * 100
    plt.annotate(f'Average Annual Growth: {growth_rate:.1f}%',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=12, ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

# Styling
plt.title('Growth of Rehabilitation Robotics Publications (2000-2025)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Publications', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Save in both PNG and PDF formats for different use cases
plt.savefig('report_figures/publication_growth_trend.png', dpi=300, bbox_inches='tight')
plt.savefig('report_figures/publication_growth_trend.pdf', bbox_inches='tight')
plt.close()
print("Figure 1: Publication growth trend chart created")

# ---------------------------------------------------------------------------------
# 2. Stacked area chart - Evolution of main categories over time
# ---------------------------------------------------------------------------------
plt.figure(figsize=(12, 7))
main_cat_by_year_plot = main_cat_by_year.copy()
main_cat_by_year_plot.plot.area(alpha=0.7, linewidth=2, figsize=(12, 7), cmap='viridis')

# Styling
plt.title('Evolution of Main Research Categories in Rehabilitation Robotics', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Publications', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(title='Category', title_fontsize=12, fontsize=10, loc='upper left')
plt.tight_layout()

plt.savefig('report_figures/main_categories_evolution.png', dpi=300, bbox_inches='tight')
plt.savefig('report_figures/main_categories_evolution.pdf', bbox_inches='tight')
plt.close()
print("Figure 2: Main categories evolution chart created")

# ---------------------------------------------------------------------------------
# 3. Horizontal bar chart - Ranking of subcategories by frequency
# ---------------------------------------------------------------------------------
plt.figure(figsize=(12, 8))

# Sort subcategories by number of papers
subcategory_counts_sorted = subcategory_counts.sort_values('Number of Papers')

# Create barplot - fixed warning by avoiding the palette parameter without hue
ax = sns.barplot(x='Number of Papers', y='Subcategory', data=subcategory_counts_sorted, color=colors[0])

# Add numbers at the end of the bars
for i, v in enumerate(subcategory_counts_sorted['Number of Papers']):
    ax.text(v + 1, i, str(v), va='center', fontweight='bold')

# Styling
plt.title('Ranking of Rehabilitation Robotics Subcategories', fontsize=16, fontweight='bold')
plt.xlabel('Number of Publications', fontsize=14)
plt.ylabel('Subcategory', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig('report_figures/subcategories_ranking.png', dpi=300, bbox_inches='tight')
plt.savefig('report_figures/subcategories_ranking.pdf', bbox_inches='tight')
plt.close()
print("Figure 3: Subcategories ranking bar chart created")

# ---------------------------------------------------------------------------------
# 4. Grouped bar chart - Comparison of subcategories within main categories
# ---------------------------------------------------------------------------------
# Prepare data in the right format for plotting
subcategory_by_main_melted = subcategory_by_main_df.reset_index().melt(
    id_vars='index', var_name='Subcategory', value_name='Count')
subcategory_by_main_melted.rename(columns={'index': 'Main Category'}, inplace=True)

plt.figure(figsize=(14, 8))
ax = sns.barplot(x='Subcategory', y='Count', hue='Main Category',
                 data=subcategory_by_main_melted, palette='viridis')

# Styling
plt.title('Subcategory Distribution Within Main Categories', fontsize=16, fontweight='bold')
plt.xlabel('Subcategory', fontsize=14)
plt.ylabel('Number of Publications', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Main Category', title_fontsize=12, fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig('report_figures/subcategories_by_main_category.png', dpi=300, bbox_inches='tight')
plt.savefig('report_figures/subcategories_by_main_category.pdf', bbox_inches='tight')
plt.close()
print("Figure 4: Subcategories by main category grouped bar chart created")


# ---------------------------------------------------------------------------------
# 5. Heatmap of subcategory combinations - FIXED VERSION
# ---------------------------------------------------------------------------------
def calculate_subcategory_combinations():
    # Initialize empty DataFrame with numeric dtype
    subcategory_combinations = pd.DataFrame(0,
                                            index=subcategory_list,
                                            columns=subcategory_list,
                                            dtype=float)  # Explicitly set dtype to float

    # Count co-occurrences
    for subcat1 in subcategory_list:
        for subcat2 in subcategory_list:
            if subcat1 != subcat2:
                # Count papers that have both subcategories
                combination_count = ((df_results[subcat1] == 1) & (df_results[subcat2] == 1)).sum()
                subcategory_combinations.loc[subcat1, subcat2] = float(combination_count)
            else:
                # Diagonal values are the total count of each subcategory
                subcategory_combinations.loc[subcat1, subcat2] = float(df_results[subcat1].sum())

    return subcategory_combinations


# Calculate the combinations matrix with proper data type
subcategory_combinations = calculate_subcategory_combinations()

plt.figure(figsize=(12, 10))

# Create mask to show only the lower triangle (to avoid redundancy)
mask = np.zeros_like(subcategory_combinations, dtype=bool)
mask[np.triu_indices_from(mask)] = True  # Upper triangle mask

ax = sns.heatmap(subcategory_combinations,
                 annot=True,  # Show numbers in cells
                 fmt='.0f',  # Format as integer without decimal points
                 cmap='YlGnBu',
                 mask=mask,  # Use the mask to show only lower triangle
                 linewidths=0.5)

# Styling
plt.title('Co-occurrence of Subcategories in Rehabilitation Robotics Publications',
          fontsize=16, fontweight='bold')
plt.tight_layout()

plt.savefig('report_figures/subcategory_combinations_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig('report_figures/subcategory_combinations_heatmap.pdf', bbox_inches='tight')
plt.close()
print("Figure 5: Subcategory combinations heatmap created")


print("\nAll high-quality visualization REVIEWS_figures have been saved to the 'report_figures' directory.")
print("PNG files are optimized for screen viewing and PDF files for inclusion in your report document.")