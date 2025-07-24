import pandas as pd
import re

# Read the Excel file
file_path = 'physical therap robot OR rehab robot REVIEW 1-1281.xlsx'
df = pd.read_excel(file_path)

# Display basic information about the file
print(f"Number of papers: {len(df)}")
print(f"Year range: {df['Publication Year'].min()} - {df['Publication Year'].max()}")
print(f"Number of papers with abstract: {df['Abstract'].notna().sum()}")

# Clean and handle missing values
columns_to_clean = ['Article Title', 'Abstract', 'Author Keywords', 'Keywords Plus']
for col in columns_to_clean:
    if col in df.columns:
        df[col] = df[col].fillna('')

# Convert text to lowercase for consistent searching
df['Title_Lower'] = df['Article Title'].str.lower()
df['Abstract_Lower'] = df['Abstract'].str.lower()
df['Keywords_Lower'] = df['Author Keywords'].str.lower() + ' ' + df['Keywords Plus'].str.lower()

# Create a combined text column for keyword searching
df['Combined_Text'] = df['Title_Lower'] + ' ' + df['Abstract_Lower'] + ' ' + df['Keywords_Lower']

# Define keywords for main categories
categories = {
    'Upper Limb': ['upper limb', 'upper extremity', 'arm', 'hand', 'wrist', 'elbow', 'shoulder',
                   'finger', 'grasp', 'reaching', 'pinch', 'upper-limb'],

    'Lower Limb': ['lower limb', 'lower extremity', 'leg', 'foot', 'ankle', 'knee', 'hip', 'gait',
                   'walking', 'locomotion', 'stance', 'step', 'treadmill', 'lower-limb', 'walking robot'],

    # General Rehabilitation will include all papers not classified in previous categories
}

# Define keywords for subcategories
subcategories = {
    'Socially Assistive Robots': ['social', 'socially assistive', 'assistive robot', 'human-robot interaction',
                                  'companion robot', 'social interaction', 'social robot', 'emotional', 'engagement'],

    'Wearable Robotics': ['wearable', 'exoskeleton', 'exosuit', 'orthosis', 'orthotic', 'soft robot',
                          'textile', 'glove', 'suit', 'portable', 'body-worn'],

    'Virtual Reality': ['virtual reality', 'vr', 'augmented reality', 'ar', 'mixed reality',
                        'simulation', 'game', 'gaming', 'virtual environment'],

    'Telerehabilitation': ['telerehab', 'tele-rehab', 'remote', 'home-based', 'teletherapy',
                           'telemedicine', 'telehealth', 'home rehabilitation'],

    'Neurological Rehabilitation': ['stroke', 'brain injury', 'spinal cord', 'parkinson', 'cerebral palsy',
                                    'multiple sclerosis', 'neurological', 'neuromuscular', 'neural'],
}


# Function to identify main category
def identify_main_category(text):
    for category, keywords in categories.items():
        for keyword in keywords:
            # More precise search with word boundaries
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text):
                return category
    return "General Rehabilitation"


# Function to identify subcategories (can be more than one)
def identify_subcategories(text):
    found_subcategories = []
    for subcategory, keywords in subcategories.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text):
                found_subcategories.append(subcategory)
                break  # If we found one match for a subcategory, no need to search for more keywords
    return list(set(found_subcategories))  # Remove duplicates


# Classify papers into main categories
df['Main_Category'] = df['Combined_Text'].apply(identify_main_category)

# Classify papers into subcategories (can be more than one)
df['Subcategories'] = df['Combined_Text'].apply(identify_subcategories)

# Create binary columns for each subcategory
for subcategory in subcategories.keys():
    df[subcategory] = df['Subcategories'].apply(lambda x: 1 if subcategory in x else 0)

# Analysis 1: Distribution of main categories
main_category_counts = df['Main_Category'].value_counts().reset_index()
main_category_counts.columns = ['Category', 'Number of Papers']
print("\nDistribution of main categories:")
print(main_category_counts)

# Analysis 2: Distribution of subcategories
subcategory_counts = df[list(subcategories.keys())].sum().reset_index()
subcategory_counts.columns = ['Subcategory', 'Number of Papers']
subcategory_counts = subcategory_counts.sort_values('Number of Papers', ascending=False)
print("\nDistribution of subcategories:")
print(subcategory_counts)

# Analysis 3: Distribution of subcategories within each main category
subcategory_by_main = {}
for main_cat in df['Main_Category'].unique():
    subcategory_by_main[main_cat] = df[df['Main_Category'] == main_cat][list(subcategories.keys())].sum()

subcategory_by_main_df = pd.DataFrame(subcategory_by_main).T
print("\nDistribution of subcategories within each main category:")
print(subcategory_by_main_df)

# Analysis 4: Trends over time - number of papers by year
yearly_counts = df.groupby('Publication Year').size().reset_index(name='Number of Papers')
print("\nNumber of papers by year:")
print(yearly_counts)

# Analysis 5: Main category trends over time
main_cat_by_year = df.groupby(['Publication Year', 'Main_Category']).size().unstack().fillna(0)
print("\nMain category trends over time:")
print(main_cat_by_year)

# Analysis 6: Subcategory trends over time
subcategories_by_year = df.groupby('Publication Year')[list(subcategories.keys())].sum()
print("\nSubcategory trends over time:")
print(subcategories_by_year)

# Create a directory for results if it doesn't exist
import os

if not os.path.exists('results'):
    os.makedirs('results')

# Create an Excel writer to save all results in different sheets of a single Excel file
with pd.ExcelWriter('results/rehabilitation_robotics_analysis.xlsx') as writer:
    # Save all dataframes to different sheets
    main_category_counts.to_excel(writer, sheet_name='Main Categories', index=False)
    subcategory_counts.to_excel(writer, sheet_name='Subcategories', index=False)
    subcategory_by_main_df.to_excel(writer, sheet_name='Subcats by Main Cat')
    yearly_counts.to_excel(writer, sheet_name='Yearly Trends', index=False)
    main_cat_by_year.to_excel(writer, sheet_name='Main Cat by Year')
    subcategories_by_year.to_excel(writer, sheet_name='Subcats by Year')

    # Also save the classified data
    df_results = df[['Article Title', 'Publication Year', 'Main_Category'] + list(subcategories.keys())]
    df_results.to_excel(writer, sheet_name='Classified Papers', index=False)

print("\nResults saved to Excel file: results/rehabilitation_robotics_analysis.xlsx")
