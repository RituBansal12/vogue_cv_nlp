import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style to match existing visualizations
plt.style.use('default')
sns.set_theme(style="whitegrid")

def extract_year_from_date(date_str):
    """Extract year from date string like '1990_02_01'"""
    if pd.isna(date_str):
        return None
    match = re.search(r'(\d{4})_', str(date_str))
    return int(match.group(1)) if match else None

def parse_brands(brands_str):
    """Safely parse brand list from string"""
    if pd.isna(brands_str) or brands_str == '[]':
        return []
    try:
        # Handle the string representation of lists
        if isinstance(brands_str, str):
            # Clean up the string and evaluate as literal
            cleaned = brands_str.replace("'", '"').replace('""', '"')
            return ast.literal_eval(cleaned)
        return brands_str
    except:
        return []

def main():
    """Main function to analyze brand popularity over time"""
    print("=== Brand Popularity Analysis (1950-2024, 5-Year Periods) ===\n")
    
    # Load the metadata attributes data
    print("Loading data...")
    df = pd.read_csv('tabular_data/covers_metadata_attributes.csv')
    print(f"Data loaded: {len(df)} covers")
    
    # Extract year from date_column
    print("Extracting years from dates...")
    df['year'] = df['date_column'].apply(extract_year_from_date)
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    
    # Filter to 1950-2024 period
    df = df[(df['year'] >= 1950) & (df['year'] <= 2024)]
    
    print(f"Years range: {df['year'].min()} to {df['year'].max()}")
    print(f"Total covers in 1950-2024: {len(df)}")
    
    # Parse brands for each row
    print("Parsing brand data...")
    df['brands_parsed'] = df['brands'].apply(parse_brands)
    
    # Filter out "Vogue" and "N" from all brand mentions
    print("Filtering out 'Vogue' and 'N' brands...")
    filtered_brands = []
    for brands_list in df['brands_parsed']:
        if isinstance(brands_list, list):
            filtered_list = [brand for brand in brands_list if brand not in ['Vogue', 'N']]
            filtered_brands.extend(filtered_list)
    
    print(f"Total brand mentions (after filtering): {len(filtered_brands)}")
    print(f"Unique brands (after filtering): {len(set(filtered_brands))}")
    
    # Create 5-year periods
    print("Creating 5-year periods...")
    df['period'] = ((df['year'] - df['year'].min()) // 5) * 5 + df['year'].min()
    df['period_end'] = df['period'] + 4
    
    # Calculate brand popularity for each 5-year period
    print("Calculating brand popularity by 5-year periods...")
    period_brand_data = []
    
    for period in sorted(df['period'].unique()):
        period_data = df[df['period'] == period]
        total_covers_period = len(period_data)
        
        if total_covers_period == 0:
            continue
        
        # Count mentions for each brand in this period
        period_brands = Counter()
        for brands_list in period_data['brands_parsed']:
            if isinstance(brands_list, list):
                for brand in brands_list:
                    if brand not in ['Vogue', 'N']:
                        period_brands[brand] += 1
        
        # Find the most popular brand in this period
        if period_brands:
            most_popular_brand, brand_count = period_brands.most_common(1)[0]
            normalized_popularity = brand_count / total_covers_period
            
            period_brand_data.append({
                'period': period,
                'period_end': period + 4,
                'period_label': f"{period}-{period + 4}",
                'most_popular_brand': most_popular_brand,
                'brand_count': brand_count,
                'total_covers': total_covers_period,
                'normalized_popularity': normalized_popularity
            })
    
    period_df = pd.DataFrame(period_brand_data)
    print(f"Created period data for {len(period_df)} 5-year periods")
    
    # Create the visualization
    print(f"\nCreating visualization...")
    plt.figure(figsize=(14, 8))
    
    # Get unique brands to assign consistent colors
    unique_brands = period_df['most_popular_brand'].unique()
    brand_colors = {}
    
    # Use YlGnBu color palette for unique brands
    colors = sns.color_palette("YlGnBu", len(unique_brands))
    for i, brand in enumerate(unique_brands):
        brand_colors[brand] = colors[i]
    
    # Create bar chart
    periods = period_df['period_label']
    popularities = period_df['normalized_popularity']
    brands = period_df['most_popular_brand']
    
    # Assign colors based on brand
    bar_colors = [brand_colors[brand] for brand in brands]
    
    bars = plt.bar(range(len(periods)), popularities, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    # Add labels
    plt.xlabel('Time Period', fontsize=14, fontweight='bold')
    plt.ylabel('Normalized Popularity (mentions per cover)', fontsize=14, fontweight='bold')
    plt.title('Most Popular Brand by Cover Mentions', fontsize=16, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(range(len(periods)), periods.tolist(), rotation=90, ha='center')
    
    # Add brand names on top of bars
    for i, (bar, brand, popularity) in enumerate(zip(bars, brands, popularities)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                brand, ha='center', va='bottom', fontweight='bold', fontsize=8,
                rotation=0)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('visualizations/most_popular_brand_by_mentions.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\nMost Popular Brands by cover mentions:")
    print(f"{'='*70}")
    
    for _, row in period_df.iterrows():
        print(f"\n{row['period_label']}:")
        print(f"  - Most popular brand: {row['most_popular_brand']}")
        print(f"  - Mentions: {row['brand_count']} out of {row['total_covers']} covers")
        print(f"  - Normalized popularity: {row['normalized_popularity']:.3f}")
    
    # Save the period data to CSV
    period_df.to_csv('tabular_data/most_popular_brand_by_mentions.csv', index=False)
    print(f"\nPeriod data saved to: tabular_data/most_popular_brand_by_mentions.csv")
    
    print(f"\n{'='*50}")
    print("BRAND POPULARITY ANALYSIS COMPLETE")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 