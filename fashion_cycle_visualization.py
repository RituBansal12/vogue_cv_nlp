import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style to match existing visualizations
plt.style.use('default')
sns.set_theme(style="whitegrid")

def extract_year_from_image(image_name):
    """Extract year from image filename like 'cover_2000_01_01.jpg'"""
    match = re.search(r'cover_(\d{4})_', image_name)
    return int(match.group(1)) if match else None

def create_attribute_type_mapping():
    """Create mapping from labels to categories based on the provided mapping"""
    attribute_type_map = {}
    
    # Silhouette / Fit
    for label in [
        "a-line", "bodycon", "straight fit", "column fit", "flared", "wide-leg", "relaxed fit", "loose fit", "slim fit", "tailored fit", "boxy", "wrap style", "empire waist", "peplum"
    ]:
        attribute_type_map[label] = "Silhouette / Fit"

    # Design Details
    for label in [
        "ruffles", "frills", "pleats", "embroidery", "appliqué", "beading", "sequins", "cut-outs", "front slit", "side slit", "back slit", "asymmetric details", "belt", "tie", "sash", "buttons", "zippers", "snaps", "visible pockets", "flap pockets", "patch pockets"
    ]:
        attribute_type_map[label] = "Design Details"
    
    return attribute_type_map

def simple_acf(series, nlags):
    """Calculate autocorrelation function using numpy"""
    n = len(series)
    acf_values = []
    
    for lag in range(nlags + 1):
        if lag >= n:
            break
        
        # Calculate correlation for this lag
        numerator = 0
        denominator = 0
        
        for i in range(n - lag):
            numerator += (series.iloc[i] - series.mean()) * (series.iloc[i + lag] - series.mean())
            denominator += (series.iloc[i] - series.mean()) ** 2
        
        if denominator != 0:
            acf_values.append(numerator / denominator)
        else:
            acf_values.append(0)
    
    return np.array(acf_values)

def process_data():
    """Load and process the covers attributes data"""
    print("Loading data...")
    df = pd.read_csv('tabular_data/covers_attributes.csv')
    
    # Extract year from image column
    df['year'] = df['image'].apply(extract_year_from_image)
    
    # Remove rows where year couldn't be extracted
    df = df.dropna(subset=['year'])
    
    # Convert year to int
    df['year'] = df['year'].astype(int)
    
    print(f"Data loaded: {len(df)} covers from {df['year'].min()} to {df['year'].max()}")
    
    return df

def analyze_labels_by_year(df):
    """Analyze label counts by year and map to categories"""
    print("Analyzing labels by year...")
    
    # Create attribute type mapping
    attribute_type_map = create_attribute_type_mapping()
    
    # Dictionary to store label counts by year
    yearly_labels = defaultdict(Counter)
    
    for _, row in df.iterrows():
        year = row['year']
        labels = row['top_labels'].split('; ')
        
        for label in labels:
            label = label.strip()
            yearly_labels[year][label] += 1
    
    # Convert to DataFrame
    yearly_data = []
    for year in sorted(yearly_labels.keys()):
        for label, count in yearly_labels[year].items():
            category = attribute_type_map.get(label, "Other")
            yearly_data.append({
                'year': year,
                'label': label,
                'count': count,
                'category': category
            })
    
    yearly_df = pd.DataFrame(yearly_data)
    
    print(f"Processed {len(yearly_df)} label-year combinations across {yearly_df['category'].nunique()} categories")
    
    return yearly_df, attribute_type_map

def calculate_popularity_series(yearly_df, category, min_occurrences=5):
    """Calculate normalized popularity time series for a category"""
    print(f"Calculating popularity series for {category}...")
    
    # Filter for the specific category
    category_data = yearly_df[yearly_df['category'] == category].copy()
    
    if category_data.empty:
        print(f"No data found for category: {category}")
        return None
    
    # Get unique labels in this category
    labels = category_data['label'].unique()
    
    # Calculate total covers per year for normalization
    total_covers_by_year = yearly_df.groupby('year')['count'].sum()
    
    # Create popularity series for each label
    popularity_series = {}
    
    for label in labels:
        label_data = category_data[category_data['label'] == label]
        
        # Calculate total occurrences across all years
        total_occurrences = label_data['count'].sum()
        
        # Only include labels with minimum occurrences
        if total_occurrences >= min_occurrences:
            # Create time series
            series = label_data.set_index('year')['count'].reindex(
                range(yearly_df['year'].min(), yearly_df['year'].max() + 1), 
                fill_value=0
            )
            
            # Normalize by total covers per year
            normalized_series = series / total_covers_by_year.reindex(series.index, fill_value=1)
            
            popularity_series[label] = normalized_series
    
    print(f"Found {len(popularity_series)} labels with sufficient data for {category}")
    
    return popularity_series

def calculate_autocorrelation(series, max_lag=20):
    """Calculate autocorrelation for a time series"""
    if len(series) < max_lag + 1:
        max_lag = len(series) - 1
    
    if max_lag <= 0:
        return None
    
    # Remove NaN values
    series_clean = series.dropna()
    
    if len(series_clean) < 3:
        return None
    
    try:
        # Calculate autocorrelation using our simple function
        acf_values = simple_acf(series_clean, max_lag)
        return acf_values
    except:
        return None

def analyze_fashion_cycles(popularity_series, category_name):
    """Analyze fashion cycles using autocorrelation with multiple peaks"""
    print(f"Analyzing fashion cycles for {category_name}...")
    
    results = []
    
    for label, series in popularity_series.items():
        # Calculate autocorrelation
        acf_values = calculate_autocorrelation(series)
        
        if acf_values is not None and len(acf_values) > 1:
            # Find peaks after lag 1 (to avoid the lag 0 peak)
            acf_without_lag0 = acf_values[1:]
            
            if acf_without_lag0.size > 0:
                # Find all significant peaks (autocorrelation > 0.3)
                significant_peaks = np.where(acf_without_lag0 > 0.3)[0]
                
                if significant_peaks.size > 0:
                    # Convert to actual lags (add 1 because we removed lag 0)
                    peak_lags = significant_peaks + 1
                    
                    # Calculate average cycle length from all peaks
                    avg_cycle_length = np.mean(peak_lags)
                    
                    # Calculate cycle length variability (standard deviation)
                    cycle_std = np.std(peak_lags)
                    
                    # Number of peaks found
                    num_peaks = len(peak_lags)
                    
                    # Store all peak lags for analysis
                    peak_lags_str = ', '.join(map(str, peak_lags))
                    
                else:
                    avg_cycle_length = None
                    cycle_std = None
                    num_peaks = 0
                    peak_lags_str = ""
                
                # Calculate average popularity
                avg_popularity = series.mean()
                
                # Calculate trend (slope of linear regression)
                years = np.arange(len(series))
                slope, _, _, _, _ = stats.linregress(years, series.fillna(0))
                
                results.append({
                    'label': label,
                    'avg_cycle_length': avg_cycle_length,
                    'cycle_std': cycle_std,
                    'num_peaks': num_peaks,
                    'peak_lags': peak_lags_str,
                    'avg_popularity': avg_popularity,
                    'trend': slope,
                    'max_acf': np.max(acf_without_lag0),
                    'series': series
                })
    
    return pd.DataFrame(results)

def plot_cycle_length_bars(results_df, category_name):
    """Create bar chart of cycle lengths for a category"""
    if results_df.empty:
        print(f"No results to plot for {category_name}")
        return
    
    # Filter for labels with significant cycles
    cycle_data = results_df.dropna(subset=['avg_cycle_length']).copy()
    
    if cycle_data.empty:
        print(f"No significant cycles found for {category_name}")
        return
    
    # Sort by average cycle length
    cycle_data = cycle_data.sort_values('avg_cycle_length', ascending=True)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart with consistent theme
    y_positions = range(len(cycle_data))
    cycle_lengths = cycle_data['avg_cycle_length']
    
    # Use a color palette similar to existing visualizations
    colors = sns.color_palette("YlGnBu", len(cycle_data))
    
    bars = plt.barh(y_positions, cycle_lengths, 
                   color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    # Add labels with consistent styling
    plt.yticks(y_positions, cycle_data['label'], fontsize=11)
    plt.xlabel('Average Cycle Length (years)', fontsize=14, fontweight='bold')
    plt.title(f'Fashion Cycle Lengths: {category_name}', fontsize=16, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, cycle_length, num_peaks) in enumerate(zip(bars, cycle_lengths, cycle_data['num_peaks'])):
        plt.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                f'{cycle_length:.1f}y ({num_peaks} peaks)', va='center', fontweight='bold', fontsize=10)
    
    # Add average line with consistent styling
    avg_cycle = cycle_lengths.mean()
    plt.axvline(avg_cycle, color='#d62728', linestyle='--', linewidth=2, 
                label=f'Average: {avg_cycle:.1f} years')
    
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f'visualizations/{category_name.lower().replace(" / ", "_").replace(" ", "_")}_cycle_lengths.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\nCycle Length Summary for {category_name}:")
    print(f"- Average cycle length: {avg_cycle:.1f} years")
    print(f"- Total styles with cycles: {len(cycle_data)}")
    print(f"- Average number of peaks per style: {cycle_data['num_peaks'].mean():.1f}")
    print(f"- Average cycle length variability (std): {cycle_data['cycle_std'].mean():.1f} years")
    
    # Print individual style details
    print(f"\nIndividual Style Details:")
    for _, row in cycle_data.iterrows():
        print(f"  - {row['label']}: {row['avg_cycle_length']:.1f} years ({row['num_peaks']} peaks)")
    
    return cycle_data

def main():
    """Main function to run the fashion cycle visualization"""
    print("=== Fashion Cycle Length Visualization ===\n")
    
    # Process data
    df = process_data()
    yearly_df, attribute_type_map = analyze_labels_by_year(df)
    
    # Focus on Silhouette and Design categories
    target_categories = ["Silhouette / Fit", "Design Details"]
    
    for category in target_categories:
        print(f"\n{'='*50}")
        print(f"ANALYZING: {category}")
        print(f"{'='*50}")
        
        # Calculate popularity series
        popularity_series = calculate_popularity_series(yearly_df, category)
        
        if popularity_series:
            # Analyze fashion cycles
            results_df = analyze_fashion_cycles(popularity_series, category)
            
            if not results_df.empty:
                print(f"\nResults for {category}:")
                print(f"- Total labels analyzed: {len(results_df)}")
                print(f"- Labels with significant cycles: {len(results_df.dropna(subset=['avg_cycle_length']))}")
                
                # Create cycle length visualization
                cycle_data = plot_cycle_length_bars(results_df, category)
                
                # Save results to CSV
                results_df.to_csv(f'tabular_data/{category.lower().replace(" / ", "_").replace(" ", "_")}_fashion_cycles.csv', index=False)
                print(f"\nResults saved to: tabular_data/{category.lower().replace(' / ', '_').replace(' ', '_')}_fashion_cycles.csv")
            else:
                print(f"No significant results found for {category}")
        else:
            print(f"No data available for {category}")
    
    print(f"\n{'='*50}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 