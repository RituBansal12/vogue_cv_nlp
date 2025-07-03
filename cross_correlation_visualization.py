import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re
from scipy import stats
from scipy.stats import pearsonr
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
    
    # Apparel Category
    for label in [
        "t-shirt", "shirt", "blouse", "mini dress", "midi dress", "maxi dress", "pants", "trousers", "jeans", "shorts", "skirt", "jumpsuit", "romper", "sweater", "hoodie", "jacket", "blazer", "trench coat", "parka", "overcoat", "coat", "cardigan", "swimwear", "bikini", "one-piece", "loungewear", "sleepwear", "activewear", "sportswear", "suit", "set", "co-ord"
    ]:
        attribute_type_map[label] = "Apparel Category"

    # Color
    for label in [
        "red", "blue", "green", "yellow", "pink", "black", "white", "gray", "brown", "beige", "pastel", "neon", "earth tones", "gradient", "ombre", "multicolor", "color-blocked"
    ]:
        attribute_type_map[label] = "Color"

    # Pattern
    for label in [
        "solid", "horizontal stripes", "vertical stripes", "diagonal stripes", "striped", "checked", "plaid", "tartan", "floral", "polka dots", "animal print", "leopard print", "zebra print", "snake print", "camouflage", "abstract print", "geometric print", "tie-dye", "batik", "logo print", "text print"
    ]:
        attribute_type_map[label] = "Pattern"

    # Silhouette / Fit
    for label in [
        "a-line", "bodycon", "straight fit", "column fit", "flared", "wide-leg", "relaxed fit", "loose fit", "slim fit", "tailored fit", "boxy", "wrap style", "empire waist", "peplum"
    ]:
        attribute_type_map[label] = "Silhouette / Fit"

    # Fabric / Material
    for label in [
        "cotton", "denim", "wool", "cashmere", "silk", "satin", "chiffon", "lace", "velvet", "leather", "faux leather", "linen", "jersey", "ribbed", "knit", "sequin", "mesh", "tulle"
    ]:
        attribute_type_map[label] = "Fabric / Material"

    # Design Details
    for label in [
        "ruffles", "frills", "pleats", "embroidery", "appliqué", "beading", "sequins", "cut-outs", "front slit", "side slit", "back slit", "asymmetric details", "belt", "tie", "sash", "buttons", "zippers", "snaps", "visible pockets", "flap pockets", "patch pockets"
    ]:
        attribute_type_map[label] = "Design Details"

    # Sleeve Type
    for label in [
        "sleeveless", "cap sleeves", "short sleeves", "elbow-length sleeves", "long sleeves", "puff sleeves", "bell sleeves", "bishop sleeves", "raglan sleeves", "off-shoulder", "cold-shoulder"
    ]:
        attribute_type_map[label] = "Sleeve Type"

    # Neckline Type
    for label in [
        "round neck", "crew neck", "v-neck", "square neck", "boat neck", "scoop neck", "turtleneck", "mock neck", "halter neck", "cowl neck", "sweetheart neckline", "plunge neckline", "collared"
    ]:
        attribute_type_map[label] = "Neckline Type"

    # Length / Hemline
    for label in [
        "cropped", "waist-length", "hip-length", "knee-length", "midi length", "maxi length", "high-low hem", "asymmetric hem", "scalloped hem"
    ]:
        attribute_type_map[label] = "Length / Hemline"

    # Occasion / Style
    for label in [
        "casual", "formal", "business", "workwear", "evening", "party", "streetwear", "resort", "beachwear", "sporty", "lounge", "sleep", "bridal", "occasion", "outerwear", "layering"
    ]:
        attribute_type_map[label] = "Occasion / Style"
    
    return attribute_type_map

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

def calculate_popularity_series(yearly_df, min_occurrences=5, target_categories=None):
    """Calculate normalized popularity time series for labels in specific categories"""
    if target_categories:
        print(f"Calculating popularity series for labels in categories: {', '.join(target_categories)}...")
    else:
        print("Calculating popularity series for all labels...")
    
    # Calculate total covers per year for normalization
    total_covers_by_year = yearly_df.groupby('year')['count'].sum()
    
    # Filter by target categories if specified
    if target_categories:
        filtered_df = yearly_df[yearly_df['category'].isin(target_categories)]
        labels = filtered_df['label'].unique()
    else:
        labels = yearly_df['label'].unique()
    
    # Create popularity series for each label
    popularity_series = {}
    
    for label in labels:
        if target_categories:
            label_data = filtered_df[filtered_df['label'] == label]
        else:
            label_data = yearly_df[yearly_df['label'] == label]
        
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
    
    print(f"Found {len(popularity_series)} labels with sufficient data")
    
    return popularity_series

def calculate_cross_correlations(popularity_series, max_lag=10):
    """Calculate cross-correlations between all pairs of time series"""
    print("Calculating cross-correlations...")
    
    labels = list(popularity_series.keys())
    n_labels = len(labels)
    
    # Initialize correlation matrices
    correlation_matrix = np.zeros((n_labels, n_labels))
    lag_matrix = np.zeros((n_labels, n_labels))
    p_value_matrix = np.zeros((n_labels, n_labels))
    
    # Calculate correlations for each pair
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            if i == j:
                # Self-correlation
                correlation_matrix[i, j] = 1.0
                lag_matrix[i, j] = 0.0
                p_value_matrix[i, j] = 0.0
            else:
                # Cross-correlation
                series1 = popularity_series[label1]
                series2 = popularity_series[label2]
                
                # Find best correlation and lag
                best_corr = 0.0
                best_lag = 0.0
                best_p_value = 1.0
                
                for lag in range(-max_lag, max_lag + 1):
                    if lag >= 0:
                        # Shift series2 forward
                        if len(series1) > lag:
                            s1 = series1.iloc[:-lag] if lag > 0 else series1
                            s2 = series2.iloc[lag:] if lag > 0 else series2
                        else:
                            continue
                    else:
                        # Shift series1 forward
                        if len(series2) > abs(lag):
                            s1 = series1.iloc[abs(lag):] if abs(lag) > 0 else series1
                            s2 = series2.iloc[:-abs(lag)] if abs(lag) > 0 else series2
                        else:
                            continue
                    
                    # Ensure same length
                    min_len = min(len(s1), len(s2))
                    if min_len < 3:
                        continue
                    
                    s1 = s1.iloc[:min_len]
                    s2 = s2.iloc[:min_len]
                    
                    # Calculate correlation
                    try:
                        corr, p_value = pearsonr(s1, s2)
                        if abs(corr) > abs(best_corr):
                            best_corr = float(corr)
                            best_lag = float(lag)
                            best_p_value = float(p_value)
                    except:
                        continue
                
                correlation_matrix[i, j] = best_corr
                lag_matrix[i, j] = best_lag
                p_value_matrix[i, j] = best_p_value
    
    return correlation_matrix, lag_matrix, p_value_matrix, labels

def find_strong_correlations(correlation_matrix, lag_matrix, p_value_matrix, labels, 
                           min_correlation=0.5, max_p_value=0.05):
    """Find strongly correlated pairs"""
    print("Finding strongly correlated pairs...")
    
    strong_correlations = []
    
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):  # Only upper triangle to avoid duplicates
            corr = correlation_matrix[i, j]
            lag = lag_matrix[i, j]
            p_val = p_value_matrix[i, j]
            
            if abs(corr) >= min_correlation and p_val <= max_p_value:
                strong_correlations.append({
                    'label1': labels[i],
                    'label2': labels[j],
                    'correlation': corr,
                    'lag': lag,
                    'p_value': p_val
                })
    
    # Sort by absolute correlation
    strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return strong_correlations

def plot_correlation_heatmap(correlation_matrix, labels, category_name="All Categories"):
    """Plot correlation heatmap"""
    print(f"Creating correlation heatmap for {category_name}...")
    
    # Create DataFrame for easier plotting
    corr_df = pd.DataFrame(correlation_matrix, index=labels, columns=labels)
    
    # Create the plot
    plt.figure(figsize=(16, 12))
    
    # Create heatmap with consistent theme
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(
        corr_df,
        mask=mask,
        cmap="RdBu_r",  # Red-Blue diverging colormap
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=False,  # Don't show numbers to avoid clutter
        fmt=".2f"
    )
    
    plt.title(f'Cross-Correlation Matrix: {category_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Fashion Labels', fontsize=14)
    plt.ylabel('Fashion Labels', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'visualizations/{category_name.lower().replace(" / ", "_").replace(" ", "_")}_correlation_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_top_correlations(strong_correlations, top_n=15, category_name="All Categories"):
    """Plot top correlations as a bar chart"""
    print(f"Creating top correlations chart for {category_name}...")
    
    if not strong_correlations:
        print("No strong correlations found")
        return
    
    # Take top N correlations and sort in descending order
    top_corrs = strong_correlations[:top_n]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    labels = [f"{corr['label1']} ↔ {corr['label2']}" for corr in top_corrs]
    correlations = [corr['correlation'] for corr in top_corrs]
    lags = [corr['lag'] for corr in top_corrs]
    
    # Use YlGnBu color palette like the cycle lengths plot
    colors = sns.color_palette("YlGnBu", len(top_corrs))
    
    # Create horizontal bar chart
    y_positions = range(len(top_corrs))
    bars = plt.barh(y_positions, correlations, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    # Add labels with consistent styling
    plt.yticks(y_positions, labels, fontsize=11)
    plt.xlabel('Correlation Coefficient', fontsize=14, fontweight='bold')
    plt.title(f'Top {top_n} Fashion Trend Correlations: {category_name}', fontsize=16, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, corr, lag) in enumerate(zip(bars, correlations, lags)):
        lag_text = f" (lag: {lag:+.0f})" if lag != 0 else ""
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{corr:.2f}{lag_text}', va='center', fontweight='bold', fontsize=10)
    
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f'visualizations/{category_name.lower().replace(" / ", "_").replace(" ", "_")}_top_correlations.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def analyze_correlations_by_category(strong_correlations, attribute_type_map):
    """Analyze correlations by category"""
    print("Analyzing correlations by category...")
    
    category_correlations = defaultdict(list)
    
    for corr in strong_correlations:
        label1_cat = attribute_type_map.get(corr['label1'], "Other")
        label2_cat = attribute_type_map.get(corr['label2'], "Other")
        
        # Group by category pairs
        cat_pair = tuple(sorted([label1_cat, label2_cat]))
        category_correlations[cat_pair].append(corr)
    
    # Print summary by category
    print(f"\nCorrelation Summary by Category:")
    for cat_pair, corrs in category_correlations.items():
        avg_corr = np.mean([abs(c['correlation']) for c in corrs])
        print(f"- {cat_pair[0]} ↔ {cat_pair[1]}: {len(corrs)} correlations (avg strength: {avg_corr:.3f})")
    
    return category_correlations

def main():
    """Main function to run the cross-correlation analysis"""
    print("=== Fashion Trend Cross-Correlation Analysis ===\n")
    
    # Define target categories
    target_categories = ["Silhouette / Fit", "Design Details"]
    category_name = "Silhouette vs Design Details"
    
    # Process data
    df = process_data()
    yearly_df, attribute_type_map = analyze_labels_by_year(df)
    
    # Calculate popularity series for target categories only
    popularity_series = calculate_popularity_series(yearly_df, target_categories=target_categories)
    
    if not popularity_series:
        print("No data available for analysis")
        return
    
    # Calculate cross-correlations
    correlation_matrix, lag_matrix, p_value_matrix, labels = calculate_cross_correlations(popularity_series)
    
    # Find strong correlations
    strong_correlations = find_strong_correlations(correlation_matrix, lag_matrix, p_value_matrix, labels)
    
    print(f"\nFound {len(strong_correlations)} strong correlations (|r| ≥ 0.5, p ≤ 0.05)")
    
    if strong_correlations:
        # Plot correlation heatmap
        plot_correlation_heatmap(correlation_matrix, labels, category_name)
        
        # Plot top correlations
        plot_top_correlations(strong_correlations, top_n=15, category_name=category_name)
        
        # Analyze by category
        category_correlations = analyze_correlations_by_category(strong_correlations, attribute_type_map)
        
        # Print top correlations
        print(f"\nTop 10 Strongest Correlations:")
        for i, corr in enumerate(strong_correlations[:10]):
            label1_cat = attribute_type_map.get(corr['label1'], "Other")
            label2_cat = attribute_type_map.get(corr['label2'], "Other")
            lag_text = f" (lag: {corr['lag']:+.0f})" if corr['lag'] != 0 else ""
            print(f"{i+1}. {corr['label1']} ({label1_cat}) ↔ {corr['label2']} ({label2_cat}): {corr['correlation']:.3f}{lag_text}")
        
        # Save results to CSV
        results_df = pd.DataFrame(strong_correlations)
        results_df.to_csv('tabular_data/fashion_trend_correlations.csv', index=False)
        print(f"\nResults saved to: tabular_data/fashion_trend_correlations.csv")
    else:
        print("No strong correlations found with the current thresholds")
    
    print(f"\n{'='*50}")
    print("CROSS-CORRELATION ANALYSIS COMPLETE")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
