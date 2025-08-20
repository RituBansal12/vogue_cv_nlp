# Vogue Covers: Fashion Trends (1935–2025)

Analyze a century of Vogue covers to discover apparel attributes, brand mentions, cyclic fashion patterns, and co-moving trends using image understanding and NLP.

## Table of Contents

1. [Overview](#overview)
2. [Articles / Publications](#articles--publications)
3. [Project Workflow](#project-workflow)
4. [File Structure](#file-structure)
5. [Data Directory](#data-directory)
6. [Visualizations / Outputs](#visualizations--outputs)
7. [Key Concepts / Variables](#key-concepts--variables)
8. [Installation and Setup](#installation-and-setup)
9. [Usage](#usage)
10. [Results / Interpretation](#results--interpretation)
11. [Technical Details](#technical-details)
12. [Dependencies](#dependencies)
13. [Notes / Limitations](#notes--limitations)
14. [Contributing](#contributing)
15. [Data License](#data-license)

---

## Overview

* **Goal**: Decode long-term fashion trends from Vogue cover images and captions (June 1935–June 2025). Identify recurring apparel attributes, brand popularity, and co-trending styles.
* **Approach**: Extract images and metadata from scanned PDFs; apply zero-shot image tagging (FashionCLIP) and NLP (Transformers NER, KeyBERT, spaCy) on captions; analyze time series via autocorrelation and cross-correlation; visualize results.
* **Highlights**:
  - End-to-end, scriptable pipeline from raw PDFs to CSVs and publication-quality figures.
  - Interpretable attribute taxonomy (silhouette, design details, color, etc.).
  - Outputs saved in `tabular_data/` and `visualizations/` for reproducibility.

---

## Articles / Publications

* Medium: https://medium.com/@ritu.bansalrb00/forecasting-fashion-using-data-science-to-decode-a-century-of-vogue-history-06e00ef27782

---

## Project Workflow

1. **Data Collection / Extraction**
   * Source: Vogue Archive via ProQuest (Toronto Public Library access).
   * Place downloaded PDFs in `raw_data/`.
2. **Data Preprocessing / Cleaning**
   * `extract_covers.py`: extract cover images from PDFs and associate to nearest date text.
   * `extract_metadata.py`: parse Volume, Issue, Year, Editor, Photographer, Caption, etc. into CSV.
3. **Modeling / Analysis**
   * `extract_cover_attributes.py`: zero-shot apparel attribute tagging with FashionCLIP.
   * `extract_metadata_attributes.py`: NER (brands/people), keyphrase and noun-phrase extraction from captions.
   * `fashion_cycle_visualization.py`: autocorrelation-based cycle length analysis for styles.
   * `cross_correlation_visualization.py`: cross-correlation among styles to find co-trends.
   * `brand_popularity_analysis.py`: 5-year period top brands by cover mentions.
4. **Evaluation / Validation**
   * Normalization by total covers per year; minimum-occurrence thresholds; significance filters (e.g., |r| ≥ 0.5, p ≤ 0.05 in cross-correlation).
5. **Visualization / Reporting**
   * Static PNGs saved to `visualizations/`; CSV summaries saved to `tabular_data/`.

---

## File Structure

### Core Scripts

#### `extract_covers.py`
* Purpose: Extract cover images from PDFs and associate each image with the closest date found on nearby pages.
* Input: `raw_data/*.pdf`
* Output: `covers/cover_YYYY_MM_DD.jpg`
* Key Features: PyMuPDF + pdfplumber parsing; regex date extraction; nearest-date association; JPEG export.

#### `extract_metadata.py`
* Purpose: Extract textual metadata from PDFs into a structured table.
* Input: `raw_data/*.pdf`
* Output: `tabular_data/covers_metadata.csv`
* Key Features: Splits by “document x of x”; regex patterns for Volume, Issue, Publication Year, Editor, Photographer, Caption, Retail information; writes CSV.

#### `extract_cover_attributes.py`
* Purpose: Zero-shot apparel attribute tagging for cover images using FashionCLIP.
* Input: `covers/*.jpg`
* Output: `tabular_data/covers_attributes.csv`
* Key Features: Predefined prompt taxonomy (category/color/pattern/silhouette/material/details/sleeve/neckline/length/occasion); cosine similarity scoring; stores `top_labels` and `top_scores`.

#### `extract_metadata_attributes.py`
* Purpose: NLP on captions/retail text to extract brands, people, and outfit descriptors.
* Input: `tabular_data/covers_metadata.csv`
* Output: `tabular_data/covers_metadata_attributes.csv`
* Key Features: Transformers NER (`dslim/bert-base-NER`); KeyBERT keyphrases; spaCy `en_core_web_sm` noun phrases; aggregation into lists per cover.

#### `fashion_cycle_visualization.py`
* Purpose: Measure cyclicity of styles via autocorrelation and visualize cycle lengths.
* Input: `tabular_data/covers_attributes.csv`
* Output: `visualizations/*_cycle_lengths.png`, `tabular_data/*_fashion_cycles.csv`
* Key Features: Year extraction from filenames; per-year label counts; normalization; simple ACF; significant-peak detection; bar plots with summary stats.

#### `cross_correlation_visualization.py`
* Purpose: Find co-moving style pairs and lags via cross-correlation.
* Input: `tabular_data/covers_attributes.csv`
* Output: `visualizations/*_correlation_heatmap.png`, `visualizations/*_top_correlations.png`, `tabular_data/fashion_trend_correlations.csv`
* Key Features: Category mapping; normalized popularity series; Pearson r across lags; p-values; heatmap and ranked bars.

#### `brand_popularity_analysis.py`
* Purpose: Identify the most-mentioned brand in rolling 5-year periods.
* Input: `tabular_data/covers_metadata_attributes.csv`
* Output: `visualizations/most_popular_brand_by_mentions.png`, `tabular_data/most_popular_brand_by_mentions.csv`
* Key Features: Extract year from `date_column`; parse brand lists; filter noise (e.g., “Vogue”, “N”); normalized popularity per period.

#### `eda.ipynb`
* Purpose: Exploratory analysis indicating cross-correlation and cyclic trends.

---

## Data Directory

The repository uses the following folders:

* **`raw_data/`**: Original Vogue Archive PDFs (place your downloads here). Example: `file_1.pdf`, `sample.pdf`.
* **`covers/`**: Generated cover images from `extract_covers.py` (e.g., `cover_1990_02_01.jpg`).
* **`tabular_data/`**: Processed CSVs, e.g.
  - `covers_metadata.csv`
  - `covers_attributes.csv`
  - `covers_metadata_attributes.csv`
  - `silhouette_fit_fashion_cycles.csv`, `design_details_fashion_cycles.csv`
  - `fashion_trend_correlations.csv`
  - `most_popular_brand_by_mentions.csv`
* **`visualizations/`**: Saved figures, e.g.
  - `silhouette_fit_cycle_lengths.png`
  - `design_details_cycle_lengths.png`
  - `silhouette_vs_design_details_correlation_heatmap.png`
  - `silhouette_vs_design_details_top_correlations.png`
  - `most_popular_brand_by_mentions.png`
  - `fit_popularity_over_time.png`, `pattern_popularity_over_time.png`

---

## Visualizations / Outputs

* **Static Figures**: PNG plots in `visualizations/` (cycle lengths, correlation heatmaps, top correlations, brand popularity).
* **Tabular Results**: CSVs in `tabular_data/` for cycles, correlations, and brand popularity.

---

## Key Concepts / Variables

* **`date_column`**: Parsed date key (e.g., `1990_02_01`).
* **`top_labels` / `top_scores`**: Zero-shot attribute labels and cosine scores per cover.
* **Categories**: Apparel Category, Color, Pattern, Silhouette / Fit, Fabric / Material, Design Details, Sleeve Type, Neckline Type, Length / Hemline, Occasion / Style.
* **NLP Fields**: `brands`, `people`, `outfit_descriptions` extracted from caption context.

---

## Installation and Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd vogue_covers
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Then download spaCy English model(s):
   ```bash
   python -m spacy download en_core_web_sm
   # (optional) python -m spacy download en_core_web_trf
   ```
3. **Prepare data**
   * Place Vogue PDFs in `raw_data/`.
   * Ensure internet access for first-time model downloads (FashionCLIP, Transformers, sentence-transformers).

---

## Usage

### Run Complete Pipeline

```bash
# 1) Extract images and dates
python extract_covers.py

# 2) Extract textual metadata
python extract_metadata.py

# 3) Zero-shot apparel attributes from images
python extract_cover_attributes.py

# 4) NLP on captions to extract brands/people/outfits
python extract_metadata_attributes.py

# 5) Cycle length analysis and plots
python fashion_cycle_visualization.py

# 6) Cross-correlation analysis and plots
python cross_correlation_visualization.py

# 7) Brand popularity by 5-year periods
python brand_popularity_analysis.py
```

Outputs will be written to `tabular_data/` and `visualizations/`.

---

## Results / Interpretation

* **Cycles**: Autocorrelation reveals recurring style cycles (e.g., silhouettes/design details) and their average lengths.
* **Co-trends**: Significant cross-correlations (|r| ≥ 0.5 with p ≤ 0.05) surface style pairs that rise/fall together, possibly with lags.
* **Brands**: Period-wise most-mentioned brands from caption text provide a high-level popularity view.

---

## Technical Details

* **Algorithms / Models**: Zero-shot image-text similarity (FashionCLIP), Transformers NER, KeyBERT keyword extraction, spaCy noun phrases, autocorrelation and Pearson cross-correlation.
* **Frameworks / Tools**: pandas, numpy, PyMuPDF, pdfplumber, Pillow, Fashion-CLIP, transformers, sentence-transformers, keybert, spaCy, matplotlib, seaborn, SciPy.
* **Implementation Notes**: Normalization by covers/year; minimum-occurrence filters; significance thresholds; consistent visual style and saved artifacts.

---

## Dependencies

See `requirements.txt`. Key libraries:

* pandas, numpy, matplotlib, seaborn, scipy
* PyMuPDF (`fitz`), pdfplumber, Pillow
* transformers, sentence-transformers, keybert, spaCy (+ `en_core_web_sm` model)
* fashion-clip

---

## Notes / Limitations

* OCR/parse noise in captions can introduce artifacts (e.g., subword tokens like `##an` or partial names in NER outputs).
* Date extraction may fail on pages without clear date text; some images may be tagged as unknown.
* Zero-shot attribute tagging is approximate and prompt-sensitive; labels reflect similarity, not ground-truth annotation.
* Brand lists parsed from free text can include non-brand terms; simple filtering is applied.

---

## Contributing

Contributions welcome via PRs/issues. Please describe changes and include before/after samples for new analyses or visualizations.

---

## Data License

Data is sourced from Vogue Archive via ProQuest; ensure usage complies with their terms.
