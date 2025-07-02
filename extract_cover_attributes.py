import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from fashion_clip.fashion_clip import FashionCLIP
import numpy as np

# Initialize FashionCLIP
fclip = FashionCLIP('fashion-clip')

# Full prompt list formatted for Python
prompts = [
    # Apparel Category
    "t-shirt", "shirt", "blouse", "mini dress", "midi dress", "maxi dress", "pants", "trousers", "jeans", "shorts", "skirt", "jumpsuit", "romper", "sweater", "hoodie", "jacket", "blazer", "trench coat", "parka", "overcoat", "coat", "cardigan", "swimwear", "bikini", "one-piece", "loungewear", "sleepwear", "activewear", "sportswear", "suit", "set", "co-ord",
    # Color
    "red", "blue", "green", "yellow", "pink", "black", "white", "gray", "brown", "beige", "pastel", "neon", "earth tones", "gradient", "ombre", "multicolor", "color-blocked",
    # Pattern
    "solid", "horizontal stripes", "vertical stripes", "diagonal stripes", "striped", "checked", "plaid", "tartan", "floral", "polka dots", "animal print", "leopard print", "zebra print", "snake print", "camouflage", "abstract print", "geometric print", "tie-dye", "batik", "logo print", "text print",
    # Silhouette / Fit
    "a-line", "bodycon", "straight fit", "column fit", "flared", "wide-leg", "relaxed fit", "loose fit", "slim fit", "tailored fit", "boxy", "wrap style", "empire waist", "peplum",
    # Fabric / Material
    "cotton", "denim", "wool", "cashmere", "silk", "satin", "chiffon", "lace", "velvet", "leather", "faux leather", "linen", "jersey", "ribbed", "knit", "sequin", "mesh", "tulle",
    # Design Details
    "ruffles", "frills", "pleats", "embroidery", "appliqué", "beading", "sequins", "cut-outs", "front slit", "side slit", "back slit", "asymmetric details", "belt", "tie", "sash", "buttons", "zippers", "snaps", "visible pockets", "flap pockets", "patch pockets",
    # Sleeve Type
    "sleeveless", "cap sleeves", "short sleeves", "elbow-length sleeves", "long sleeves", "puff sleeves", "bell sleeves", "bishop sleeves", "raglan sleeves", "off-shoulder", "cold-shoulder",
    # Neckline Type
    "round neck", "crew neck", "v-neck", "square neck", "boat neck", "scoop neck", "turtleneck", "mock neck", "halter neck", "cowl neck", "sweetheart neckline", "plunge neckline", "collared",
    # Length / Hemline
    "cropped", "waist-length", "hip-length", "knee-length", "midi length", "maxi length", "high-low hem", "asymmetric hem", "scalloped hem",
    # Occasion / Style
    "casual", "formal", "business", "workwear", "evening", "party", "streetwear", "resort", "beachwear", "sporty", "lounge", "sleep", "bridal", "occasion", "outerwear", "layering"
]

COVERS_DIR = "covers"
TABULAR_DATA_DIR = "tabular_data"
os.makedirs(TABULAR_DATA_DIR, exist_ok=True)
CSV_FILE = os.path.join(TABULAR_DATA_DIR, "covers_attributes.csv")

rows = []
TOP_N = 15

# Precompute text features for all prompts
text_features = fclip.encode_text(prompts, batch_size=32)
text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

for fname in tqdm(os.listdir(COVERS_DIR), desc="Analyzing covers"):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    image_path = os.path.join(COVERS_DIR, fname)
    image = Image.open(image_path).convert("RGB")
    image_features = fclip.encode_images([image], batch_size=1)[0]
    image_features = image_features / np.linalg.norm(image_features)
    # Compute cosine similarity
    scores = np.dot(text_features, image_features)
    # Get indices of top N scores
    top_indices = np.argsort(scores)[-TOP_N:][::-1]
    top_labels = [prompts[i] for i in top_indices]
    top_scores = [float(scores[i]) for i in top_indices]
    rows.append({
        "image": fname,
        "top_labels": "; ".join(top_labels),
        "top_scores": "; ".join([f"{s:.3f}" for s in top_scores])
    })

df = pd.DataFrame(rows)
df.to_csv(CSV_FILE, index=False)
print(f"Saved top {TOP_N} attributes to {CSV_FILE}") 