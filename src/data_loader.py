"""
data_loader.py
--------------
Synthetic sample-data generator and CSV loader for the
PCA-Based Compliance Risk Clustering project (SIH26051).

All generated data is clearly marked as DEMONSTRATION data and does NOT
represent verified regulatory evidence or real product listings.
"""

from __future__ import annotations

import logging
import random
import io
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = [
    "Electronics", "Clothing & Apparel", "Food & Beverages",
    "Health & Personal Care", "Toys & Games", "Home & Kitchen",
    "Automotive", "Sports & Outdoors", "Books & Stationery",
    "Industrial & Scientific",
]

COUNTRIES = [
    "China", "India", "USA", "Germany", "Japan",
    "South Korea", "Vietnam", "Bangladesh", "Mexico", "Turkey",
]

BRANDS = [
    "BrandA", "BrandB", "BrandC", "BrandX", "NoName",
    "GenericCo", "PrimeProd", "EcoMark", "TechLine", "SafeGoods",
]

CERTIFICATIONS = {
    "Electronics":          ["CE", "FCC", "RoHS", "BIS", "UL"],
    "Food & Beverages":     ["FSSAI", "ISO22000", "HACCP", "Organic", "Halal"],
    "Health & Personal Care": ["FDA", "GMP", "ISO13485", "CE", "BIS"],
    "Toys & Games":         ["EN71", "ASTM", "BIS", "CE", "CPSIA"],
    "Automotive":           ["ECE", "ARAI", "ISO9001", "BIS", "AIS"],
    "Industrial & Scientific": ["ISO9001", "CE", "ATEX", "RoHS", "UL"],
}

ADJECTIVES = [
    "Premium", "Durable", "Eco-Friendly", "Compact", "Portable",
    "Heavy-Duty", "Multi-Purpose", "Smart", "Wireless", "Ultra",
]

NOUNS = {
    "Electronics":          ["Charger", "Speaker", "Cable", "Adapter", "Headphones"],
    "Clothing & Apparel":   ["Shirt", "Jacket", "Jeans", "Dress", "Hoodie"],
    "Food & Beverages":     ["Juice", "Snacks", "Spices", "Sauce", "Tea"],
    "Health & Personal Care": ["Shampoo", "Cream", "Supplement", "Toothpaste", "Soap"],
    "Toys & Games":         ["Puzzle", "Doll", "Car", "Board Game", "Blocks"],
    "Home & Kitchen":       ["Pan", "Blender", "Knife", "Mug", "Organiser"],
    "Automotive":           ["Filter", "Bulb", "Wiper", "Seat Cover", "Cleaner"],
    "Sports & Outdoors":    ["Bottle", "Gloves", "Bag", "Mat", "Shoes"],
    "Books & Stationery":   ["Notebook", "Pen", "Folder", "Planner", "Marker"],
    "Industrial & Scientific": ["Sensor", "Relay", "Cable Tie", "Gauge", "Valve"],
}

# Declarations required per category
REQUIRED_DECLARATIONS: dict[str, list[str]] = {
    "Electronics":          ["voltage_declared", "warranty_declared", "cert_declared"],
    "Food & Beverages":     ["ingredients_declared", "expiry_declared", "license_declared"],
    "Health & Personal Care": ["ingredients_declared", "license_declared", "batch_declared"],
    "Toys & Games":         ["age_group_declared", "cert_declared", "material_declared"],
    "Automotive":           ["compatibility_declared", "cert_declared", "warranty_declared"],
    "Industrial & Scientific": ["cert_declared", "voltage_declared", "safety_class_declared"],
    "Clothing & Apparel":   ["material_declared", "size_guide_declared"],
    "Home & Kitchen":       ["material_declared", "warranty_declared"],
    "Sports & Outdoors":    ["material_declared", "size_guide_declared"],
    "Books & Stationery":   ["publisher_declared"],
}

ALL_DECLARATION_COLS = [
    "voltage_declared", "warranty_declared", "cert_declared",
    "ingredients_declared", "expiry_declared", "license_declared",
    "batch_declared", "age_group_declared", "material_declared",
    "size_guide_declared", "compatibility_declared", "safety_class_declared",
    "publisher_declared",
]

SCHEMA_COLUMNS = [
    "product_id", "category", "title", "description",
    "declared_price", "declared_brand", "manufacturer", "country_of_origin",
    "certifications",
    *ALL_DECLARATION_COLS,
    "missing_declaration_count", "declaration_completeness_pct",
    "price_anomaly", "is_duplicate_listing", "title_desc_mismatch",
    "desc_length", "listing_metadata_completeness",
    "data_source",
]


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_sample_data(n: int = 500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic product-listing compliance dataset.

    Parameters
    ----------
    n : int
        Number of rows to generate (default 500).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Demonstration dataset with all schema columns present.
        A 'data_source' column is set to 'SYNTHETIC_DEMO' on every row.
    """
    rng = np.random.default_rng(random_seed)
    random.seed(random_seed)

    rows: list[dict] = []

    # We create three cohorts:
    #   - 60 % compliant-looking
    #   - 25 % incomplete / moderate risk
    #   - 15 % potentially high-risk
    cohorts = rng.choice(["compliant", "moderate", "high_risk"], size=n,
                         p=[0.60, 0.25, 0.15])

    for i, cohort in enumerate(cohorts):
        pid = f"PROD-{i+1:04d}"
        category = random.choice(CATEGORIES)
        adj = random.choice(ADJECTIVES)
        noun = random.choice(NOUNS.get(category, ["Item"]))
        title = f"{adj} {noun} by {random.choice(BRANDS)}"
        brand = random.choice(BRANDS)
        manufacturer = f"{brand} Manufacturing Ltd."
        country = random.choice(COUNTRIES)

        # Price generation
        base_price = float(rng.uniform(50, 5000))
        if cohort == "high_risk":
            # Unusually low or high price
            base_price = base_price * rng.choice([0.05, 0.08, 12.0, 20.0])
        declared_price = round(base_price, 2)

        # Description
        desc_words = rng.integers(15, 120)
        if cohort == "high_risk":
            desc_words = int(rng.integers(2, 15))
        description = _fake_description(category, int(desc_words), rng)
        desc_length = len(description)

        # Certification list
        cert_pool = CERTIFICATIONS.get(category, ["ISO9001"])
        if cohort == "compliant":
            certs = random.sample(cert_pool, k=min(rng.integers(1, 3), len(cert_pool)))
        elif cohort == "moderate":
            certs = random.sample(cert_pool, k=min(1, len(cert_pool))) if rng.random() > 0.4 else []
        else:
            certs = []
        certifications = ",".join(certs) if certs else ""

        # Declaration fields
        req_decls = REQUIRED_DECLARATIONS.get(category, [])
        decl_row: dict[str, int] = {}
        for col in ALL_DECLARATION_COLS:
            if col not in req_decls:
                decl_row[col] = 0  # not applicable → 0
            else:
                if cohort == "compliant":
                    decl_row[col] = 1 if rng.random() > 0.05 else 0
                elif cohort == "moderate":
                    decl_row[col] = 1 if rng.random() > 0.45 else 0
                else:
                    decl_row[col] = 1 if rng.random() > 0.80 else 0

        # Derived declaration metrics
        if req_decls:
            filled = sum(decl_row[c] for c in req_decls)
            missing_count = len(req_decls) - filled
            completeness_pct = round(100.0 * filled / len(req_decls), 1)
        else:
            missing_count = 0
            completeness_pct = 100.0

        # Anomaly flags
        price_anomaly = int(declared_price < 10 or declared_price > 50_000)
        is_duplicate = int(cohort == "high_risk" and rng.random() > 0.5)
        title_desc_mismatch = int(cohort == "high_risk" and rng.random() > 0.4)

        # Metadata completeness (0–100)
        if cohort == "compliant":
            meta_complete = round(float(rng.uniform(80, 100)), 1)
        elif cohort == "moderate":
            meta_complete = round(float(rng.uniform(40, 80)), 1)
        else:
            meta_complete = round(float(rng.uniform(5, 45)), 1)

        row = {
            "product_id": pid,
            "category": category,
            "title": title,
            "description": description,
            "declared_price": declared_price,
            "declared_brand": brand,
            "manufacturer": manufacturer,
            "country_of_origin": country,
            "certifications": certifications,
            **decl_row,
            "missing_declaration_count": missing_count,
            "declaration_completeness_pct": completeness_pct,
            "price_anomaly": price_anomaly,
            "is_duplicate_listing": is_duplicate,
            "title_desc_mismatch": title_desc_mismatch,
            "desc_length": desc_length,
            "listing_metadata_completeness": meta_complete,
            "data_source": "SYNTHETIC_DEMO",
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    logger.info("Generated %d synthetic demo rows.", len(df))
    return df


def _fake_description(category: str, n_words: int, rng: np.random.Generator) -> str:
    """Generate a plausible-looking product description."""
    word_pools = {
        "Electronics": ["high-performance", "fast-charging", "energy-efficient",
                        "compatible", "wireless", "Bluetooth", "certified", "compact"],
        "Food & Beverages": ["organic", "natural", "fresh", "preservative-free",
                             "healthy", "tasty", "nutritious", "authentic"],
        "Health & Personal Care": ["gentle", "dermatologist-tested", "hypoallergenic",
                                   "moisturising", "effective", "safe", "clinically"],
        "Toys & Games": ["safe", "non-toxic", "educational", "durable",
                         "colourful", "fun", "age-appropriate", "interactive"],
    }
    pool = word_pools.get(category, ["quality", "product", "available", "durable",
                                      "reliable", "standard", "certified", "item"])
    words = [random.choice(pool) for _ in range(n_words)]
    return " ".join(words) + f". Category: {category}."


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_csv(source: str | Path | io.BytesIO, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Load a CSV file from a file path or an uploaded BytesIO buffer.

    Parameters
    ----------
    source : str | Path | io.BytesIO
        File path or Streamlit UploadedFile / BytesIO object.
    encoding : str
        Text encoding to try first (default 'utf-8').

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If the file is empty or cannot be parsed.
    """
    try:
        if isinstance(source, (str, Path)):
            df = pd.read_csv(source, encoding=encoding, low_memory=False)
        else:
            # BytesIO / Streamlit UploadedFile
            try:
                df = pd.read_csv(source, encoding=encoding, low_memory=False)
            except UnicodeDecodeError:
                source.seek(0)
                df = pd.read_csv(source, encoding="latin-1", low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded CSV file is empty.") from exc
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("The CSV file contains no data rows.")

    # Deduplicate column names
    if df.columns.duplicated().any():
        new_cols = []
        seen: dict[str, int] = {}
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        df.columns = new_cols
        logger.warning("Duplicate column names detected and renamed.")

    logger.info("Loaded CSV with %d rows and %d columns.", *df.shape)
    return df


def get_sample_csv_bytes(n: int = 500, random_seed: int = 42) -> bytes:
    """Return the synthetic dataset as UTF-8 encoded CSV bytes (for download)."""
    df = generate_sample_data(n=n, random_seed=random_seed)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
