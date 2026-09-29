"""
Advertisement Feature Analyzer Module
Extracts prototype AI-derived characteristics from ad copy and visual assets.
NOTE: Clearly labeled as prototype AI-derived characteristics, not validated psychological measurements.
"""

import re
import numpy as np
from PIL import Image

# Semantic feature lexicons
URGENCY_KEYWORDS = [
    r"\blimited\b", r"\bhurry\b", r"\bnow\b", r"\btoday\b", r"\bflash\b",
    r"\blast\s+chance\b", r"\bexpire[s]?\b", r"\bmidnight\b", r"\bonly\b",
    r"\bhours?\b", r"\bquick\b", r"\bfast\b", r"\bdon'?t\s+miss\b", r"\bends\b"
]

DISCOUNT_KEYWORDS = [
    r"\b\d+%\b", r"\bdiscount\b", r"\bsave\b", r"\bcashback\b", r"\bdeal\b",
    r"\bfree\b", r"\bvoucher\b", r"\b৳\d+\b", r"\btk\b", r"\boffer\b", r"\bgb\b",
    r"\bprice\b", r"\bcheap\b", r"\bbonus\b"
]

EMOTIONAL_KEYWORDS = [
    r"\bfamily\b", r"\bheart\b", r"\blove\b", r"\btogether\b", r"\bcare\b",
    r"\bmoments?\b", r"\bshare\b", r"\bjoy\b", r"\bsmile\b", r"\bconnect\b",
    r"\bhome\b", r"\bcelebrate\b", r"\balways\b", r"\btrust\b"
]

CTA_KEYWORDS = [
    r"\bget\b", r"\bbuy\b", r"\bactivate\b", r"\bclaim\b", r"\bdial\b",
    r"\bdownload\b", r"\btap\b", r"\bclick\b", r"\bstart\b", r"\bjoin\b",
    r"\bupgrade\b", r"\brecharge\b", r"\bcall\b"
]

BANGLA_TRANSLIT_KEYWORDS = [
    r"\btk\b", r"\b৳\b", r"\bmygp\b", r"\bshobar\b", r"\bbhalo\b", r"\bduronto\b",
    r"\bbondhu\b", r"\bekhon\b", r"\bmatro\b", r"\bpaben\b"
]


def analyze_ad_text(text: str) -> dict:
    """
    Parses advertisement copy and calculates normalized prototype AI feature scores.
    """
    text_lower = text.lower()
    total_words = max(len(text_lower.split()), 1)
    
    def score_lexicon(patterns):
        matches = 0
        for pattern in patterns:
            found = re.findall(pattern, text_lower)
            matches += len(found)
        # Scaled saturation curve
        return float(np.clip(matches / max(total_words * 0.15, 1.5), 0.0, 1.0))

    urgency_raw = score_lexicon(URGENCY_KEYWORDS)
    discount_raw = score_lexicon(DISCOUNT_KEYWORDS)
    emotional_raw = score_lexicon(EMOTIONAL_KEYWORDS)
    cta_raw = score_lexicon(CTA_KEYWORDS)

    # Exclamation mark booster
    exclamations = text.count("!")
    urgency = float(np.clip(urgency_raw * 0.75 + min(exclamations * 0.08, 0.25), 0.05, 0.98))
    discount = float(np.clip(discount_raw * 0.85 + (0.15 if any(char in text for char in ['%', '৳', '$']) else 0.0), 0.05, 0.98))
    emotional = float(np.clip(emotional_raw * 0.90, 0.05, 0.98))
    cta = float(np.clip(cta_raw * 0.80 + (0.15 if exclamations > 0 else 0.0), 0.10, 0.98))

    # Persuasive intensity: combination of CTA, urgency, and discount
    persuasive = float(np.clip(0.35 * urgency + 0.35 * discount + 0.30 * cta, 0.10, 0.98))

    # Sentiment estimation (warm vs promotional)
    positive_cues = len(re.findall(r"\b(best|great|free|save|love|joy|fast|easy|happy)\b", text_lower))
    sentiment = float(np.clip(0.40 + 0.12 * positive_cues + 0.20 * emotional, 0.10, 0.95))

    # Language detection
    has_bangla_chars = bool(re.search(r'[\u0980-\u09FF]', text))
    has_banglish = any(re.search(pat, text_lower) for pat in BANGLA_TRANSLIT_KEYWORDS)
    if has_bangla_chars:
        language = "Bengali (Bangla Script)"
    elif has_banglish:
        language = "English / Bengali Transliteration"
    else:
        language = "English"

    # Default visual intensity without image
    visual_intensity = float(np.clip(0.40 + 0.35 * urgency + 0.15 * discount, 0.20, 0.90))

    explanation = generate_feature_explanation(urgency, emotional, discount, cta)

    return {
        "urgency": round(urgency, 2),
        "emotional_intensity": round(emotional, 2),
        "discount_emphasis": round(discount, 2),
        "cta_strength": round(cta, 2),
        "persuasive_intensity": round(persuasive, 2),
        "sentiment": round(sentiment, 2),
        "visual_intensity": round(visual_intensity, 2),
        "language": language,
        "explanation": explanation
    }


def analyze_ad_image(image: Image.Image) -> dict:
    """
    Extracts visual features from an uploaded ad image using computer vision heuristics.
    """
    img_rgb = image.convert("RGB")
    arr = np.array(img_rgb)
    
    # Luminance / brightness
    gray = np.mean(arr, axis=2)
    brightness = float(np.mean(gray) / 255.0)
    
    # Contrast (standard deviation of grayscale values)
    contrast = float(np.std(gray) / 128.0)
    contrast = np.clip(contrast, 0.0, 1.0)
    
    # Colorfulness / Saturation variance
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    rg = np.abs(r.astype(float) - g.astype(float))
    yb = np.abs(0.5 * (r.astype(float) + g.astype(float)) - b.astype(float))
    colorfulness = float((np.std(rg) + np.std(yb) + 0.3 * (np.mean(rg) + np.mean(yb))) / 180.0)
    colorfulness = np.clip(colorfulness, 0.0, 1.0)
    
    # Visual intensity composite
    visual_intensity = float(np.clip(0.4 * contrast + 0.4 * colorfulness + 0.2 * (1.0 - abs(brightness - 0.5)), 0.1, 0.98))
    
    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "colorfulness": round(colorfulness, 2),
        "visual_intensity": round(visual_intensity, 2)
    }


def generate_feature_explanation(urgency: float, emotional: float, discount: float, cta: float) -> str:
    """
    Generates natural language architectural interpretation of the ad's tactical framing.
    """
    drivers = []
    if urgency >= 0.70:
        drivers.append("strong urgency cues")
    elif urgency >= 0.40:
        drivers.append("moderate timeliness")
        
    if discount >= 0.70:
        drivers.append("heavy discount & monetary incentives")
    elif discount >= 0.40:
        drivers.append("value-focused promotional framing")
        
    if emotional >= 0.60:
        drivers.append("deep emotional & social connection appeals")
        
    if cta >= 0.75:
        drivers.append("a direct, imperative call-to-action")
        
    if not drivers:
        drivers.append("subtle informational framing with low pressure")
        
    explanation = f"This advertisement uses {', '.join(drivers)}."
    return explanation
