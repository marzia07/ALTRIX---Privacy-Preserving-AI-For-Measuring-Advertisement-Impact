"""
ALTRIX Configuration & Design System
"Understand the Impact. Not the Identity."
"""

# Color Palette (Dark Navy, White, Subtle Blue/Cyan Accents)
THEME_COLORS = {
    "bg_dark": "#070D18",
    "bg_card": "#0F172A",
    "bg_card_hover": "#16223B",
    "border_card": "#1E293B",
    "border_highlight": "#38BDF8",
    "accent_cyan": "#00F0FF",
    "accent_blue": "#38BDF8",
    "accent_indigo": "#6366F1",
    "accent_green": "#10B981",
    "accent_amber": "#F59E0B",
    "accent_rose": "#F43F5E",
    "text_primary": "#F8FAFC",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
}

# Custom CSS for Sleek Dark Navy AI / Security Interface
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #F8FAFC;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace;
}

/* Background overrides */
.stApp {
    background: radial-gradient(circle at 15% 15%, #0B1528 0%, #070D18 60%, #040810 100%);
    background-attachment: fixed;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0A1124;
    border-right: 1px solid #1E293B;
}

[data-testid="stSidebar"] hr {
    border-color: #1E293B;
}

/* Custom Card Container */
.altrix-card, .altim-card {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    transition: all 0.25s ease;
}

.altrix-card:hover, .altim-card:hover {
    border-color: #334155;
    box-shadow: 0 8px 30px -4px rgba(0, 240, 255, 0.05);
}

.altrix-card-glow, .altim-card-glow {
    background: #0F172A;
    border: 1px solid rgba(0, 240, 255, 0.3);
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 0 25px rgba(0, 240, 255, 0.1);
}

/* Hero Badge */
.altrix-badge, .altim-badge {
    display: inline-block;
    padding: 4px 12px;
    background: rgba(0, 240, 255, 0.1);
    border: 1px solid rgba(0, 240, 255, 0.3);
    border-radius: 20px;
    color: #00F0FF;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.altrix-badge-green, .altim-badge-green {
    display: inline-block;
    padding: 4px 12px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 20px;
    color: #10B981;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.altrix-badge-amber, .altim-badge-amber {
    display: inline-block;
    padding: 4px 12px;
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 20px;
    color: #F59E0B;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
}

/* Metric Display Cards */
.metric-box {
    background: #0B132B;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}

.metric-title {
    font-size: 0.8rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: #00F0FF;
    font-family: 'JetBrains Mono', monospace;
}

.metric-delta-positive {
    font-size: 0.82rem;
    color: #10B981;
    font-weight: 600;
    margin-top: 4px;
}

/* Progress bar container */
.gauge-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: #E2E8F0;
    margin-bottom: 4px;
    font-weight: 500;
}

.gauge-track {
    background-color: #1E293B;
    border-radius: 6px;
    height: 10px;
    width: 100%;
    margin-bottom: 14px;
    overflow: hidden;
}

.gauge-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #38BDF8 0%, #00F0FF 100%);
}

/* Callout Note */
.altrix-callout, .altim-callout {
    background: rgba(30, 41, 59, 0.6);
    border-left: 4px solid #38BDF8;
    padding: 14px 18px;
    border-radius: 0 8px 8px 0;
    font-size: 0.9rem;
    color: #CBD5E1;
    margin: 15px 0;
}

.altrix-disclaimer, .altim-disclaimer {
    background: rgba(245, 158, 11, 0.08);
    border-left: 4px solid #F59E0B;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 0.82rem;
    color: #FCD34D;
    margin: 12px 0;
}

/* Streamlit button overrides */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
    color: #FFFFFF;
    font-weight: 600;
    border: 1px solid #38BDF8;
    border-radius: 8px;
    padding: 0.55rem 1.4rem;
    transition: all 0.2s ease;
}

div.stButton > button:first-child:hover {
    background: linear-gradient(135deg, #0369A1 0%, #075985 100%);
    border-color: #00F0FF;
    box-shadow: 0 0 16px rgba(0, 240, 255, 0.35);
    color: #FFFFFF;
}

/* Dataframe styling */
div[data-testid="stDataFrame"] {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 8px;
}

/* Visually Dominant Hero Impact Card */
.dominant-impact-card {
    background: radial-gradient(circle at 50% 0%, #112240 0%, #0A1428 100%);
    border: 2px solid #00F0FF;
    border-radius: 16px;
    padding: 32px 28px;
    margin: 24px 0;
    box-shadow: 0 0 35px rgba(0, 240, 255, 0.22);
}

.impact-col-box {
    background: #071020;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.impact-huge-num {
    font-size: 2.8rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin: 8px 0;
}

/* Closing Statement Banner */
.closing-banner {
    background: radial-gradient(circle at 50% 30%, #16264C 0%, #091224 85%);
    border: 2px solid rgba(0, 240, 255, 0.4);
    border-radius: 16px;
    padding: 48px 30px;
    text-align: center;
    margin: 35px 0 25px 0;
    box-shadow: 0 0 45px rgba(0, 240, 255, 0.18);
}

/* Presentation Flow Stepper */
.nav-stepper {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #0B132B;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 24px;
    gap: 8px;
}

.nav-step {
    flex: 1;
    text-align: center;
    padding: 8px 10px;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748B;
    background: #070D18;
    border: 1px solid #16223B;
    text-decoration: none;
    transition: all 0.2s ease;
}

.nav-step.active {
    color: #00F0FF;
    background: rgba(0, 240, 255, 0.12);
    border: 1px solid #00F0FF;
    box-shadow: 0 0 12px rgba(0, 240, 255, 0.2);
}

.nav-step.completed {
    color: #10B981;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.3);
}
</style>
"""

# Sample Advertisements for Quick Testing & Demonstration
SAMPLE_ADS = [
    {
        "id": "ad_telecom_promo",
        "title": "Telecom Promo Offer (Urgency + Discount)",
        "text": "LIMITED TIME! Get 20GB for ৳299. Don't miss out! Dial *121# or recharge on MyGP now to claim this flash pack before midnight.",
        "image_desc": "Vibrant cyan & orange telecom banner with countdown timer badge and data pack graphics.",
        "tag": "Telecom Campaign A",
        "scores": {
            "urgency": 0.86,
            "emotional_intensity": 0.28,
            "discount_emphasis": 0.92,
            "cta_strength": 0.90,
            "persuasive_intensity": 0.85,
            "sentiment": 0.65,
            "visual_intensity": 0.82,
            "language": "English / Banglish Transliteration"
        },
        "explanation": "This advertisement relies heavily on time urgency and monetary promotional framing with a high-friction direct CTA."
    },
    {
        "id": "ad_family_connectivity",
        "title": "Family Connectivity Campaign (High Emotion)",
        "text": "Stay connected with your family. Share heartfelt moments across distances with seamless HD video calling and reliable 4G network coverage nationwide.",
        "image_desc": "Warm photo of family video-calling across rural and urban landscapes with clean blue network lines.",
        "tag": "Telecom Campaign B",
        "scores": {
            "urgency": 0.18,
            "emotional_intensity": 0.88,
            "discount_emphasis": 0.15,
            "cta_strength": 0.42,
            "persuasive_intensity": 0.62,
            "sentiment": 0.82,
            "visual_intensity": 0.55,
            "language": "English"
        },
        "explanation": "This advertisement appeals strongly to affective emotional bonding and network reliability rather than urgency or financial discounting."
    },
    {
        "id": "ad_esim_tech",
        "title": "Modern Tech / eSIM Launch (Informational CTA)",
        "text": "Upgrade to next-generation eSIM. Activate your secure mobile profile instantly in minutes. Zero plastic waste, seamless multi-SIM flexibility.",
        "image_desc": "Minimalist dark navy visual showcasing microchip silhouette and green eco badge.",
        "tag": "Innovation Campaign",
        "scores": {
            "urgency": 0.38,
            "emotional_intensity": 0.32,
            "discount_emphasis": 0.10,
            "cta_strength": 0.84,
            "persuasive_intensity": 0.70,
            "sentiment": 0.45,
            "visual_intensity": 0.68,
            "language": "English"
        },
        "explanation": "This advertisement emphasizes technical innovation, utility, and environmental consciousness with a moderate instructional call-to-action."
    },
    {
        "id": "ad_weekend_flash",
        "title": "Weekend Flash Cashback (Extreme Urgency)",
        "text": "FLASH DEAL! 50% Instant Cashback on 50GB Monthly Pack. Only 4 Hours Remaining! Tap to claim your voucher now before stocks expire.",
        "image_desc": "High-contrast electric yellow flash icon with big bold discount percentage.",
        "tag": "Telecom Campaign C",
        "scores": {
            "urgency": 0.96,
            "emotional_intensity": 0.35,
            "discount_emphasis": 0.95,
            "cta_strength": 0.94,
            "persuasive_intensity": 0.92,
            "sentiment": 0.70,
            "visual_intensity": 0.90,
            "language": "English"
        },
        "explanation": "This advertisement combines maximum urgency and extreme discount framing to generate immediate impulse response."
    }
]
