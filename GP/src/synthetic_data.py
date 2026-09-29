"""
Synthetic Anonymous Population Generator
Generates privacy-safe synthetic user profiles using realistic parametric distributions.
ABSOLUTE GUARDRAILS:
- Zero PII: No names, emails, phone numbers, locations, device fingerprints, or real histories.
- Abstract behavioral representation vectors only.
"""

import numpy as np
import pandas as pd


def generate_synthetic_population(n_users: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generates n_users synthetic anonymous behavioral profiles.
    Uses realistic statistical distributions (Beta, Gamma) rather than uniform random.
    """
    rng = np.random.default_rng(seed)
    
    # Anonymous alphanumeric IDs (e.g. USER_001, USER_002, ..., USER_047, ..., USER_1000)
    user_ids = [f"USER_{i+1:03d}" if n_users <= 1000 else f"USER_{i+1:04d}" for i in range(n_users)]
    
    # 1. Baseline Engagement (Beta(2, 4.5) - skewed towards moderate/low casual users)
    baseline_engagement = rng.beta(2.0, 4.5, size=n_users)
    
    # 2. Discount Sensitivity (Beta(2.8, 3.2) - bell-shaped around 0.47)
    discount_sensitivity = rng.beta(2.8, 3.2, size=n_users)
    
    # 3. Urgency Sensitivity (Beta(2.0, 4.0) - right-skewed; only a minority panic-react)
    urgency_sensitivity = rng.beta(2.0, 4.0, size=n_users)
    
    # 4. Emotional Response Tendency (Beta(3.0, 3.0) - symmetric centered around 0.50)
    emotional_response = rng.beta(3.0, 3.0, size=n_users)
    
    # 5. Baseline Action Probability P(Action | No Ad)
    # Driven primarily by baseline engagement with slight organic variance
    organic_noise = rng.normal(0, 0.02, size=n_users)
    baseline_action_prob = np.clip(
        0.05 + 0.35 * baseline_engagement + organic_noise,
        0.02, 0.45
    )
    
    df = pd.DataFrame({
        "user_id": user_ids,
        "baseline_engagement": np.round(baseline_engagement, 3),
        "discount_sensitivity": np.round(discount_sensitivity, 3),
        "urgency_sensitivity": np.round(urgency_sensitivity, 3),
        "emotional_response_tendency": np.round(emotional_response, 3),
        "baseline_action_prob": np.round(baseline_action_prob, 3)
    })
    
    return df


def simulate_ad_exposure(df_users: pd.DataFrame, ad_features: dict, seed: int = 42) -> pd.DataFrame:
    """
    Simulates the ground-truth response probabilities and randomized A/B trial.
    Calculates:
      - Ground Truth P(Action | No Ad)
      - Ground Truth P(Action | Ad)
      - Ground Truth True Incremental Impact = P(Ad) - P(No Ad)
      - Randomized Treatment assignment (T=1: Ad exposed, T=0: Control)
      - Observed binary action outcome (Y in {0, 1})
    """
    rng = np.random.default_rng(seed)
    n = len(df_users)
    
    u_ad = ad_features.get("urgency", 0.5)
    d_ad = ad_features.get("discount_emphasis", 0.5)
    e_ad = ad_features.get("emotional_intensity", 0.5)
    cta_ad = ad_features.get("cta_strength", 0.5)
    
    # Ground truth causal response mechanism:
    # Individual responsiveness is an interaction between Ad tactical framing and User sensitivities
    u_user = df_users["urgency_sensitivity"].values
    d_user = df_users["discount_sensitivity"].values
    e_user = df_users["emotional_response_tendency"].values
    eng_user = df_users["baseline_engagement"].values
    
    interaction = (
        0.18 * (u_ad * u_user) +
        0.20 * (d_ad * d_user) +
        0.12 * (e_ad * e_user) +
        0.06 * (cta_ad * eng_user)
    )
    
    # Latent true uplift tau* (bounded incremental probability)
    unobserved_idiosyncrasy = rng.normal(0, 0.010, size=n)
    true_incremental_impact = np.clip(0.005 + interaction + unobserved_idiosyncrasy, 0.005, 0.55)
    
    p_no_ad = df_users["baseline_action_prob"].values
    p_ad = np.clip(p_no_ad + true_incremental_impact, 0.01, 0.98)
    
    # Randomized Controlled Trial assignment: 50% Treatment (T=1), 50% Control (T=0)
    treatment = rng.binomial(1, 0.50, size=n)
    
    # Realized binary conversion outcome Y ~ Bernoulli(P(Action | T))
    active_prob = np.where(treatment == 1, p_ad, p_no_ad)
    observed_action = rng.binomial(1, active_prob, size=n)
    
    df_exp = df_users.copy()
    df_exp["true_p_no_ad"] = np.round(p_no_ad, 4)
    df_exp["true_p_ad"] = np.round(p_ad, 4)
    df_exp["true_incremental_impact"] = np.round(true_incremental_impact, 4)
    df_exp["treatment"] = treatment
    df_exp["observed_action"] = observed_action
    
    return df_exp
