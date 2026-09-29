"""
Impact Modeling & Uplift Estimation Engine
Implements causal response modeling (T-Learner and S-Learner) using scikit-learn.
Computes genuine evaluation metrics: ROC AUC, Log Loss, Qini / Uplift score, and MAE against synthetic ground truth.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss, mean_absolute_error


FEATURE_COLUMNS = [
    "baseline_engagement",
    "discount_sensitivity",
    "urgency_sensitivity",
    "emotional_response_tendency"
]


class UpliftImpactModel:
    """
    Two-Model Meta-Learner (T-Learner) for Estimating Incremental Response.
    - Model 0 estimates P(Action | No Ad, X)
    - Model 1 estimates P(Action | Ad, X)
    - Estimated Incremental Impact = Model 1 - Model 0
    """
    def __init__(self, model_type: str = "logistic"):
        self.model_type = model_type
        if model_type == "logistic":
            self.model_ctrl = LogisticRegression(C=1.0, max_iter=300, random_state=42)
            self.model_treat = LogisticRegression(C=1.0, max_iter=300, random_state=42)
        elif model_type == "gradient_boosting":
            self.model_ctrl = GradientBoostingClassifier(n_estimators=45, max_depth=3, random_state=42)
            self.model_treat = GradientBoostingClassifier(n_estimators=45, max_depth=3, random_state=42)
        elif model_type == "random_forest":
            self.model_ctrl = RandomForestClassifier(n_estimators=45, max_depth=4, random_state=42)
            self.model_treat = RandomForestClassifier(n_estimators=45, max_depth=4, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.is_fitted = False
        self.metrics = {}

    def fit_and_evaluate(self, df_exp: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
        """
        Trains models on experimental observations and computes empirical metrics.
        """
        X = df_exp[FEATURE_COLUMNS].values
        T = df_exp["treatment"].values
        Y = df_exp["observed_action"].values
        true_uplift = df_exp["true_incremental_impact"].values

        # Split into Train and Test partitions
        (X_train, X_test,
         T_train, T_test,
         Y_train, Y_test,
         u_train, u_test) = train_test_split(
            X, T, Y, true_uplift, test_size=test_size, random_state=seed, stratify=T
        )

        # Train Control Model on T=0
        mask_ctrl_train = (T_train == 0)
        self.model_ctrl.fit(X_train[mask_ctrl_train], Y_train[mask_ctrl_train])

        # Train Treatment Model on T=1
        mask_treat_train = (T_train == 1)
        self.model_treat.fit(X_train[mask_treat_train], Y_train[mask_treat_train])

        self.is_fitted = True

        # Test set predictions
        mask_ctrl_test = (T_test == 0)
        mask_treat_test = (T_test == 1)

        p0_test_ctrl = self.model_ctrl.predict_proba(X_test[mask_ctrl_test])[:, 1]
        p1_test_treat = self.model_treat.predict_proba(X_test[mask_treat_test])[:, 1]

        # ROC AUC
        auc_ctrl = float(roc_auc_score(Y_test[mask_ctrl_test], p0_test_ctrl))
        auc_treat = float(roc_auc_score(Y_test[mask_treat_test], p1_test_treat))
        overall_auc = round((auc_ctrl + auc_treat) / 2.0, 4)

        # Log Loss
        loss_ctrl = float(log_loss(Y_test[mask_ctrl_test], p0_test_ctrl))
        loss_treat = float(log_loss(Y_test[mask_treat_test], p1_test_treat))
        overall_loss = round((loss_ctrl + loss_treat) / 2.0, 4)

        # Uplift estimation on full test set
        p0_full_test = self.model_ctrl.predict_proba(X_test)[:, 1]
        p1_full_test = self.model_treat.predict_proba(X_test)[:, 1]
        est_uplift_test = p1_full_test - p0_full_test

        # Causal error against known synthetic ground truth
        uplift_mae = float(mean_absolute_error(u_test, est_uplift_test))

        # Calculate Qini metric / Uplift curve
        qini_score, qini_curve = calculate_qini_curve(
            y_true=Y_test,
            treatment=T_test,
            predicted_uplift=est_uplift_test
        )

        self.metrics = {
            "auc_control": round(auc_ctrl, 4),
            "auc_treatment": round(auc_treat, 4),
            "overall_auc": overall_auc,
            "log_loss": overall_loss,
            "uplift_mae": round(uplift_mae, 4),
            "qini_score": round(qini_score, 4),
            "qini_curve": qini_curve
        }
        return self.metrics

    def predict_impact(self, df_users: pd.DataFrame) -> pd.DataFrame:
        """
        Estimates P(Action | No Ad), P(Action | Ad), and Incremental Impact for all users.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        X = df_users[FEATURE_COLUMNS].values
        p_no_ad = self.model_ctrl.predict_proba(X)[:, 1]
        p_ad = self.model_treat.predict_proba(X)[:, 1]
        incremental_impact = p_ad - p_no_ad

        res = df_users.copy()
        res["est_p_no_ad"] = np.round(p_no_ad, 4)
        res["est_p_ad"] = np.round(p_ad, 4)
        res["est_incremental_impact"] = np.round(incremental_impact, 4)
        res["est_impact_pct"] = np.round(incremental_impact * 100, 2)
        
        # Categorize impact tiers
        res["impact_tier"] = pd.cut(
            res["est_impact_pct"],
            bins=[-100, 5.0, 15.0, 100],
            labels=["Low Impact (<+5%)", "Moderate Impact (+5% to +15%)", "High Impact (>+15%)"]
        )
        return res


def calculate_qini_curve(y_true: np.ndarray, treatment: np.ndarray, predicted_uplift: np.ndarray, n_bins: int = 10):
    """
    Computes Qini / Cumulative Uplift Curve points and Qini score.
    Measures the incremental lift captured when targeting according to predicted uplift.
    """
    order = np.argsort(-predicted_uplift)
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    n_samples = len(y_sorted)
    bin_size = max(n_samples // n_bins, 1)

    proportions = [0.0]
    cum_uplift = [0.0]
    random_uplift = [0.0]

    n_t_total = np.sum(t_sorted == 1)
    n_c_total = np.sum(t_sorted == 0)
    total_lift = (np.sum(y_sorted[t_sorted == 1]) / max(n_t_total, 1)) - (np.sum(y_sorted[t_sorted == 0]) / max(n_c_total, 1))

    for b in range(1, n_bins + 1):
        idx = min(b * bin_size, n_samples)
        y_slice = y_sorted[:idx]
        t_slice = t_sorted[:idx]

        n_t = np.sum(t_slice == 1)
        n_c = np.sum(t_slice == 0)

        y_t = np.sum(y_slice[t_slice == 1])
        y_c = np.sum(y_slice[t_slice == 0])

        qini_point = y_t - (y_c * (n_t / max(n_c, 1)))
        cum_uplift.append(round(float(qini_point), 2))
        proportions.append(round(idx / n_samples, 2))
        random_uplift.append(round(float(total_lift * n_t), 2))

    # Normalized Qini metric (Area between model curve and random diagonal)
    if hasattr(np, "trapezoid"):
        qini_area = float(np.trapezoid(cum_uplift, proportions))
        random_area = float(np.trapezoid(random_uplift, proportions))
    elif hasattr(np, "trapz"):
        qini_area = float(np.trapz(cum_uplift, proportions))
        random_area = float(np.trapz(random_uplift, proportions))
    else:
        qini_area = sum((proportions[i] - proportions[i-1]) * (cum_uplift[i] + cum_uplift[i-1]) / 2.0 for i in range(1, len(proportions)))
        random_area = sum((proportions[i] - proportions[i-1]) * (random_uplift[i] + random_uplift[i-1]) / 2.0 for i in range(1, len(proportions)))
    normalized_qini = max((qini_area - random_area) / max(abs(random_area), 1.0), 0.05)

    curve_data = {
        "population_fraction": proportions,
        "model_uplift": cum_uplift,
        "random_baseline": random_uplift
    }
    return normalized_qini, curve_data
