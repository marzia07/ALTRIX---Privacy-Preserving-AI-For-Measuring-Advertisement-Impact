# ALTRIX
> **“Understand the Impact. Not the Identity.”**

ALTRIX is a privacy-preserving AI prototype designed to estimate the **incremental causal impact** of digital advertisements on anonymous user populations, while minimizing the centralization of individual behavioral information.

---

## 🎯 The Core Conceptual Distinction

| Conventional Ad Analytics | ALTRIX Causal Architecture |
| :--- | :--- |
| **Core Question:** *“Who clicked?”* | **Core Question:** *“How much did the advertisement change the probability of action?”* |
| **Mechanics:** Tracks persistent user identifiers, cookies, and centralized clickstream histories. | **Mechanics:** Learns from ephemeral, anonymous behavioral sensitivity vectors. |
| **Targeting Trap:** Spends ad budget converting "sure things" (users who would act organically anyway). | **Targeting Efficiency:** Isolates true incremental uplift: $P(\text{Action} \mid \text{Ad}) - P(\text{Action} \mid \text{No Ad})$. |
| **Privacy Risk:** Creates permanent central individual dossiers vulnerable to leaks or regulatory penalties. | **Privacy Guarantee:** Local processing, federated aggregation, and differential privacy perturbation. |

### Concrete Example: Heterogeneous Incremental Response
Consider two anonymous users responding to the same campaign:
* **Anonymous User A:**
  * $P(\text{Action} \mid \text{No Ad}) = 20\%$
  * $P(\text{Action} \mid \text{Ad}) = 23\%$
  * **Estimated Incremental Impact:** **$+3$ percentage points** *(High organic propensity; low ad sensitivity)*
* **Anonymous User B:**
  * $P(\text{Action} \mid \text{No Ad}) = 20\%$
  * $P(\text{Action} \mid \text{Ad}) = 48\%$
  * **Estimated Incremental Impact:** **$+28$ percentage points** *(High campaign sensitivity; massive causal shift)*

The same advertisement produces radically different estimated effects across anonymous profiles without ever knowing their personal identities.

---

## 🛡️ Privacy by Design Architecture

ALTRIX adheres to the foundational principle: **“Learn more while knowing less.”**

```
┌──────────────┐     ┌───────────────────────┐     ┌──────────────────────┐
│    DEVICE    │ ──> │ Local Behavioral Data │ ──> │ Local Model Training │
└──────────────┘     └───────────────────────┘     └──────────────────────┘
                                                              │
                                                              ▼
┌───────────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│  Aggregate Insights   │ <── │    Global Model     │ <── │   FedAvg / Laplace   │
│   (Zero PII Stored)   │     │  Consensus Updates  │     │ Protected Aggregation│
└───────────────────────┘     └─────────────────────┘     └──────────────────────┘
```

### Privacy Guarantee Checklist
- [x] **No Names** collected or stored
- [x] **No Phone Numbers** accessed
- [x] **No Email Addresses** queried
- [x] **No Exact GPS or Cell-Tower Locations**
- [x] **No Raw Web Browsing Histories** centralized
- [x] **100% Synthetic Anonymous Profiles** (`USER_001` ... `USER_1000`)
- [x] **Exclusively Aggregate Statistical Outputs**

---

## 🔬 System Transparency: Implemented vs. Simulated vs. Future Research

In accordance with scientific honesty, ALTRIX explicitly declares the implementation boundary of every module:

| Component | Status | Technical Description |
| :--- | :--- | :--- |
| **Ad Text & Image Analyzer** | **Fully Implemented** | Heuristic regex semantic parser + PIL computer-vision contrast/luminance analyzer. Evaluates urgency, emotional intensity, discount emphasis, and CTA strength. |
| **Synthetic Population Engine** | **Fully Implemented** | Parametric non-uniform distributions (Beta, Gamma) generating realistic behavioral vectors without PII. |
| **Causal Impact Meta-Learner** | **Fully Implemented** | T-Learner using scikit-learn Logistic Regression, Gradient Boosting, or Random Forests. Computes authentic ROC AUC, Log Loss, Qini uplift score, and ground-truth MAE. |
| **Federated Learning (FedAvg)** | **Functional Simulation** | In-memory decentralized simulation across 5 simulated client devices with local SGD fitting and parameter weight aggregation. |
| **Differential Privacy** | **Educational Simulation** | Parameter clipping ($C=1.0$) with calibrated Laplace noise injection scaled to sensitivity ($\Delta S$) and privacy budget ($\epsilon$). |
| **Hardware Enclave Cryptography** | **Future Research** | Trusted Execution Environments (TEEs) and Secure Multi-Party Computation (SMPC) protocols. |

---

## 📱 Telecom Application Scenario: Grameenphone (GP) Context

> **Scientific & Fair Framing:** This scenario illustrates a forward-looking application for a telecommunications operator such as Grameenphone. It does **not** claim that Grameenphone currently lacks privacy safeguards, nor does it make claims about proprietary existing infrastructure.

ALTRIX explores how telecom operators can evaluate campaign impact while upholding customer trust:
* **Campaign A (“20GB for ৳299”):** Evaluates whether recharges were caused by the promotional SMS or if subscribers were renewing expiring quotas organically.
* **Campaign B (“Stay Connected with Family”):** Measures long-term retention lift driven by social connectivity messaging.
* **Campaign C (“Flash Cashback Offer”):** Analyzes impulse recharge elasticity across urgency-sensitive subscriber segments.

**Key Point:** ALTRIX does not replace personalization; it explores how advertising effectiveness can be measured while eliminating dependence on centralized individual behavioral dossiers.

---

## 🧭 Consolidated Application Structure (6 Cohesive Sections)

1. **HOME:** System overview, core value propositions, conceptual distinction, and live 3–5 minute presentation roadmap.
2. **ANALYZE & SIMULATE:** 
   - **Advertisement Analyzer:** AI copy/visual parsing with pre-calibrated sample ads (telecom flash promo, family emotional bond, eSIM launch). Labeled as prototype AI-derived characteristics, not validated psychological measurements.
   - **Synthetic Users:** Parametric generator producing 1,000 anonymous profiles with non-uniform distributions. Prominently labeled: *“Synthetic data — no real users.”*
3. **IMPACT (Core Section):** 
   - **Visually Dominant Result:** Large display showing Without Ad (20%), With Ad (48%), and Incremental Impact (+28 percentage points).
   - **Heterogeneous Response:** Direct side-by-side comparison of Anonymous User A (+3 pp) vs User B (+28 pp).
   - **Authentic Computed Metrics:** True ROC AUC, Log Loss, Qini curve, and ground-truth MAE.
   - **Creative Comparison:** Overlay distribution comparing Ad A vs Ad B over the same population.
4. **PRIVACY:** 
   - Visual decentralized pipeline (`RAW DATA → LOCAL PROCESSING → MODEL UPDATE → PRIVACY PROTECTION → SECURE AGGREGATION → GLOBAL MODEL → AGGREGATE INSIGHTS`) with privacy checklist.
   - **Federated Learning (FUNCTIONAL SIMULATION):** 5 simulated client devices performing local training and FedAvg parameter aggregation with before/after weight inspection.
   - **Differential Privacy (EDUCATIONAL/PROTOTYPE SIMULATION):** Interactive $\epsilon$ slider demonstrating calibrated noise injection and the empirical privacy–utility trade-off.
5. **RESULTS:** 
   - Multi-point empirical Pareto frontier curve illustrating classification utility vs. estimation error.
   - Objective side-by-side comparison (Conventional Analytics vs. ALTRIX).
   - Formal transparency matrix and strict ethical guardrails.
6. **GP USE CASE & FINAL MESSAGE:** 
   - Exploratory application for telecom campaigns (Grameenphone context) as a privacy-preserving measurement layer complementing personalization.
   - Closing Vision: **# LEARN MORE. KNOW LESS.**

---

## ⚡ Fast-Track 3–5 Minute Competition Demo Flow

For presentations to competition judges, execute this seamless flow:
1. **Open ALTRIX (`1. HOME`):** Introduce the thesis: *“We don't need to know who you are to study how an ad changes behavior.”*
2. **Analyze & Simulate (`2. ANALYZE & SIMULATE`):** Select the sample ad *“Telecom Promo Offer (20GB for ৳299)”* and click **Generate Synthetic Population** to showcase 1,000 anonymous profiles (highlighting `USER_047`).
3. **Inspect Core Impact (`3. IMPACT`):** Walk through the dominant result:
   * **Without Ad:** 20%
   * **With Ad:** 48%
   * **Incremental Impact:** **+28 percentage points**
   * Explain: *“This estimates how much the advertisement changed the probability of action for this anonymous behavioral profile.”*
   * Point out the contrasting response between **User A (+3 pp)** and **User B (+28 pp)**. Show the Qini curve and ROC AUC.
4. **Demonstrate Privacy (`4. PRIVACY`):**
   * Show the 7-step decentralized pipeline flow and privacy checklist.
   * Click **Execute Next Federated Round** across the 5 simulated devices.
   * Adjust the Privacy Budget slider ($\epsilon$) to demonstrate calibrated noise injection.
5. **Review Results (`5. RESULTS`):** Show the empirical Privacy vs. Utility frontier curve.
6. **Wrap-up (`6. GP USE CASE`):** Conclude with the telecom industry application and the prominent closing statement: **# LEARN MORE. KNOW LESS.**

---

## 💻 Setup & Installation Instructions

ALTRIX runs locally on an ordinary laptop without requiring a GPU.

### Prerequisites
* Python 3.10+ (Tested on Python 3.10, 3.11, 3.12, 3.13)
* Git

### Installation
```bash
# Clone the repository
cd GP

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Launching the Prototype
```bash
# Using the convenience startup script
./run_altrix.sh  # or ./run_altim.sh

# Or directly via Streamlit
./.venv/bin/streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## ⚖️ Ethical Guardrails

The software strictly enforces that it must **NOT**:
1. Identify individuals or attempt to reconstruct real-world identities.
2. Infer names, phone numbers, or protected demographic attributes.
3. Diagnose psychological vulnerability or label individuals as vulnerable.
4. Recommend predatory marketing strategies targeting susceptible populations.
5. Claim certainty about individual human psychology.
6. Utilize proprietary customer records without consent.

**Core Purpose:** Responsible measurement of digital advertising influence while minimizing individual data exposure.
