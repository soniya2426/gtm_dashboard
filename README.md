# IOTA Water UAE | GTM Analytics Dashboard (Streamlit)

This repository contains an interactive, decision-grade Go-To-Market (GTM) analytics dashboard for **IOTA Water**, built using **Streamlit**.  
It translates UAE consumer survey data (200 respondents) into actionable insights aligned to **STP (Segmentation, Targeting, Positioning)**.

## What this dashboard does (business outcomes)

The dashboard helps founders, strategy teams, and investors answer questions like:

- Who are the highest-value consumer segments for IOTA Water in the UAE?
- Which purchase drivers matter most (taste, source, value-for-money, availability, packaging, brand)?
- What channel and pack-size strategy fits actual consumer behavior?
- Which factors are most associated with higher monthly water spend (proxy for willingness to pay)?
- Where should IOTA position (value vs premium, quality cues vs brand/trust cues)?

Every chart includes a short “So what?” insight and a GTM implication.

---

## Dataset

- **File:** `data/Research_for_bottled_water_UAE_200_respondents.csv`  
- **Type:** Primary consumer research survey  
- **Market:** UAE  
- **Respondents:** 200  
- **Note:** Some strategic constructs like “purchase intent”, “trust”, or “sustainability” may not exist as direct variables in the dataset.  
  In those cases the dashboard uses defensible proxies:
  - **Willingness to pay:** monthly spend midpoint (AED)
  - **Price sensitivity:** value-for-money importance
  - **Quality seeking:** taste + source importance
  - **Sustainability cue proxy:** packaging type importance
  - **Trust cue proxy:** brand-name importance

---

## Dashboard sections

1. **Data Overview**
   - Dataset shape, data types
   - Missing values summary (raw)
   - Cleaned preview

2. **Consumer Insights**
   - Demographics distribution (where available)
   - Consumption behavior (frequency, channel, pack size, eat-out behavior)
   - Brand awareness vs most purchased
   - Driver importance ranking
   - Cross-tabs (e.g., age vs spend, channel vs pack size)
   - Correlation heatmap (numeric variables)

3. **Hypothesis Testing & Modeling**
   - OLS Linear Regression  
     - Dependent variable: monthly spend (AED midpoint proxy for WTP)
     - Predictors: importance drivers + behavior scores  
   - Coefficients, p-values, R²
   - Hypothesis accept/reject logic (p < 0.05)

4. **Segmentation (STP)**
   - K-Means clustering using standardized features
   - Segment size + profiles
   - Auto-labeled segments with targeting playbooks

5. **Positioning & Perceptual Maps**
   - Perceptual map: Price sensitivity vs Quality seeking (brand-level)
   - Perceptual map: Packaging importance vs Brand importance (brand-level)
   - Positioning implications explained in plain language

---

## Repository structure

