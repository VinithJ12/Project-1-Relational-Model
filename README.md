# DS 4320 Project 1: Predicting FEMA Housing Assistance Needs

**Name:** Vinith Jayamani  
**NetID:** uhe5bj  
**DOI:** [![DOI](https://zenodo.org/badge/1191117026.svg)](https://doi.org/10.5281/zenodo.19354270)                  
**Press Release:** [press_release.md](press_release.md)  
**Data and Background Reading:** [UVA OneDrive Folder](https://myuva-my.sharepoint.com/:f:/g/personal/uhe5bj_virginia_edu/IgBSQ5AQLTDES4ZOwpFBc-MrAaSVj_qqcQxvivzJ14UM6-I?e=0M6Xs5) 
**RAW DATAFOLDER:**[RAW DATASET FOLDER](https://myuva-my.sharepoint.com/:f:/g/personal/uhe5bj_virginia_edu/IgDmqPcIuJ5wQr5U5EpRJ0cyARMvFTWd2hDzkrFYAH5dCZE?e=qBH6b8)
**Pipeline:** [pipeline.ipynb](pipeline_final.ipynb) | [pipeline.md](pipeline_final.md)                                                                                
**COLAB Pipeline Option:** [Pipeline.ipynb](https://colab.research.google.com/drive/1jgfsSzeFVxpDYwF-ckdhbLhCPfeeTKHG?usp=sharing)                                          
**License:** MIT License  [LICENSE](LICENSE)

---

## Executive Summary

This project builds a machine learning pipeline to predict whether disaster-affected households require habitability repairs following federally declared large-scale disasters. Using FEMA's Individual Assistance Housing Registrants dataset — 6.3 million raw registrant records across FL, TX, PR, and LA — the project normalizes the data into a four-table relational schema, conducts exploratory analysis using DuckDB SQL queries, and trains two models: a Random Forest classifier (81.2% accuracy) and a Gradient Boosting regressor (R² = 0.664). The goal is to demonstrate that FEMA's existing administrative data contains sufficient signal to enable proactive, data-driven triage of disaster housing needs.

---

## Problem Definition

### General and specific problem statement

**General problem:** Allocating emergency response resources

**Specific problem:** Following large-scale disasters, FEMA's Individual Assistance program distributes housing aid reactively — inspectors visit homes one by one, and eligibility decisions are made household by household with no predictive framework to guide prioritization. This project asks: given the demographic and damage information FEMA already collects at the time of registration, can we predict which households will require habitability repairs before or during the inspection process? The target variable is `habitabilityRepairsRequired`, a binary indicator recorded after physical inspection, and the secondary target is `rpfvl` (real property field visit loss in dollars).

### Rationale for refinement

The general problem of emergency resource allocation is intentionally broad — it could apply to ambulance routing, firefighter dispatch, or shelter placement. The refinement to FEMA housing triage was driven by the availability of a uniquely rich administrative dataset: FEMA's Individual Assistance Housing Registrants records capture self-reported household demographics, insurance status, and residence type at registration, alongside ground-truth damage assessments and aid outcomes recorded weeks later. This pairing of pre-inspection features with post-inspection labels is exactly what supervised machine learning requires, and it exists at a scale (6.3 million records) large enough to train robust models. The specific focus on `habitabilityRepairsRequired` — rather than, say, aid dollar amounts — was chosen because it represents the most actionable binary decision a field team can make: is this home safe to live in or not?

### Motivation

Every hour of delay in disaster resource allocation has direct human consequences. After Hurricane Harvey in 2017, FEMA received over 1 million assistance applications in weeks. After Hurricane Maria in Puerto Rico, processing backlogs left thousands of families in damaged homes for months. The current inspection-driven process is necessarily sequential — inspectors visit homes one by one — but the data FEMA collects at registration is available immediately. A predictive model trained on historical patterns could allow emergency managers to flag high-probability repair cases before inspectors arrive, enabling smarter scheduling, faster resource pre-positioning, and more equitable outcomes. This project demonstrates that the signal to build such a tool already exists in FEMA's own records.

**Press Release Headline:** *FEMA Data Can Predict Which Disaster Survivors Need Help Most — Before an Inspector Ever Arrives*
[Read the full press release](press_release.md)

---

## Domain Exposition

### Terminology

| Term | Definition |
|---|---|
| FEMA IA | Individual Assistance — FEMA's direct aid program for households affected by presidentially declared disasters |
| Habitability repair | Physical repair required to make a home safe and livable; determines whether a household can return home |
| RPFVL | Real Property Field Visit Loss — dollar value of assessed damage to the physical structure of the home |
| PPFVL | Personal Property Field Visit Loss — dollar value of assessed damage to household belongings |
| TSA | Transitional Sheltering Assistance — FEMA program covering temporary hotel stays for displaced households |
| SBA eligible | Eligible for a Small Business Administration low-interest disaster loan for home repair |
| Inspection | Physical visit by a FEMA inspector to assess damage; required for most aid determinations |
| Renter damage level | FEMA classification of damage severity for renting households (e.g., Renter-Major Damage) |
| Census block ID | 15-digit geographic identifier linking a household to a specific census block for demographic enrichment |
| Disaster number | FEMA's unique identifier for each federally declared disaster event |
| Class imbalance | When one outcome (e.g., no repairs needed) is more common than the other; addressed with `class_weight='balanced'` |
| F1 score | Harmonic mean of precision and recall; preferred over accuracy for imbalanced classification problems |
| R² | Coefficient of determination; proportion of variance in the target explained by the model (0–1) |

### Domain paragraph

This project lives at the intersection of disaster informatics, public policy, and applied machine learning. The domain is federally administered disaster recovery in the United States, specifically the Individual Assistance program run by FEMA under the Stafford Act. When a president declares a major disaster, FEMA opens registration for affected households to apply for housing assistance. The program serves millions of Americans annually across a wide range of disaster types — hurricanes, floods, tornadoes, and wildfires — and involves complex eligibility determinations based on damage severity, insurance coverage, household income, and residency status. The data generated by this process is administrative in nature: collected under time pressure, self-reported by distressed households, and processed by field inspectors operating in chaotic post-disaster environments. This context introduces significant data quality challenges (missing values, self-reporting bias, geographic concentration) that are characteristic of real-world public sector data science problems and must be explicitly addressed in any analytical pipeline.

### Background readings

| # | Title | Description | File |
|---|---|---|---|
| 1 | FEMA OpenFEMA Data Documentation | Official documentation for the Individual Assistance Housing Registrants dataset including field definitions and collection methodology | [fema_openfema.pdf](https://myuva-my.sharepoint.com/:b:/g/personal/uhe5bj_virginia_edu/IQAQc_QtEJe9SrK_R3kZpS1HASwPmzSMseCAecko6RTKR5E?e=dj4zCz) |
| 2 | GAO Report: FEMA Should Take Additional Steps to Streamline Disaster Assistance | Government Accountability Office report finding FEMA lacks systematic use of historical data to anticipate resource needs | [gao_fema_report.pdf](https://myuva-my.sharepoint.com/:b:/g/personal/uhe5bj_virginia_edu/IQBUduklIV5JTq9lef-v0cWqAer_ZfxHIvwvp4irZQQq8k4?e=OoHCfT) |
| 3 | Center for American Progress: Fixing FEMA | Policy analysis recommending FEMA shift to needs-based rather than damage-based aid allocation | [cap_fixing_fema.pdf](https://myuva-my.sharepoint.com/:b:/g/personal/uhe5bj_virginia_edu/IQCMDDu1G4aiQYMjwXETT2CQAYBqEtudxL3ZJJptSd-eeSY?e=HSGXJa) |
| 4 | Tulane University: Barriers to Equitable Disaster Recovery | 25-year literature review documenting how race, income, and renter status affect disaster aid outcomes | [tulane_equity_review.pdf](https://myuva-my.sharepoint.com/:b:/g/personal/uhe5bj_virginia_edu/IQA87maeobcSTY4dMEEuplRGAfhM_mSG-b4iAZ51_4j4-og?e=NYIBo9) |
| 5 | NPR Investigation: FEMA Aid Inequity | Investigative journalism analysis of 4.8 million FEMA registrations finding poorest renters 23% less likely to receive housing help | [npr_fema_inequity.pdf](https://myuva-my.sharepoint.com/:b:/g/personal/uhe5bj_virginia_edu/IQBS4bWGFDdCQ6i4JMBYOiROAa4sur1GxGMNpRKQydtAmQk?e=RTMLtf) |

---

## Data Creation

### Provenance

The raw dataset is FEMA's Individual Assistance Housing Registrants Large Disasters dataset, obtained from the FEMA OpenFEMA API (https://www.fema.gov/openfema-data-page/individual-assistance-housing-registrants-large-disasters-v1). This is a publicly available administrative dataset containing household-level records for every registrant who applied for FEMA Individual Assistance housing aid following a federally declared large-scale disaster. Each row represents one household's application and includes self-reported demographic information collected at registration, physical damage assessments recorded during inspector visits, and final aid eligibility determinations made by FEMA. The dataset covers disaster events primarily across FL, TX, PR, and LA and contains 6,367,325 raw registrant records across 42 fields. The file was downloaded directly from the OpenFEMA API in CSV format and is 979 MB uncompressed.

### Code table

| File | Description | Link |
|---|---|---|
| `src/load.py` | Loads raw CSV into DuckDB using `read_csv_auto`, prints null audit and column type summary, writes to `pipeline.log` | [src/load.py](src/load.py) |
| `src/clean.py` | Filters to inspected households, imputes missing income, fills structural zeros, normalizes into 4 relational tables, exports CSVs | [src/clean.py](src/clean.py) |
| `src/analysis.py` | Runs 4 exploratory SQL queries across joined tables, builds ML feature matrix, generates 3 publication-quality figures | [src/analysis.py](src/analysis.py) |
| `src/model.py` | Trains Random Forest classifier and Gradient Boosting regressor, evaluates performance, generates confusion matrix and feature importance charts | [src/model.py](src/model.py) |

### Bias identification

Four sources of bias are present in this dataset and must be acknowledged. First, **inspection bias**: only ~35% of registrants were physically inspected. The decision of who receives an inspection is not random — it may be influenced by geographic accessibility, inspector capacity, or applicant follow-through. By restricting modeling to inspected rows, we risk learning patterns that reflect inspection access rather than true damage severity. Second, **income reporting bias**: approximately 18% of registrants did not report gross income. Non-reporting likely correlates with distrust of government, language barriers, or lower digital literacy — all of which correlate with vulnerability — meaning imputed income values may underestimate need for the most at-risk households. Third, **geographic concentration**: over 97% of records come from four states (FL, TX, PR, LA), meaning models trained on this data may not generalize well to other regions with different housing stock or disaster types. Fourth, **self-reporting bias**: household composition, income, and insurance status are self-reported and unverified, introducing potential for both honest error and strategic misreporting.

### Bias mitigation

Inspection bias is mitigated by restricting analysis to inspected households and clearly documenting this scope limitation — the model is intended to support triage *after* inspection teams are deployed, not replace inspection. Income imputation uses state-level medians rather than a global median to preserve geographic income variation, and a binary flag column (`grossIncome_imputed`) is retained in the feature matrix so models can learn whether imputation occurred. Geographic concentration is documented as a known limitation; disaster number is retained as a feature so the model can learn disaster-specific patterns. Self-reported values are treated as-is, consistent with how FEMA itself uses this data in eligibility determinations.

### Rationale for critical decisions

**Decision 1 — Filter to `inspected = 1` AND `habitabilityRepairsRequired IS NOT NULL`:** Our primary ML target is only recorded when a physical inspection occurred. Of 6,367,325 raw rows, 4,091,508 (64%) have a null value for this column. Including these rows would introduce noise with no signal. After filtering, 2,275,817 rows remain — more than sufficient for robust modeling.

**Decision 2 — Fill dollar amount columns with 0:** Columns such as `repairAmount`, `replacementAmount`, `foundationDamageAmount`, and `roofDamageAmount` are null when no damage of that type occurred — not because the value is unknown. A null `foundationDamageAmount` means the foundation was not damaged. Replacing with 0 is semantically correct.

**Decision 3 — Drop `highWaterLocation` entirely:** This column is 100% null across all 6.3 million rows and contains zero information content.

**Decision 4 — Drop rental resource and end date columns:** `rentalResourceCity`, `rentalResourceStateAbbreviation`, `rentalResourceZipCode`, and `rentalAssistanceEndDate` are 99.8% null and describe post-aid logistics rather than household conditions at registration. They are not predictive features.

**Decision 5 — Impute `grossIncome` with state-level median:** Income is a critical predictor of both damage severity and aid eligibility. Dropping 18% of rows would disproportionately remove lower-income households, introducing worse selection bias than imputation itself.

**Decision 6 — Sample 500,000 rows for ML training:** The full 2.27M row dataset is computationally expensive for ensemble models. A stratified sample of 500k rows (22% of the data) retains the class distribution and is statistically more than sufficient for reliable model training.

---

## Metadata

### Schema

The four tables share `id` (VARCHAR, UUID format) as their primary key. All relationships are one-to-one — each registrant appears exactly once in each table. See the ER diagram in the repository root.

```
registrants (id PK)
    ||--|| damage_assessment (id PK+FK)
    ||--|| assistance_outcomes (id PK+FK)
    ||--|| location (id PK+FK)
```
<img width="1024" height="824" alt="image" src="https://github.com/user-attachments/assets/3098f177-a8bd-41c5-a7fc-62d8cd5b00df" />

### Data table

| Table | Description | Link |
|---|---|---|
| registrants.csv | Household demographics: income, composition, tenure, insurance, residency | [registrants.csv](https://myuva-my.sharepoint.com/:x:/g/personal/uhe5bj_virginia_edu/IQD-H_OqaKLZQa_d2RHnRZ78AcnMCxLkduNGoHWSh1JMkJI?e=TXEj8N) |
| damage_assessment.csv | Physical damage indicators from inspector visits: flood, roof, foundation damage, water level, property loss | [damage_assessment.csv](https://myuva-my.sharepoint.com/:x:/g/personal/uhe5bj_virginia_edu/IQB9HZ2wVSuFQbNVX7dZmY9iAQdgonTp71GDSfgdtWnvVo8?e=lPShC6) |
| assistance_outcomes.csv | FEMA aid eligibility and dollar amounts: rental, repair, replacement, TSA, SBA | [assistance_outcomes.csv](https://myuva-my.sharepoint.com/:x:/g/personal/uhe5bj_virginia_edu/IQD2nYqkHBMBT7UDsi8IkUraASIS492a4jp82emM_G_XYqg?e=qP6XxM) |
| location.csv | Geographic identifiers: city, state, zip, census block, disaster number | [location.csv](https://myuva-my.sharepoint.com/:x:/g/personal/uhe5bj_virginia_edu/IQCsilpCooMsRqzY6gmEVbHCAYwYu4OWHDiatAOpP4eBQ74?e=huQBLp) |

### Data dictionary

| Feature | Table | Type | Description | Example | Uncertainty |
|---|---|---|---|---|---|
| id | all | VARCHAR | UUID primary key, unique per registrant | d214ed1e-951e-... | None — system generated |
| householdComposition | registrants | BIGINT | Number of people in household | 3 | Self-reported; may exclude temporary residents |
| grossIncome | registrants | DOUBLE | Annual household gross income (USD) | 28000.0 | Self-reported; 18% imputed with state median |
| grossIncome_imputed | registrants | BIGINT | 1 if grossIncome was imputed, 0 if observed | 0 | Flag only; no uncertainty |
| specialNeeds | registrants | BIGINT | 1 if household has special needs member | 0 | Self-reported; likely undercounted |
| ownRent | registrants | VARCHAR | Whether household owns or rents | Renter | Self-reported |
| residenceType | registrants | VARCHAR | Type of dwelling | House/Duplex | Self-reported |
| homeOwnersInsurance | registrants | BIGINT | 1 if household has homeowners insurance | 0 | Self-reported; may be inaccurate |
| floodInsurance | registrants | BIGINT | 1 if household has flood insurance | 1 | Self-reported; may be inaccurate |
| primaryResidence | registrants | BIGINT | 1 if damaged home is primary residence | 1 | Self-reported |
| inspected | damage_assessment | BIGINT | 1 if FEMA inspector visited the property | 1 | Administrative; reliable |
| habitabilityRepairsRequired | damage_assessment | BIGINT | 1 if home requires repairs to be livable (ML target) | 1 | Inspector assessed; subject to inspector variability |
| destroyed | damage_assessment | BIGINT | 1 if home is destroyed | 0 | Inspector assessed |
| waterLevel | damage_assessment | BIGINT | Water intrusion level in feet | 3 | Inspector measured; 51% were 0 (no flooding) |
| floodDamage | damage_assessment | BIGINT | 1 if flood damage present | 1 | Inspector assessed |
| foundationDamage | damage_assessment | BIGINT | 1 if foundation damage present | 0 | Inspector assessed |
| foundationDamageAmount | damage_assessment | DOUBLE | Dollar value of foundation damage | 0.0 | 99.5% are 0 (structural zero) |
| roofDamage | damage_assessment | BIGINT | 1 if roof damage present | 1 | Inspector assessed |
| roofDamageAmount | damage_assessment | DOUBLE | Dollar value of roof damage (USD) | 79.3 | 96.3% are 0; high variance in non-zero values |
| rpfvl | damage_assessment | DOUBLE | Real property field visit loss (USD) | 5069.74 | Inspector assessed; right-skewed distribution |
| ppfvl | damage_assessment | DOUBLE | Personal property field visit loss (USD) | 262.5 | Inspector assessed; 51% are 0 |
| tsaEligible | assistance_outcomes | BIGINT | 1 if eligible for transitional shelter | 0 | FEMA determined |
| tsaCheckedIn | assistance_outcomes | BIGINT | 1 if household used TSA hotel program | 0 | Administrative |
| rentalAssistanceEligible | assistance_outcomes | BIGINT | 1 if eligible for rental assistance | 1 | FEMA determined |
| rentalAssistanceAmount | assistance_outcomes | DOUBLE | Dollar amount of rental assistance awarded | 1061.0 | 84.2% are 0 (structural zero) |
| repairAssistanceEligible | assistance_outcomes | BIGINT | 1 if eligible for repair assistance | 0 | FEMA determined |
| repairAmount | assistance_outcomes | DOUBLE | Dollar amount of repair assistance awarded | 0.0 | 93.4% are 0 (structural zero) |
| replacementAssistanceEligible | assistance_outcomes | BIGINT | 1 if eligible for home replacement | 0 | FEMA determined |
| replacementAmount | assistance_outcomes | DOUBLE | Dollar amount of replacement assistance | 0.0 | 99.9% are 0 (structural zero) |
| sbaEligible | assistance_outcomes | BIGINT | 1 if eligible for SBA disaster loan | 0 | FEMA determined |
| renterDamageLevel | assistance_outcomes | VARCHAR | FEMA damage severity classification for renters | Renter-Major Damage | 96% null for non-renters |
| personalPropertyEligible | assistance_outcomes | BIGINT | 1 if eligible for personal property assistance | 0 | FEMA determined |
| disasterNumber | location | BIGINT | FEMA unique identifier for the disaster event | 4332 | Administrative; reliable |
| damagedCity | location | VARCHAR | City where damaged property is located | HOUSTON | Self-reported; minor spelling variants possible |
| damagedStateAbbreviation | location | VARCHAR | State abbreviation | TX | Administrative; reliable |
| damagedZipCode | location | VARCHAR | ZIP code of damaged property | 77036 | Self-reported; 0.002% null |
| censusBlockId | location | BIGINT | 15-digit census block identifier | 482014329011042 | Geocoded; 1.2% null |
| censusYear | location | BIGINT | Census year used for block assignment | 2010 | Administrative |

---

## **How the Pipeline Solves the Problem**

The goal of this project is to shift FEMA’s housing assistance process from a **reactive, inspection-based system** to a **predictive triage system**. The pipeline operationalizes this shift across five key stages:

### **1\. Data Transformation: From Raw Records to Decision Systems**

The original FEMA dataset contains over **6.3 million records** in a flat structure. This pipeline restructures that data into a relational model with four specialized tables:

-   **`registrants`**: Household characteristics.
    
-   **`damage_assessment`**: Historical inspection outcomes.
    
-   **`assistance_outcomes`**: Final aid decisions.
    
-   **`location`**: Geographic and environmental context.
    

> **Impact:** This transformation enables efficient querying and scalable analysis, turning raw administrative data into a "decision-ready" dataset.

### **2\. Learning Historical Outcomes (Classification)**

Using the structured data, the pipeline trains a **Random Forest classifier** to predict the core decision FEMA inspectors currently make on-site:

-   **Target:** `habitabilityRepairsRequired` (Binary: 0/1)
    
-   **The Logic:** By mapping **registration-time features** $\rightarrow$ **inspection outcomes**, the model replicates the decision process in advance.
    

### **3\. Quantifying Severity (Regression)**

A secondary model (**Gradient Boosting Regressor**) predicts the financial scale of the impact:

-   **Target:** `rpfvl` (Real Property Federal Verified Loss in dollars)
    
-   **The Benefit:** This provides a continuous measure of damage, allowing FEMA to distinguish between minor and critical cases and prioritize by urgency.
    

### **4\. Enabling Predictive Triage (The Core Solution)**

Together, these models allow FEMA to move away from a "wait and see" approach.

While the model is trained on historically inspected households, it is designed to be applied at the time of registration to prioritize future inspections.

| **Feature** | **Traditional Process** | **Predictive Pipeline** |
| --- | --- | --- |
| **Workflow** | Inspect $\rightarrow$ Decide | Predict $\rightarrow$ Prioritize $\rightarrow$ Act |
| **Speed** | Sequential & Reactive | Immediate & Proactive |
| **Resource Use** | First-come, first-served | High-risk households flagged first |

### **5\. Evidence of Efficacy**

The effectiveness of this approach is validated by strong model performance metrics:

-   **Classification:** **81.1% accuracy** ($F_1 = 0.755$) in predicting repair needs.
    
-   **Regression:** $R^2 = 0.66$ in estimating property loss.
    

* * *

## **Conclusion: Transforming Emergency Response**

The original problem was the inefficient allocation of emergency resources. This pipeline solves that by converting historical data into a predictive tool that reduces delays for households in need.

**The Bottom Line:** This project transforms emergency response from **sequential and reactive** to **prioritized and proactive.**


