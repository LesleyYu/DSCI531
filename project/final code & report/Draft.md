# **Fair ICU Mortality Prediction: Logistic Regression, XGBoost, and MLP Models**







## **1. Introduction**





Predicting in‐ICU mortality is a critical task in modern healthcare analytics. Accurate forecasts of patient risk can inform clinical decision‐making, optimize resource allocation, and ultimately improve patient outcomes. However, high predictive performance alone is insufficient: models must also operate fairly across demographic groups. Unchecked algorithmic biases can reinforce existing health disparities, undermining trust and perpetuating inequities in care delivery.



Our project—titled **“Fair ICU Mortality Prediction”**—aims to develop, evaluate, and compare three classes of predictive models (Logistic Regression, XGBoost, and Multilayer Perceptron) on the MIMIC‑IV dataset. We pursue two intertwined goals: (1) maximize discrimination and calibration for mortality risk, and (2) diagnose and mitigate biases with respect to sensitive attributes (gender, age, and race). In this report, we present our **research questions**, **objectives**, **methodology**, **results**, and **discussion**, culminating in recommendations for deploying equitable clinical risk models.





### **1.1 Research Questions**





This study addresses the overarching question:



> **How can predictive models for ICU mortality be developed and evaluated to ensure both high accuracy and fairness across demographic subgroups?**



To operationalize this goal, we examine four sub‑questions:



1. **Bias Sources:** Which sensitive attributes contribute most to disparities in model outcomes?
2. **Model Comparison:** How do Logistic Regression (LR), XGBoost, and MLP differ in accuracy–fairness trade‑offs?
3. **Metric Selection:** Which fairness metrics (Equal Opportunity, Statistical Parity, Equalized Odds, Calibration) most sensitively detect subgroup biases?
4. **Mitigation Strategies:** What bias‑reduction techniques can improve fairness without unduly sacrificing performance?







### **1.2 Objectives**





1. **Data Preparation:** Curate and preprocess the MIMIC‑IV ICU cohort, extracting demographics, vital signs, and lab features.
2. **Baseline Modeling:** Train LR, XGBoost, and MLP models—first excluding sensitive features (Baseline), then including them (Full).
3. **Threshold Tuning:** Select classification thresholds via F₁‐score and Precision–Recall trade‑off.
4. **Fairness Audit:** Quantify subgroup gaps in True Positive Rate (TPR), False Positive Rate (FPR), Positive Prediction Rate (PPR), and calibration.
5. **Bias Mitigation:** Implement three pipelines (reweighting, exponentiated gradients, adversarial debiasing) on top of the Full LR model and replicate for XGBoost and MLP.
6. **Comparative Analysis:** Evaluate performance–fairness trade‑offs across all models and mitigation methods to identify optimal configurations.





------





## **2. Methodology**







### **2.1 Data Description**





We leverage the publicly available MIMIC‑IV v3.1 database, comprising over 50,000 ICU admissions at Beth Israel Deaconess Medical Center (2008–2019) \cite{johnson2023mimiciv}. Key tables include:



- **core/patients**, **core/admissions**, **icu/icustays** (demographics and ICU timelines)
- **hosp/labevents**, **icu/chartevents**, **icu/outputevents** (laboratory and vital signs)





After applying inclusion criteria, our cohort contains **94,444** ICU stays with a mortality rate of ~8%.





### **2.2 Preprocessing**





A multi‐step pipeline in Python (Pandas, NumPy) was developed:



1. **Table Merging:** Align records via subject_id and stay_id to assemble a unified DataFrame with demographics, vital signs, and labs.

2. **Missing Data Handling:**

   

   - **Outcome (dod/outtime):** Missing date‐of‐death implies survival; rows missing ICU discharge times were removed.
   - **Numeric Features:** Rows with non‐numeric or null measurements in valuenum were excluded due to scale.
   - **Categorical Features:** Rare categories in insurance, language, marital status were set to “UNKNOWN”; primary focus was on gender, age, and race.

   

3. **Feature Engineering:**

   

   - **Demographics:**

     

     - **Gender:** Binary encoding (0 = Female, 1 = Male)
     - **Race:** Consolidated into seven groups (WHITE, BLACK, ASIAN, HISPANIC/LATINO, OTHER, UNKNOWN, EUROPEAN) via map_race() \cite{meng2022interpretability}.
     - **Age Brackets:** Five bins (18–29, 30–49, 50–64, 65–79, 80–91) to reduce dimensionality and capture nonlinearity.

     

   - **Clinical Measurements:** For each selected vital sign (heart rate, respiratory rate, systolic/diastolic blood pressure, temperature) and top‑5 lab tests, compute min/max/mean and flag missingness.

   

4. **Imputation & Scaling:** Continuous features standardized (zero mean, unit variance).

5. **Data Split:** Stratified 80/20 train/test partition preserving mortality prevalence.







### **2.3 Model Training and Threshold Selection**





We trained three base learners on the same feature set:



1. **Logistic Regression (LR):**

   

   - Regularizer: ℓ2 penalty; solver: liblinear; max iter = 1,000.
   - **SMOTE** oversampling on training data to address class imbalance (minority class ~8%).

   

2. **XGBoost:**

   

   - Booster: gbtree; parameters tuned via 5‑fold CV (max_depth, η, colsample_bytree, scale_pos_weight).
   - Early stopping on validation AUC.

   

3. **Multilayer Perceptron (MLP):**

   

   - Architecture: input layer → dense(128, ReLU) → dense(64, ReLU) → output(sigmoid).
   - Optimizer: Adam (lr = 1e‑3); epochs = 50; batch = 256.

   





For each model, we first trained a **Baseline** variant (excluding gender, age, race) and a **Full** variant (including these features). We then selected optimal classification thresholds by maximizing the F₁‐score on validation data, balancing precision (avoiding false positives) against recall (detecting true deaths).





### **2.4 Fairness Diagnosis**





On the held‑out test set, we computed subgroup metrics for **Gender**, **Age**, and **Race**:



- **Equal Opportunity (ΔTPR):** Gap in recall between groups.
- **Statistical Parity (ΔPPR):** Gap in positive prediction rates.
- **Equalized Odds (ΔFPR & ΔTPR):** Combined disparity in both false and true positive rates.
- **Calibration:** Reliability curves comparing predicted probabilities to observed outcomes.





We report **∆ metric vs. Baseline** for each subgroup, highlighting the magnitude and direction of fairness shifts induced by including sensitive features and by each mitigation pipeline.





### **2.5 Bias Mitigation Strategies**





To reduce identified disparities—most pronounced across race—we designed three mitigation pipelines applied to the **Full** LR model and replicated for XGBoost and MLP:



1. **SMOTE + Reweighting (BM1):**

   

   - **Group Definition:** Intersectional strata combining gender, age bracket, and race.
   - **Weight Computation:** Inverse of group sample counts, normalized to sum to one.
   - **Training:** Apply sample weights during LR fitting on SMOTE‑balanced data.

   

2. **SMOTE + Exponentiated Gradient (BM2):**

   

   - **Framework:** Fairlearn’s ExponentiatedGradient wrapper around the base learner.
   - **Constraint:** EqualizedOdds; slack parameter ε = 0.1.
   - **Procedure:** Fit on SMOTE‑balanced data with race as the sensitive feature.

   

3. **SMOTE + Adversarial Debiasing (BM3):**

   

   - **Predictor:** Simple one‐neuron sigmoid network.
   - **Adversary:** Two‐layer softmax network predicting race given the predictor’s joint output.
   - **Classifier:** Fairlearn’s AdversarialFairnessClassifier optimizing equalized odds through joint training.

   





Each pipeline preserves the same threshold‐selection process. We similarly applied BM1–BM3 to XGBoost and MLP by wrapping their estimators in the respective debiasing frameworks and re‑tuning thresholds on validation splits.



------





## **3. Results**







### **3.1 Logistic Regression**







#### **3.1.1 Performance vs. Baseline**



| **Model** | **F₁ Score Δ** | **AUC (Base)** | **AUC (Full)** |
| --------- | -------------- | -------------- | -------------- |
| Baseline  | –              | 0.912          | —              |
| Full      | +0.002         | —              | 0.920          |
| BM1       | –0.028         | 0.898          | —              |
| BM2       | –0.032         | 0.876          | —              |
| BM3       | –0.074         | 0.916          | —              |

Including sensitive features yielded a modest F₁ and AUC gain. Mitigation pipelines traded overall F₁ for reduced subgroup gaps.





#### **3.1.2 Fairness vs. Baseline**





- **Equal Opportunity (EO):**

  

  - *Full* decreased race-gap by 0.025; gender-gap slight +0.013.
  - *BM1/BM2/BM3* progressively narrowed race EO-gap to near zero (BM3 Δ ≈ –0.19).

  

- **Statistical Parity (PPR):**

  

  - *BM1* increased age PPR-gap (Δ ≈ +0.23) due to aggressive reweighting.
  - *BM2/BM3* balanced PPR across groups within Δ < 0.05.

  

- **Equalized Odds (TPR & FPR):**

  

  - Adversarial (BM3) achieved the smallest combined Δ, at the cost of a larger drop in overall F₁.

  





Plots in Figures 1–5 illustrate these ∆‐metrics for LR.





### **3.2 XGBoost**







#### **3.2.1 Performance vs. Baseline**





- *Baseline AUC:* 0.935; *Full AUC:* 0.942
- *F₁ Δ (Full vs. Base):* +0.018







#### **3.2.2 Fairness vs. Baseline**





- Greater baseline disparities by race (EO-gap ≈0.08) compared to LR.
- **BM1 (Reweighting):** EO-gap race reduced by 0.06; slight PPR overcorrection.
- **BM2 (ExpGrad):** Balanced EO within Δ 0.02; F₁ Δ ≈ –0.015.
- **BM3 (Adversarial):** EO-gap near zero; FPR-gap remained under 0.03.





Figure 6 (XGBoost Metrics vs. Baseline) and Figure 7 (Fairness vs. Performance) show trade‑offs. XGBoost’s greater capacity yielded higher initial AUC and F₁ but required stronger mitigation to match LR’s fairness.





### **3.3 Multilayer Perceptron (MLP)**







#### **3.3.1 Performance vs. Baseline**





- *Baseline AUC:* 0.927; *Full AUC:* 0.933
- *F₁ Δ (Full vs. Base):* +0.022







#### **3.3.2 Fairness vs. Baseline**





- Baseline MLP exhibited the largest race‐based EO-gap (~0.12).
- **BM1:** Race EO-gap halved (Δ ≈ –0.06) but PPR-gap inflated.
- **BM2:** Achieved balanced EO (Δ < 0.03) with minimal F₁ loss (–0.020).
- **BM3:** Nearly eliminated EO-gap; F₁ Δ ≈ –0.055, indicating a steeper fairness–accuracy trade.





Figures 8–10 depict MLP’s performance–fairness scatter and ∆‐bar charts.



------





## **4. Discussion**





Our comparative analysis across three model families reveals several key insights:



1. **Inclusion of Sensitive Features**

   

   - Adding gender, age, and race slightly improves global discrimination (AUC ↑ by 0.005–0.010) but can introduce or accentuate subgroup disparities.
   - For LR and XGBoost, the *Full* models modestly narrowed EO-gaps for age but widened race-related EO in some cases, underscoring the importance of targeted fairness auditing.

   

2. **Mitigation Trade‑offs**

   

   - **Reweighting (BM1):** Simple to implement; effective at reducing minority‐group under‑prediction but prone to over‑compensating and inflating PPR-gaps.
   - **Exponentiated Gradient (BM2):** Consistently balanced EO within Δ 0.02 for all attributes with moderate F₁ loss (1.5–3%).
   - **Adversarial Debiasing (BM3):** Most powerful at eliminating EO‐gaps (<0.01) but incurred the highest F₁ penalty (5–7%), especially for MLP.

   

3. **Model Capacity and Fairness**

   

   - **XGBoost** delivered the highest baseline AUC (0.942) and F₁ (0.71) but also exhibited larger initial EO‐gaps (up to 0.09) due to its flexibility.
   - **MLP** showed intermediate performance and bias patterns, benefiting more from ExpGrad than from adversarial methods.
   - **LR**, while less accurate overall, demonstrated more stable fairness profiles and the smallest F₁ drop under BM2, making it an appealing choice when interpretability and fairness consistency are paramount.

   

4. **Metric Sensitivity**

   

   - **EO-gap (True Positive Rate)** emerged as the most revealing measure of subgroup under‑serving, particularly for race.
   - **Statistical Parity** metrics were more volatile under reweighting but less informative when groups had drastically different positive‐rate baselines.
   - **Calibration** analysis (not shown here) indicated that adversarial methods may slightly degrade probability reliability for small subgroups, warranting further investigation.

   

5. **Practical Considerations**

   

   - **Threshold Selection:** Operating points optimized for global F₁ do not necessarily optimize subgroup equity; future work should explore subgroup‐aware threshold tuning.
   - **Computational Cost:** ExpGrad and adversarial training add overhead—adversarial methods require careful tuning of learning rates and architectures.
   - **Clinical Deployment:** Adversarial debiasing’s heavy performance penalty may be unacceptable in high‑stakes settings; ExpGrad offers a middle ground with modest accuracy loss and strong fairness gains.

   







### **4.1 Limitations and Future Work**





- **Data Scope:** Our focus on age, gender, and race excludes other legally protected or clinically relevant attributes (e.g., ethnicity, language, insurance).
- **Single Dataset:** Findings on MIMIC‑IV may not generalize to other health systems with different case‐mixes.
- **Model Variants:** We limited ourselves to basic MLPs; exploring deeper architectures or ensemble methods could yield improved trade‑offs.





**Next steps** include:



1. Integrating **group‐specific thresholding** to further equalize performance across subgroups.
2. Extending adversarial frameworks to jointly address multiple fairness constraints (e.g., both EO and calibration).
3. Validating on external ICU datasets to assess robustness and generalizability.





------



**Conclusion.** This work demonstrates that fairness‐aware machine learning for ICU mortality prediction is both feasible and imperative. Systematic auditing coupled with targeted mitigation can substantially reduce demographic disparities. Among the methods evaluated, **Exponentiated Gradient** offers a favorable accuracy–fairness balance across LR, XGBoost, and MLP. By adopting such strategies, healthcare organizations can harness AI tools responsibly, ensuring equitable care for all patients.