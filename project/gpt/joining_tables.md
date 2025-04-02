Below is an outline of how you can (1) pick out the essential columns (both for the label and for your features, including sensitive attributes), and then (2) join all of these tables into a single “analysis-ready” dataframe (one row per ICU stay). 

---

## 1) Which columns do we need?

### **Target variable (Label)** 
You’ve defined “ICU mortality” as:
```
icu_mortality = 1 if dod <= outtime  else  0
```
where:
- `dod` is the date of death from `patients_clean`
- `outtime` is the ICU discharge time from `icustays_clean`

Hence, you will need:
- **`dod`** (from **`patients_clean`**)
- **`outtime`** (from **`icustays_clean`**)

### **Sensitive attributes** 
From your proposal, the sensitive attributes are:

1. **Age**  
   - In MIMIC-IV, you have `anchor_age` in **`patients_clean`**.  

2. **Gender**  
   - In MIMIC-IV, you have `gender` in **`patients_clean`**.  

3. **Race / Ethnicity**  
   - In your cleaned data, you have `race` in **`admissions_clean`** (MIMIC sometimes calls this column `ethnicity`, but you’ve labeled it as `race`).  

4. **Insurance type**  
   - Column `insurance` from **`admissions_clean`**.  

5. **Language**  
   - Column `language` from **`admissions_clean`**.  

6. **Marital status**  
   - Column `marital_status` from **`admissions_clean`**.  

7. **Socioeconomic status**  
   - Often approximated in MIMIC by combining `insurance` and/or “urban/rural” from external data. If you are simply using MIMIC data, `insurance` is a common proxy.  

### **Additional features**  
Beyond the sensitive attributes, you typically want other predictive variables (the model needs actual clinical/demographic signals beyond the protected attributes). Common ones:

1. **Admission details**  
   - `admission_type` from **`admissions_clean`** (e.g., “Emergency”, “Elective”)  
   - `first_careunit` from **`icustays_clean`** (e.g., “MICU”, “SICU”, etc.)  
   - `los` (ICU length of stay) from **`icustays_clean`** can be used, although be mindful that sometimes LOS can be highly correlated with in-hospital mortality in ways that might bias or “leak” future info. Use carefully depending on your analysis design.

2. **Diagnoses**  
   - From **`diagnoses_clean`**. Often, people either (a) encode the top-level ICD categories as binary indicators or (b) compute the total count of diagnoses.  

3. **Labs**  
   - From **`labevents_clean`** + **`d_labitems_clean`**. Typically aggregated over the ICU stay (e.g., min, max, mean, or last value).  

4. **Vitals**  
   - From **`chartevents_clean`** + **`d_items_clean`**. Similarly aggregated over the ICU stay (e.g., min, max, mean, or last 6-hour average).  

---

## 2) Joining the tables into a single dataframe

### **High-level plan**

1. **Start with** `icustays_clean` (one row per ICU stay).  
2. **Join** to `admissions_clean` on (`subject_id`, `hadm_id`).  
3. **Join** to `patients_clean` on `subject_id`.  
4. **Define** the `icu_mortality` label.  
5. **Aggregate** time-series data (labs & vitals) by (`subject_id`, `hadm_id`, `stay_id`) and merge.  
6. **Aggregate/encode** diagnoses by (`subject_id`, `hadm_id`) and merge.  
7. **(Optional)** Do any final filtering (e.g. adult patients only) and handle missing data.  

### **Example join steps in code**

Below is an illustrative snippet. You can adapt to your variable names and chunked reading strategy.

```python
import pandas as pd

##############################################################################
# 1. Start with icustays_clean (one row per ICU stay).
##############################################################################
df = icustays_clean.copy()  
# columns: subject_id, hadm_id, stay_id, intime, outtime, los, first_careunit

##############################################################################
# 2. Join to admissions_clean on (subject_id, hadm_id).
##############################################################################
df = df.merge(
    admissions_clean[[
        'subject_id', 'hadm_id',
        'admittime', 'dischtime',
        'insurance', 'language',
        'marital_status', 'race',
        'admission_type'
    ]],
    on=['subject_id','hadm_id'],
    how='left'
)

##############################################################################
# 3. Join to patients_clean on subject_id
##############################################################################
df = df.merge(
    patients_clean[['subject_id', 'gender', 'anchor_age', 'dod']],
    on='subject_id',
    how='left'
)

##############################################################################
# 4. Define ICU mortality label: (dod <= outtime)
##############################################################################
df['icu_mortality'] = (
    (df['dod'].notnull()) & (df['dod'] <= df['outtime'])
).astype(int)

# Now df has columns for your label (icu_mortality), plus
# demographic/sensitive attributes, and timestamps for the ICU stay.

##############################################################################
# 5. Aggregate chartevents (vitals) over the ICU stay
##############################################################################
# Typically: group by (subject_id, hadm_id, stay_id), filter charttime 
# within [intime, outtime], then compute stats.
# For large data, you’ll do this in chunks or with an efficient groupby.

# Example pseudo-code:
vitals = chartevents_clean.copy()
# Ensure charttime is in [intime, outtime]
vitals = vitals.merge(
    df[['subject_id','hadm_id','stay_id','intime','outtime']],
    on=['subject_id','hadm_id','stay_id'],
    how='inner'
)

# Filter only rows where charttime is within the ICU stay
vitals = vitals[
    (vitals['charttime'] >= vitals['intime']) &
    (vitals['charttime'] <= vitals['outtime'])
]

# Then group by itemid or by “vital sign label” (if you map itemid -> label),
# and compute aggregated features. For example:
vitals_agg = vitals.groupby(
    ['subject_id','hadm_id','stay_id','itemid']
)['valuenum'].agg(['min','max','mean','sum']).reset_index()

# Pivot so that each itemid’s aggregated stats become columns.
vitals_agg = vitals_agg.pivot_table(
    index=['subject_id','hadm_id','stay_id'],
    columns='itemid',
    values=['min','max','mean','sum']
)
# Flatten multi-level column names if needed
vitals_agg.columns = [
    f'vital_{stat}_{itemid}'
    for stat, itemid in vitals_agg.columns
]
vitals_agg = vitals_agg.reset_index()

# Merge back to df
df = df.merge(vitals_agg, on=['subject_id','hadm_id','stay_id'], how='left')

##############################################################################
# 6. Aggregate labevents (labs) similarly
##############################################################################
labs = labevents_clean.copy()
# Usually you want labs within the ICU stay time window too, 
# so similarly merge with (intime, outtime) and filter charttime in [intime, outtime].
# Then group by itemid and pivot.

# (Exact same pattern as vitals, just different source table.)
# ...
# Suppose you end up with labs_agg, then:
df = df.merge(labs_agg, on=['subject_id','hadm_id','stay_id'], how='left')

##############################################################################
# 7. Aggregate/encode diagnoses
##############################################################################
# For diagnoses, we typically only have (subject_id, hadm_id) but not stay_id.
# So we can do a groupby on (subject_id, hadm_id).
# For example, get the top-level ICD categories or just do a count of diagnoses:
diag_agg = (
    diagnoses_clean
    .groupby(['subject_id','hadm_id'])['icd_code']
    .count()
    .reset_index(name='num_diagnoses')
)

# Merge that into df
df = df.merge(diag_agg, on=['subject_id','hadm_id'], how='left')

##############################################################################
# 8. Final cleanup
##############################################################################
# - Possibly restrict to adult patients: df = df[df['anchor_age'] >= 18]
# - Fill or drop missing data
# - Re-check that you still have 1 row per (subject_id, hadm_id, stay_id)
# - Confirm you have the columns you want for modeling
```

### **Final feature set for modeling**
After these joins, you will end up with a wide dataframe, something like:

- **ID columns**:  
  `subject_id, hadm_id, stay_id`  
- **Label**:  
  `icu_mortality`  
- **Sensitive attributes**:  
  `gender, anchor_age, race, insurance, language, marital_status, ...`  
- **Other demographic/clinical features**:  
  `admission_type, first_careunit, los, num_diagnoses, ...`  
- **Aggregated vitals**:  
  e.g. `vital_min_220045`, `vital_mean_220045`, etc.  
- **Aggregated labs**:  
  e.g. `lab_min_50912` (Creatinine), `lab_max_51222` (WBC), etc.  

When feeding this into XGBoost (or any other classifier), you typically:
1. Drop the ID columns (`subject_id`, `hadm_id`, `stay_id`) and any time/date columns.  
2. Keep your label (`icu_mortality`) separate as `y`.  
3. Use the rest as `X`.  

---

## Summary

1. **Essential columns** for XGBoost (and your fairness analysis) will be:  
   - **Label**: `icu_mortality` (defined via `dod` and `outtime`)  
   - **Sensitive attributes**: `gender`, `anchor_age`, `race`, `insurance`, `language`, `marital_status`  
   - **Additional predictive columns**: `first_careunit`, `admission_type`, aggregated labs & vitals, possibly diagnoses info.  

2. **Joining** is typically done by starting with one row per ICU stay (`icustays_clean`) and left-merging the relevant columns from `patients_clean` and `admissions_clean`. Then you create the `icu_mortality` label. After that, you pivot/aggregate time-series data (from `chartevents_clean` and `labevents_clean`) and diagnoses (from `diagnoses_clean`) so that each ICU stay ends up as a single row with all your features.  

Once you have that final, wide dataframe, you are ready to train your XGBoost model and carry out fairness analysis on the resulting predictions.