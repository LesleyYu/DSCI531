# Joining tables

I am using mimic_IV dataset to do fairness analysis. Attached is the summary of my goal. Below are the dataframes and columns I now have cleaned. 


### 1. patients 

Rows: 1 per patient

KeyID: `subject_id`

| Column      | Type       | Description                                              |
|------------|-----------|----------------------------------------------------------|
| `subject_id`  | ID        | Unique patient ID                                      |
| `gender`      | 🔒 Sensitive | Male or Female                                       |
| `anchor_age`  | 🔒 Sensitive | Approximate age at time of admission                |
| `dod`         | 🧠 Label   | Date of death (used with ICU discharge to define ICU mortality) |

```Python
# Drop synthetic time columns
patients_clean = patients.copy()
patients_clean = patients_clean.drop(columns=['anchor_year', 'anchor_year_group'])

# Convert gender to binary
patients_clean['gender'] = patients_clean['gender'].map({'M': 1, 'F': 0})

# Convert dod to datetime
patients_clean['dod'] = pd.to_datetime(patients_clean['dod'])
```

patients_clean:
(364627, 4)
----------------------------------------------------------------------------------------------------
subject_id    364627
gender        364627
anchor_age    364627
dod            38301
dtype: int64


### 2. admissions

Rows: 1 per hospital admission(hadm_id)

Key IDs: `subject_id`, `hadm_id`

| Column                  | Type                 | Description                                           |
|-------------------------|----------------------|-------------------------------------------------------|
| `subject_id`           | ID                   | Patient ID                                           |
| `hadm_id`             | ID                   | Hospital admission ID                                |
| `ethnicity`           | 🔒 Sensitive         | Patient-reported race/ethnicity                     |
| `insurance`           | 🔒 Sensitive         | Primary insurance type (e.g., Medicaid, Private)    |
| `language`            | 🔒 Sensitive         | Preferred language                                  |
| `marital_status`      | 🔒 Sensitive         | Married, Single, Divorced, etc.                     |
| `admittime / dischtime` | 🧠 Predictive (optional) | Can be used to compute hospital LOS (length of stay) |

```Python
cols = ['subject_id', 'hadm_id', 'admittime', 'dischtime',
             #'insurance', 'language', 'marital_status',
            'race', 'hospital_expire_flag', 'admission_type']

admissions_clean = admissions[cols].copy()

# Convert datetime columns
admissions_clean['admittime'] = pd.to_datetime(admissions_clean['admittime'])
admissions_clean['dischtime'] = pd.to_datetime(admissions_clean['dischtime'])

# Fill missing values for sensitive attributes
'''“This particular NaN value has special meaning — it tells us that the information was unknown or missing at the time. 
Instead of dropping or guessing it, we treat it as its own category by converting it into a label like 'UNKNOWN', so the model can process it.”'''

admissions_clean[['insurance', 'language', 'marital_status']] = admissions_clean[['insurance', 'language', 'marital_status']].fillna('UNKNOWN')
```

admissions_clean:
(546028, 10)
----------------------------------------------------------------------------------------------------
subject_id              546028
hadm_id                 546028
admittime               546028
dischtime               546028
insurance               546028
language                546028
marital_status          546028
race                    546028
hospital_expire_flag    546028
admission_type          546028
dtype: int64


### 3. icustays

Rows: 1 per ICU stay

Key IDs: `subject_id`, `hadm_id`, `stay_id`

| Column      | Type             | Description                                                        |
|-------------|------------------|--------------------------------------------------------------------|
| subject_id  | ID               | Patient ID                                                         |
| hadm_id     | ID               | Hospital admission ID                                              |
| stay_id     | ID               | ICU stay ID                                                        |
| intime      | 🔮 Predictive    | ICU admission timestamp                                            |
| outtime     | 🔮 Predictive & Label | ICU discharge timestamp; used with `dod` to compute ICU mortality (`dod ≤ outtime`) |


```Python
# Drops entire rows where the column outtime is NaN
icustays_clean = icustays.copy()
icustays_clean = icustays_clean.dropna(subset=['outtime'])

# Convert intime and outtime to datetime
icustays_clean['intime'] = pd.to_datetime(icustays_clean['intime'])
icustays_clean['outtime'] = pd.to_datetime(icustays_clean['outtime'])

cols = ['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los', 'first_careunit']
icustays_clean = icustays_clean[cols]
```

icustays_clean:
(94444, 7)
----------------------------------------------------------------------------------------------------
subject_id        94444
hadm_id           94444
stay_id           94444
intime            94444
outtime           94444
los               94444
first_careunit    94444
dtype: int64


### 4. diagnoses

Rows: 1 per diagnosis

Key IDs: `subject_id`, `hadm_id`

| Column      | Type             | Description         |
|-------------|------------------|---------------------|
| subject_id  | ID               | Patient ID            |
| hadm_id     | ID               | Hospital admission ID             |
| seq_num     | label       |A pseudo-order for the ICD codes within a stay (1 most relevant, 9 the least)  |
| icd_code      | label    | The coded representation of the diagnosis using the ICD ontology      |
| icd_version   | label | The textual description of the ICD code |

```Python
cols = ['subject_id', 'hadm_id', 'icd_code']
diagnoses_clean = diagnoses_clean[cols]

# Convert icd_code to string and standardize format
diagnoses_clean = diagnoses_clean[cols]
diagnoses_clean['icd_code'] = diagnoses_clean['icd_code'].astype(str).str.strip().str.upper()
```

diagnoses_clean:
(2908741, 3)
----------------------------------------------------------------------------------------------------
subject_id    2908741
hadm_id       2908741
icd_code      2908741
dtype: int64


### 5. labevents

Rows: Time-stamped lab measurements

Key IDs: `subject_id`, `hadm_id`

| Column                | Type          | Description                                      |
|-----------------------|---------------|--------------------------------------------------|
| subject_id, hadm_id   | ID            | For joining and grouping                         |
| itemid                | Lookup        | Lab test code (mapped via `d_labitems.csv.gz`)   |
| charttime             | Time          | When the lab was drawn                           |
| valuenum              | 🔮 Predictive  | Numeric result (e.g., WBC = 12.3)                |

```Python
labevents_clean = labevents.dropna(subset=['valuenum']).copy()
labevents_clean['charttime'] = pd.to_datetime(labevents_clean['charttime'])
```

labevents_clean:
(136884423, 5)
----------------------------------------------------------------------------------------------------
subject_id    136884423
hadm_id        77365985
itemid        136884423
charttime     136884423
valuenum      136884423
dtype: int64


### 6. d_items

Rows: One per itemid(dictionary)

Key: `itemid`

| Column   | Type              | Description                                              |
|----------|-------------------|----------------------------------------------------------|
| itemid   | Lookup            | Numeric code used in `chartevents.csv.gz`                |
| label    | Descriptive       | Human-readable label like "Heart Rate", "SpO2", etc.     |
| category | Optional grouping | Sometimes helpful (e.g., "Vitals", "Labs", "Ventilation") |

```Python
# Filter only rows that belong to chartevents
d_items_clean = d_items.copy()
d_items_clean = d_items_clean[d_items_clean['linksto'] == 'chartevents']

d_items_clean['is_vital_sign'] = (d_items_clean['category'] == 'Routine Vital Signs').map({True: 1, False: 0})

# drop linksto, category after filtering and other unneeded columns 
cols = ['itemid', 'label', 'is_vital_sign']
d_items_clean = d_items_clean[cols]
```

d_items_clean:
(3055, 3)
----------------------------------------------------------------------------------------------------
itemid           3055
label            3055
is_vital_sign    3055
dtype: int64


### 7. chartevents

Rows: Time-stamped vitals & notes

Key IDs: `subject_id`, `hadm_id`, `stay_id`

| Column                  | Type         | Description      |
|-------------------------|--------------|------------------|
| `subject_id`, `hadm_id`, `stay_id`  | ID           | For joining and grouping  |
| `itemid`       | Lookup       | Code for the type of measurement (mapped via  `d_items.csv.gz`)  |
| `charttime`      | Time         | Timestamp of the recorded measurement                      |
| `valuenum`     | 🔮 Predictive | The actual numeric value (e.g., heart rate = 85)          |

```Python
# Charted ICU data (vitals)
cols = ['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'valuenum']

vital_signs_itemids = d_items_clean[d_items_clean['is_vital_sign'] == 1]['itemid'].unique()

# Read in chunks and filter
chunk_size = 500000
chunks = pd.read_csv("dataset/mimic4/icu/chartevents.csv.gz", usecols=cols, chunksize=chunk_size)

# Filter chunks and concatenate
chartevents = pd.concat(chunk[chunk["itemid"].isin(vital_signs_itemids)] for chunk in chunks)

chartevents_clean = chartevents.dropna(subset=['valuenum']).copy()
chartevents_clean['charttime'] = pd.to_datetime(chartevents_clean['charttime'])
```

chartevents_clean:
(38336836, 6)
----------------------------------------------------------------------------------------------------
subject_id    38336836
hadm_id       38336836
stay_id       38336836
charttime     38336836
itemid        38336836
valuenum      38336836
dtype: int64


### 8. d_labitems

Rows: One per lab item

Key: `itemid`

| Column | Type | Description |
|--------| ---- | ----------- |
| `itemid` | Lookup | Lab test code used in `labevents.csv.gz` |
| `label`  | Descriptive | Name like "WBC", "Creatinine", "Sodium", etc. |

```Python
d_labitems_clean = d_labitems[d_labitems['fluid'] == 'Blood'].copy()
cols = ['itemid', 'label']
d_labitems_clean = d_labitems_clean[cols]

# Drop the missing label
d_labitems_clean = d_labitems_clean.dropna(subset=['label'])
```

d_labitems_clean:
(820, 2)
----------------------------------------------------------------------------------------------------
itemid    820
label     820
dtype: int64



Please help me with the following tasks:
1. do one hot encoding


<!-- 1. Identify which collumns are needed to do this analysis using XGBost. 
2. Join these tables and columns in a format suitable for future machine learning steps. -->


# 计划

1. 只保留top5 的labevents和chartevents，重新filter
2. 找到y
   1. 看hospital_expire_flag (?)
   2. 看dod是否在icu内
3. one-hot encoding race and age
4. sensitive features只保留 age, gender, race


# 主要改动：
1. 在导入时候就filter了chartevent，用了分chunk的方式导入
2. 对age和race做了one-hot encoding，并生成了两个图用来应付中期报告
3. 最后的Joining Tables 那部分代码可以做参考，没有完全写完