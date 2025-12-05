# CareWatch – MLOps Use Cases and Pipelines (Assignment A10)

## 1. ML Use Cases

### Use Case 1: Food Recommendation Engine
Given:
- User health profile (age, weight, conditions like diabetes / hypertension)
- Meal input (food name / dish) or planned meal
The system will:
- Analyze nutrients and risks
- Suggest healthier alternatives or portion sizes
- Recommend daily/weekly diet suggestions

### Use Case 2: Medical Report Risk Analysis
Given:
- Uploaded medical report (PDF) processed by OCR
The system will:
- Extract key metrics (e.g., glucose, cholesterol, blood pressure)
- Classify risk level (normal / borderline / high risk)
- Suggest diet focus areas (e.g., “low sodium”, “low sugar”)

---

## 2. MLOps Pipelines per Use Case

### 2.1 Pipelines for Use Case 1 – Food Recommendation Engine

We will implement the following pipeline stages:

1. **Data Collection Pipeline**
   - Input: Nutrition datasets (USDA / custom CSV like `carewatch_master_food_clean.csv`)
   - Task: Load raw CSVs, keep only needed columns, save cleaned dataset.
   - Output: Cleaned dataset for EDA and training.

2. **Preprocessing + Training Pipeline**
   - Input: Cleaned food dataset.
   - Task:
     - Encode categorical features (food type, category)
     - Normalize numeric nutrients (calories, fat, sodium, sugar, etc.)
     - Train a simple model (e.g., rule-based or a regression/classifier) to score how “healthy” a food is.
   - Output: Saved model/pipeline object (e.g., `nutrition_model.pkl`).

3. **Validation Pipeline**
   - Input: Trained model + test/validation split.
   - Task: Evaluate model (accuracy / precision / recall or custom health score).
   - Output: Metrics logged in console / file.

4. **Model Serving Pipeline**
   - Input: User food input + health profile.
   - Task:
     - Run preprocessing
     - Call trained model / rules
     - Return recommendation + explanation.
   - Output: API / function used by Streamlit UI.

---

### 2.2 Pipelines for Use Case 2 – Medical Report Risk Analysis

We will implement the following pipeline stages:

1. **Data Collection / Ingestion Pipeline**
   - Input: Uploaded PDF report.
   - Task: Run OCR, extract text, parse lab values into a structured format (DataFrame/dict).
   - Output: Structured medical metrics.

2. **Preprocessing Pipeline**
   - Input: Extracted metrics (e.g., glucose, HbA1c, cholesterol, BP).
   - Task:
     - Handle missing values
     - Convert strings to numeric
     - Normalize ranges where needed.
   - Output: Cleaned feature vector.

3. **Risk Scoring / Rule-Based “Model” Pipeline**
   - Input: Cleaned metrics.
   - Task:
     - Apply threshold-based rules (e.g., high sugar → high risk)
     - Optionally apply a simple ML model later.
   - Output: Risk label (normal / borderline / high) + reasons.

4. **Integration / Serving Pipeline**
   - Input: Risk label + user health profile.
   - Task:
     - Combine risk score with food recommendation module
     - Generate personalized diet recommendation screen in Streamlit.
   - Output: Response used by Medical Report tab in the UI.

---

## 3. Link to Planned API Services (for later steps)

For each use case, these pipelines will be encapsulated behind API-like functions, for example:

- `POST /api/food/predict` → Uses Food Recommendation Serving Pipeline.
- `POST /api/report/analyze` → Uses Medical Report pipelines (OCR → preprocess → risk scoring).
- `POST /api/food/train` → Triggers training pipeline (can be run manually).
- `POST /api/food/validate` → Returns latest metrics from Validation Pipeline.

(Detailed service contracts will be defined in the next step.)

---

## 4. Detailed Service Contracts – Use Case 1 (Food Recommendation Engine)

Below are the API-style service contracts for the Food Recommendation use case.

### 4.1 Service 1 – Refresh Food Dataset (Data Collection)

- **Name**: `RefreshFoodDataset`
- **Endpoint**: `POST /api/food/data/refresh`
- **Purpose**: Reload and clean the nutrition data from CSV files.
- **Parameters**:
  - `source_path` (string, required) – path to raw CSV file or folder.
  - `output_path` (string, required) – path to save cleaned dataset.
- **Returns**:
  - `status` (string) – e.g., "success" / "error".
  - `rows_processed` (int) – number of rows loaded/cleaned.

---

### 4.2 Service 2 – Train Food Health Model (Preprocessing + Training)

- **Name**: `TrainFoodHealthModel`
- **Endpoint**: `POST /api/food/train`
- **Purpose**: Train the nutrition / health scoring model using cleaned data.
- **Parameters**:
  - `dataset_path` (string, required) – path to cleaned dataset.
  - `model_output_path` (string, required) – where to save model/pipeline (`.pkl`).
  - `test_size` (float, optional, default = 0.2) – train/test split ratio.
- **Returns**:
  - `status` (string) – "success" / "error".
  - `train_rows` (int) – number of training samples.
  - `test_rows` (int) – number of test samples.

---

### 4.3 Service 3 – Validate Food Model (Validation)

- **Name**: `ValidateFoodModel`
- **Endpoint**: `POST /api/food/validate`
- **Purpose**: Evaluate the trained model on a test set.
- **Parameters**:
  - `dataset_path` (string, required) – path to cleaned dataset.
  - `model_path` (string, required) – saved model file path.
- **Returns**:
  - `accuracy` (float, optional) – if using classification.
  - `mae` (float, optional) – if using regression.
  - `custom_health_score_metric` (float, optional).
  - `status` (string).

---

### 4.4 Service 4 – Get Food Recommendation (Model Serving)

- **Name**: `GetFoodRecommendation`
- **Endpoint**: `POST /api/food/recommend`
- **Purpose**: Given a user health profile and a food item, return recommendation.
- **Parameters**:
  - `food_name` (string, required) – user-entered dish / item.
  - `age` (int, optional).
  - `weight` (float, optional).
  - `conditions` (list[string], optional) – e.g., `["diabetes", "hypertension"]`.
- **Returns**:
  - `recommended_foods` (list[object]) – list of suggested alternatives or portion sizes.
  - `message` (string) – human-friendly explanation.
  - `risk_level` (string, optional) – e.g., "low", "medium", "high".

---

### 4.5 Service 5 – Get Food Details by ID (Integration Helper)

- **Name**: `GetFoodDetails`
- **Endpoint**: `GET /api/food/item/{food_id}`
- **Purpose**: Fetch detailed nutrient information for a specific food item.
- **Parameters**:
  - `food_id` (string, path parameter, required).
- **Returns**:
  - `food_name` (string).
  - `nutrients` (object) – calories, carbs, protein, fat, sodium, sugar, etc.

---

### 4.6 Service 6 – Explain Food Health Score (Explainability)

- **Name**: `ExplainFoodScore`
- **Endpoint**: `POST /api/food/explain`
- **Purpose**: Explain why a given food was classified as recommended / risky.
- **Parameters**:
  - `food_name` (string, required).
  - `conditions` (list[string], optional).
- **Returns**:
  - `score` (float).
  - `factors` (list[string]) – e.g., "High sodium", "High sugar".
  - `explanation` (string).

---

## 5. Detailed Service Contracts – Use Case 2 (Medical Report Risk Analysis)

### 5.1 Service 1 – Upload & Extract Report (Data Ingestion + OCR)

- **Name**: `UploadAndExtractReport`
- **Endpoint**: `POST /api/report/upload`
- **Purpose**: Receive a PDF, run OCR, and extract plain text.
- **Parameters**:
  - `file` (binary, required) – uploaded PDF medical report.
- **Returns**:
  - `report_id` (string) – internal ID to reference this report.
  - `raw_text` (string) – extracted text.
  - `status` (string).

---

### 5.2 Service 2 – Parse Medical Metrics (Preprocessing)

- **Name**: `ParseMedicalMetrics`
- **Endpoint**: `POST /api/report/parse`
- **Purpose**: Convert extracted text into structured medical metrics.
- **Parameters**:
  - `report_id` (string, required).
  - `raw_text` (string, optional, if not using ID).
- **Returns**:
  - `metrics` (object) – key-value pairs like:
    - `glucose_fasting`, `cholesterol_total`, `systolic_bp`, `diastolic_bp`, etc.
  - `status` (string).

---

### 5.3 Service 3 – Compute Risk Score (Risk “Model”)

- **Name**: `ComputeRiskScore`
- **Endpoint**: `POST /api/report/risk-score`
- **Purpose**: Apply rule-based or ML logic to classify risk level.
- **Parameters**:
  - `metrics` (object, required) – structured metrics from parsing step.
  - `age` (int, optional).
  - `conditions` (list[string], optional).
- **Returns**:
  - `risk_level` (string) – e.g., "normal", "borderline", "high".
  - `reasons` (list[string]) – e.g., "Glucose above normal range".
  - `status` (string).

---

### 5.4 Service 4 – Generate Diet Recommendation From Report (Integration + Serving)

- **Name**: `GenerateDietFromReport`
- **Endpoint**: `POST /api/report/recommend-diet`
- **Purpose**: Combine risk scoring with food recommendation logic.
- **Parameters**:
  - `metrics` (object, required).
  - `conditions` (list[string], optional).
- **Returns**:
  - `diet_plan` (list[object]) – suggested meal ideas / focus areas.
  - `focus_areas` (list[string]) – e.g., "low sodium", "low sugar".
  - `status` (string).

---

### 5.5 Service 5 – Get Report History (Integration)

- **Name**: `GetReportHistory`
- **Endpoint**: `GET /api/report/history/{user_id}`
- **Purpose**: Fetch list of previous analyzed reports for a user.
- **Parameters**:
  - `user_id` (string, path parameter, required).
- **Returns**:
  - `reports` (list[object]) – each with `report_id`, `date`, `risk_level`.
  - `status` (string).

---

### 5.6 Service 6 – Check & Confirm Latest Report

- **Name**: `CheckAndConfirmLatestReport`
- **Endpoint**: `POST /api/report/latest-check`
- **Purpose**: Make sure the uploaded report is the latest before giving recommendations.
- **Parameters**:
  - `user_id` (string, required).
  - `report_date` (string/date, required).
- **Returns**:
  - `is_latest` (bool) – true if this is the latest report.
  - `message` (string) – e.g., "This is not your latest report. Please upload the newest one."
  - `status` (string).
