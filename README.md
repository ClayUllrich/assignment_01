## **Project Overview**

This project focuses on analyzing the AnnoMI dataset and creating a text classification model to predict the `main_therapist_behaviour` label. The dataset contains annotated transcripts from motivational interviewing sessions and includes features that reflect therapist and client behaviors.

The model aims to classify therapist behaviors into one of four categories: `reflection`, `question`, `therapist_input`, or `other`. This project explores the dataset, performs necessary preprocessing, builds a machine learning pipeline, and evaluates the model using interpretability techniques.

---

## **Problem Statement**

1. **Dataset**: The dataset contains transcript annotations, including textual, categorical, and numerical features.
2. **Objective**: Create a text classification model to predict the `main_therapist_behaviour` label and evaluate its performance.
3. **Evaluation Criteria**:
   - Strong model performance (accuracy, precision, recall, and F1-score).
   - Logical, clear, and creative approaches to the solution.
   - Application of industry best practices for coding and machine learning.

---

## **Approach and Solution**

### **Step 1: Exploratory Data Analysis (EDA)**
- Performed statistical summaries of the dataset to understand distributions of categorical and numerical features.
- Visualized the dataset using:
  - Feature distributions.
  - Word clouds for `utterance_text` to identify common patterns in text data.
- Dropped the data leakage columns
- **Split the dataset into training and validation sets (80%-20%) before feature engineering to avoid data leakage**
### **Step 2: Data Preprocessing**
1. **Text Features**:
   - Clean and processed the `utterance_text` column using `TfidfVectorizer` to extract relevant textual features.
   - Limited the number of features to 5000 for memory efficiency.
2. **Categorical Features**:
   - Applied `OneHotEncoder` for categorical variables like `mi_quality` and `topic`.
3. **Numerical Features**:
   - Scaled numerical features (`utterance_id`, `transcript_id`, `annotator_id`) using `MinMaxScaler`.

### **Step 3: Model Development**
- Built multiple classification models:
  - **Logistic Regression**.
  - **Random Forest**.
  - **XGBoost**.
  - **SVM**.
  - **BERT**.
- Implemented a neural network for text embeddings using **`BERT`** to enhance predictions by combining textual and structured data features.

### **Step 4: Model Training**
- Trained models using the preprocessed features.
- Applied **early stopping** to prevent overfitting during training.

### **Step 5: Model Evaluation**
- Evaluated the models using metrics:
  - **Accuracy**, **Precision**, **Recall**, and **F1-score**.
  - Confusion matrix analysis to understand misclassification patterns.
- Applied interpretability techniques:
  1. **SHAP (SHapley Additive ExPlanations)**:
     - Generated global and local explanations for predictions.
     - Visualized feature importance and impact.
  2. **LIME (Local Interpretable Model-Agnostic Explanations)**:
     - Provided instance-level explanations.
---

## **Results**

1. **Model Performance**:
   - Achieved strong classification metrics for dominant classes (`other` and `question`).
   - Identified areas for improvement in minority classes (`therapist_input` and `reflection`).
2. **Confusion Matrix Insights**:
   - Most misclassifications occurred between similar classes, such as `question` and `reflection`.
3. **Feature Importance**:
   - Textual features (`utterance_text`) contributed the most to predictions, followed by categorical features like `topic`.
---

## **Future Improvements**

- Addressing **class imbalance** using advanced techniques like focal loss or data augmentation.
- Enhancing **feature engineering** to create more distinctive features for overlapping classes.
- Experimenting with **alternative architectures** (e.g., Transformer-based models or ensemble learning) to boost performance on challenging classes.

---

## **Setup Instructions**

1. **Clone this repository:**
   ```bash
   git clone https://github.com/ClayUllrich/assignment_01
   
2. **Install the Required Dependencies**:
   ```bash
   pip install -r requirements.txt

3. Run the jupyter notebook
