[colab link
](https://colab.research.google.com/drive/1_WpwaIP5Fq0ZJX3LQ1VG76GO4MCmbwva#scrollTo=5WRiZ8q9p-d2)
# 🌳 Employee Attrition Prediction using Decision Tree

## 📌 Project Overview

Employee attrition can create significant challenges for organizations, including recruitment costs, productivity loss, and the need for continuous employee replacement.

In this project, I built a **Decision Tree Classification model** to predict whether an employee is likely to leave an organization based on factors such as overtime, job satisfaction, age, monthly income, and years at the company.

The project focuses not only on building the model but also on **model evaluation, class imbalance, overfitting, and model improvement using Gini impurity and hyperparameters**.

---

## 🎯 Business Problem

> Can we use employee-related information to predict whether an employee is likely to leave the organization?

The target variable is:

* `0` → Employee stays
* `1` → Employee leaves

---

## 📊 Dataset

The project uses the **IBM HR Analytics Employee Attrition & Performance** dataset.

The dataset contains employee information including:

* Age
* Monthly Income
* Job Satisfaction
* Overtime
* Job Role
* Years at Company
* Years in Current Role
* Work-Life Balance
* Distance From Home
* Total Working Years
* And other employee attributes

---

## 🔍 Exploratory Data Analysis

Several factors were explored to understand their relationship with employee attrition.

### Key observations

**1. Overtime**

Employees working overtime showed a higher tendency toward attrition.

**2. Job Satisfaction**

Lower job satisfaction was associated with higher attrition.

**3. Age**

Employees who left tended to be younger than employees who stayed.

**4. Monthly Income**

Employees with lower monthly income showed a higher tendency toward attrition.

**5. Years at Company**

Employees with shorter tenure showed higher attrition.

These observations helped identify potentially important features for the classification model.

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

* Removed irrelevant columns
* Converted the target variable into binary format
* Encoded categorical variables
* Separated features (`X`) and target (`y`)
* Performed train/test split
* Used stratification because of class imbalance

Decision Trees do not require feature scaling, so standardization was not necessary.

---

## 🌳 Model Building

### Baseline Decision Tree

The initial model was built using:

```python
DecisionTreeClassifier(random_state=42)
```

The baseline model achieved:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 78.57% |
| Precision | 34.00% |
| Recall    | 36.17% |
| F1 Score  | 35.05% |

The model achieved **100% training accuracy**, while testing accuracy was only **78.57%**, indicating significant overfitting.

---

## ⚠️ Model Improvement

The dataset contains considerably more employees who stayed than employees who left.

Therefore, accuracy alone was not sufficient to evaluate model performance.

The focus was shifted toward improving the detection of the minority class — employees who actually left.

### Balanced Decision Tree

A balanced Decision Tree was created using:

```python
DecisionTreeClassifier(
    criterion="gini",
    class_weight="balanced",
    random_state=42
)
```

This gave the minority class greater importance during model training.

### Balanced Model Results

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  |     76.87% |
| Precision |     35.62% |
| Recall    | **55.32%** |
| F1 Score  | **43.33%** |

---

## 🔬 Model Comparison

Three Decision Tree configurations were compared:

| Model                  |   Accuracy |  Precision |     Recall |   F1 Score |
| ---------------------- | ---------: | ---------: | ---------: | ---------: |
| Baseline Decision Tree |     78.57% |     34.00% |     36.17% |     35.05% |
| Model 2                | **84.35%** | **52.63%** |     21.28% |     30.30% |
| Balanced Decision Tree |     76.87% |     35.62% | **55.32%** | **43.33%** |

---

## 🏆 Final Model Selection

The **Balanced Decision Tree** was selected as the preferred model.

Although Model 2 achieved the highest overall accuracy of **84.35%**, its recall for the attrition class was only **21.28%**.

This means that the model was missing a large proportion of employees who actually left.

The Balanced Decision Tree achieved:

* **55.32% recall**
* **43.33% F1 score**

Compared with the baseline:

* Recall improved from **36.17% → 55.32%**
* F1 improved from **35.05% → 43.33%**

For this business problem, identifying employees who are actually at risk of leaving is more important than maximizing overall accuracy.

Therefore, **recall and F1 score were prioritized over accuracy alone**.

---

## 💡 Key Learnings

Through this project, I learned that:

* Accuracy alone can be misleading when classes are imbalanced.
* Decision Trees can easily overfit training data.
* Gini impurity helps the tree determine useful splits.
* `max_depth` and `min_samples_leaf` can control tree complexity.
* `class_weight="balanced"` can help improve minority-class detection.
* Precision and recall represent different aspects of classification performance.
* The "best" model depends on the business objective, not necessarily the highest accuracy.

### Most important takeaway

> **The model with the highest accuracy is not always the best model.**

For employee attrition, missing an employee who is actually going to leave can be more costly than incorrectly flagging an employee who stays.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab
* Decision Tree Classification

---

## 🚀 Future Improvements

Possible next steps include:

* Hyperparameter tuning using `GridSearchCV`
* Feature importance analysis
* ROC-AUC analysis
* Threshold optimization
* Comparing Decision Tree with Random Forest
* Exploring other ensemble techniques

---

## 📌 Project Takeaway

This project helped me move beyond simply training a machine-learning model toward understanding **how to diagnose model performance, identify weaknesses, and select a model based on the actual business objective.**
