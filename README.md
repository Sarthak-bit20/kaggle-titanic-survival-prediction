# kaggle-titanic-survival-prediction
My first Kaggle competition - Titanic survival prediction with 82% accuracy using ensemble methods

# 🚢 Kaggle Titanic Survival Prediction

## 📊 Project Overview
My first Kaggle machine learning competition predicting passenger survival on the Titanic with **82-83% accuracy** using ensemble methods and advanced feature engineering.

**Goal**: Build a binary classification model to predict which passengers survived the Titanic disaster based on demographic and booking information.

## 📈 Dataset Overview
- **Source**: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- **Training Set**: 891 passengers with target variable
- **Test Set**: 418 passengers for prediction
- **Features**: 11 input variables (Passenger class, age, gender, fare, family size, etc.)
- **Target**: Survived (0 = Did not survive, 1 = Survived)

## 🔍 My Complete Approach

### 1. Exploratory Data Analysis (EDA)
- Analyzed missing values: Age (177), Cabin (687), Embarked (2)
- **Key Finding**: 74% of female passengers survived vs only 19% of males
- Discovered correlation between passenger class and survival rates
- Identified age and fare distributions patterns

### 2. Data Preprocessing & Cleaning
- **Missing Values**:
  - Filled Age with median value (26.0 years)
  - Filled Embarked with mode (Port 'S')
  - Dropped Cabin column due to 77% missing values
- **Encoding**: Converted categorical variables to numeric
- **Scaling**: Normalized Age and Fare for better model performance

### 3. Feature Engineering
Created new powerful features from existing data:
- **FamilySize**: SibSp + Parch + 1 (total family members)
- **IsAlone**: Binary feature (1 if traveling alone)
- **Title**: Extracted from passenger names (Mr, Mrs, Miss, Master, Rare)
- These engineered features improved model accuracy by 3-5%

### 4. Model Development & Training
Tested multiple machine learning algorithms:
- **Logistic Regression**: 80% accuracy (baseline)
- **Random Forest**: 81% accuracy
- **XGBoost**: 82% accuracy ⭐ (best single model)
- **Gradient Boosting**: 81.5% accuracy

### 5. Hyperparameter Optimization
Used systematic approach to fine-tune models:
- **GridSearchCV** with 5-fold cross-validation
- XGBoost parameters tuned:
  - max_depth: [3, 5, 7]
  - learning_rate: [0.01, 0.1, 0.2]
  - n_estimators: [100, 200, 300]

### 6. Ensemble Methods
Combined best models for superior predictions:
- Weighted average: 60% XGBoost + 40% Random Forest
- **Final Ensemble Accuracy: 82-83%** ✨

## 📊 Results & Performance

| Model | Accuracy | Cross-Validation |
|-------|----------|-----------------|
| Logistic Regression | 80% | 79.5% |
| Random Forest | 81% | 80.8% |
| XGBoost | 82% | 81.5% |
| **Ensemble (Final)** | **82-83%** | **82.1%** |

**Key Insights**:
- Gender is the strongest predictor (74% feature importance)
- Passenger class significantly affects survival chances
- Young children had higher survival rates
- First class passengers had better survival odds

## 🛠️ Technologies & Libraries Used
- **Language**: Python 3.x
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn
- **Notebook**: Jupyter
- **Version Control**: Git & GitHub

## 📋 Project Structure
kaggle-titanic-survival-prediction/
├── README.md # Project documentation
├── requirements.txt # Project dependencies
└── notebookf7310468b3.ipynb # Complete Jupyter notebook
├── Data loading & exploration
├── Missing value handling
├── Feature engineering
├── Model training & evaluation
├── Hyperparameter tuning
└── Ensemble predictions

## How to Run This Project

### Running the Analysis
1. Open `notebookf7310468b3.ipynb` in Jupyter
2. Run all cells to reproduce results (Shift + Enter on each cell)
3. View predictions and model performance metrics

## 📚 What I Learned

✅ **Data Science Workflow**: Complete ML pipeline from data to deployment  
✅ **Feature Engineering**: Creating meaningful features improves model accuracy  
✅ **Model Selection**: Testing multiple algorithms to find the best one  
✅ **Hyperparameter Tuning**: Systematic optimization increases performance  
✅ **Ensemble Methods**: Combining models often beats individual models  
✅ **Cross-Validation**: Proper validation prevents overfitting  
✅ **Git & GitHub**: Professional version control and documentation  

## 🎯 Future Improvements & Next Steps

- [ ] Achieve 90%+ accuracy with deep learning (Neural Networks)
- [ ] Implement SHAP for model interpretability and explainability
- [ ] Try advanced feature selection techniques (SelectKBest, RFE)
- [ ] Build production-ready API for real-time predictions
- [ ] Create interactive dashboard with Streamlit
- [ ] Compare with Stack Overflow top solutions
- [ ] Participate in more Kaggle competitions

## 🔗 Important Links
- **[Kaggle Competition](https://www.kaggle.com/c/titanic)**: Original challenge
- **[My Kaggle Profile](https://www.kaggle.com/sarthak-bit20)**: View all my projects
- **[Titanic Dataset](https://www.kaggle.com/c/titanic/data)**: Download the data

## 📝 Project Metadata
- **Author**: Sarthak Khedkar
- **Email**: soomeg2005@gmail.com
- **Location**: Nagpur, Maharashtra, India
- **Education**: BSc at Shivaji Science College, Nagpur
- **Goal**: Pursue GSoC 2026 & Erasmus Mundus Scholarship
- **Started**: September 2025
- **Completed**: November 2025
- **GitHub**: [Sarthak-bit20](https://github.com/Sarthak-bit20)

## 🤝 Let's Connect
- Interested in ML/AI collaboration? Feel free to reach out!
- Open to constructive feedback and suggestions for improvement

---

*This is my first Kaggle competition! It was an amazing learning experience applying machine learning concepts to real-world data. Every mistake taught me something valuable about data science, and I'm excited to apply these skills to more challenging problems and contribute to open-source AI projects.*


### Prerequisites
- Python 3.7+
- Git
- Jupyter Notebook
