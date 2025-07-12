# Machine Learning Laboratory - Artificial Intelligence

This repository contains implementations of various machine learning algorithms applied to different prediction problems. The project demonstrates the practical application of supervised learning techniques including classification algorithms for real-world scenarios.

## 📋 Table of Contents

- [Features](#features)
- [Algorithms Implemented](#algorithms-implemented)
- [Datasets](#datasets)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results and Output](#results-and-output)
- [Contributing](#contributing)

## ✨ Features

- **Multiple ML Algorithms**: Implementation of 5 different machine learning algorithms
- **Real-world Datasets**: Practical datasets for business, education, and lifestyle prediction
- **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, and F1-score
- **Data Visualization**: Graphical representation of decision boundaries and model performance
- **Bilingual Support**: Code comments and outputs in Russian/Romanian for local context

## 🤖 Algorithms Implemented

| Algorithm | File | Purpose | Dataset |
|-----------|------|---------|---------|
| **Naive Bayes** | `Lab2.py` | Customer Premium Purchase Prediction | Customer Survey Data |
| **AdaBoost** | `Lab2Meta.py` | Business Success Prediction | Business Success (Lei currency) |
| **Decision Trees** | `Lab2Rules.py` | Student Scholarship Eligibility | Student Academic Data |
| **Support Vector Machine (SVM)** | `Lab2SVM.py` | Student Scholarship Prediction | Student Academic Data |
| **Random Forest** | `Lab2Tree.py` | Life Success Prediction | Life Success Factors |

## 📊 Datasets

### 1. Customer Survey (`Customer_Survey.csv`)
- **Features**: Hours Online, Spending ($)
- **Target**: Premium Purchase (0/1)
- **Use Case**: E-commerce customer behavior analysis

### 2. Business Success (`Business_Success_Lei.csv`)
- **Features**: Age, Education, Experience, Investment (Lei), Team, Risk, Innovation
- **Target**: Business Success (0/1)
- **Use Case**: Entrepreneurship success prediction

### 3. Student Data (`Student_Data.csv`)
- **Features**: Study Hours, Grade
- **Target**: Scholarship Eligibility (0/1)
- **Use Case**: Academic performance evaluation

### 4. Life Success (`Life_Success.csv`)
- **Features**: Age, Education levels, Employment, Income
- **Target**: Life Success (0/1)
- **Use Case**: Socioeconomic success factors analysis

## 🛠 Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Kwameldx666/IA.git
cd IA
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy scikit-learn matplotlib
```

## 💻 Usage

### Running Individual Algorithms

1. **Naive Bayes - Customer Analysis:**
```bash
python3 Lab2.py
```

2. **AdaBoost - Business Success:**
```bash
python3 Lab2Meta.py
```

3. **Decision Trees - Scholarship Prediction:**
```bash
python3 Lab2Rules.py
```

4. **SVM - Student Performance:**
```bash
python3 Lab2SVM.py
```

5. **Random Forest - Life Success:**
```bash
python3 Lab2Tree.py
```

### Expected Output

Each script will provide:
- **Data Analysis**: First 5 rows of processed data
- **Model Training**: Training and test set information
- **Predictions**: Sample predictions for new examples
- **Performance Metrics**: Confusion matrix, accuracy, precision, recall, F1-score
- **Visualization**: Decision boundaries and data distribution plots
- **Interpretations**: Detailed analysis of model performance

### Example Output
```
Первые 5 строк данных:
   ЧасыУчебы  Оценка  Стипендия
0        7.5    97.0          1
1       19.0    77.5          1
...

Предсказание для ЧасыУчебы=10, Оценка=85: Да

Оценка модели:
Точность (Accuracy): 0.85
Точность (Precision): 0.83
Полнота (Recall): 0.88
F1-мера: 0.85
```

## 📁 Project Structure

```
IA/
├── Lab2.py              # Naive Bayes implementation
├── Lab2Meta.py          # AdaBoost meta-learning
├── Lab2Rules.py         # Decision Trees with rules extraction
├── Lab2SVM.py           # Support Vector Machine
├── Lab2Tree.py          # Random Forest implementation
├── Customer_Survey.csv  # Customer behavior dataset
├── Business_Success_Lei.csv # Business success dataset
├── Student_Data.csv     # Academic performance dataset
├── Life_Success.csv     # Life success factors dataset
├── Business_Success_Lei.zip # Archived business data
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 📈 Results and Output

### Performance Metrics Explained

- **Accuracy**: Overall correctness of the model
- **Precision**: Ratio of correctly predicted positive observations
- **Recall**: Ratio of correctly predicted positive observations to all actual positives
- **F1-Score**: Weighted average of Precision and Recall

### Visualizations

Each algorithm generates:
1. **Decision Boundary Plots**: Show how the algorithm separates different classes
2. **Training vs Test Performance**: Compare model performance on different datasets
3. **Feature Importance**: (Where applicable) Show which features matter most
4. **Decision Trees**: Visual representation of decision rules

### Model Interpretations

The scripts provide automatic interpretation of results:
- Performance assessment (high/low accuracy analysis)
- Business implications of the predictions
- Recommendations for model improvement

## 🤝 Contributing

This is an educational project demonstrating machine learning concepts. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions about this project, please open an issue in the repository.

---

*This project is part of a machine learning laboratory course focusing on practical applications of artificial intelligence algorithms.*