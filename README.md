# Sentiment Analysis Model Comparison Lab

This lab demonstrates systematic model comparison and selection for sentiment analysis using multiple candidate solutions.

## Overview

This educational lab provides hands-on experience with:
- Multiple model configuration testing
- Systematic performance comparison
- Feature engineering evaluation
- Model selection based on empirical results

## Features

- **5 Candidate Models**: Different TF-IDF and Logistic Regression configurations
- **Systematic Evaluation**: Comprehensive accuracy testing and ranking
- **Feature Analysis**: Feature count and preprocessing impact assessment
- **Performance Insights**: Detailed analysis of model strengths and weaknesses

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy

## Installation

Install the required packages using pip:

```bash
pip install scikit-learn pandas numpy
```

## Usage

Run the model comparison lab:

```bash
python sentiment_model_comparison.py
```

## Lab Structure

### 1. Dataset Generation
- Creates 360 sentiment samples with balanced labels
- 18 unique text samples repeated 20 times each
- 50% positive, 50% negative distribution

### 2. Candidate Models
The lab tests 5 different model configurations:

#### 1. Baseline TF-IDF + LR
- Standard TF-IDF vectorization (max 50 features)
- Logistic Regression with default parameters

#### 2. No Stopwords
- TF-IDF with English stopwords removed
- Tests impact of stopword removal

#### 3. N-grams
- TF-IDF with unigrams and bigrams (1,2)
- Evaluates n-gram feature benefits

#### 4. Drop Short Texts
- Preprocessing: removes texts with ≤2 words
- Tests data filtering impact

#### 5. Regularized LR
- TF-IDF with L2 regularization (C=0.1)
- Evaluates regularization effects

### 3. Evaluation Process
- 80/20 train-test split with stratification
- Accuracy-based performance measurement
- Error handling for failed configurations
- Comprehensive results ranking

### 4. Results Analysis
- Performance ranking of all models
- Feature count analysis
- Preprocessing impact assessment
- Actionable recommendations

## Sample Output

### Console Output Example:
```
SENTIMENT ANALYSIS MODEL COMPARISON LAB
============================================================
Generating sentiment dataset...
Dataset created: 360 samples
Positive samples: 180 (50.0%)
Negative samples: 180 (50.0%)

Splitting dataset...
Train set: 288 samples
Test set: 72 samples

Creating candidate models...
1. Baseline TF-IDF + LR
2. No Stopwords
3. N-grams
4. Drop Short Texts
5. Regularized LR

============================================================
EVALUATING CANDIDATE MODELS
============================================================

Testing 1/5: Baseline TF-IDF + LR
  Accuracy: 1.000
  Features: 35

Testing 2/5: No Stopwords
  Accuracy: 1.000
  Features: 34

Testing 3/5: N-grams
  Accuracy: 1.000
  Features: 50

Testing 4/5: Drop Short Texts
Applying preprocessing: 288 -> 0 samples
  ERROR: empty vocabulary; perhaps the documents only contain stop words

Testing 5/5: Regularized LR
  Accuracy: 1.000
  Features: 35

============================================================
FINAL RESULTS
============================================================
                name  accuracy  features  train_samples  test_samples
Baseline TF-IDF + LR     1.000        35            288            72
        No Stopwords     1.000        34            288            72
             N-grams     1.000        50            288            72
      Regularized LR     1.000        35            288            72
    Drop Short Texts     0.000         0              0             0

WINNER: Baseline TF-IDF + LR
   Accuracy: 1.000
   Insight: Graph-based approach led to optimal model selection!

Model Ranking:
  1st Baseline TF-IDF + LR: 1.000
  2nd No Stopwords: 1.000
  3rd N-grams: 1.000
  4th. Regularized LR: 1.000
  5th. Drop Short Texts: 0.000
```

## Key Learning Outcomes

### Technical Skills:
- **Model Configuration**: Setting up multiple ML pipelines
- **Feature Engineering**: Testing different vectorization strategies
- **Performance Evaluation**: Systematic accuracy measurement
- **Error Handling**: Managing failed model configurations

### Analytical Skills:
- **Comparative Analysis**: Ranking models by performance
- **Feature Impact**: Understanding feature count effects
- **Preprocessing Assessment**: Evaluating data cleaning strategies
- **Decision Making**: Selecting optimal configurations

## Code Architecture

The lab is structured with modular functions:

- `generate_sentiment_dataset()`: Creates balanced sentiment data
- `create_candidate_models()`: Defines 5 model configurations
- `preprocess_data()`: Applies data preprocessing steps
- `evaluate_candidates()`: Tests all models systematically
- `display_results()`: Shows comprehensive results and ranking
- `analyze_model_insights()`: Provides performance analysis
- `main()`: Orchestrates the entire lab workflow

## Troubleshooting

### Common Issues:

1. **Empty Vocabulary Error**: 
   - Occurs when preprocessing removes all text samples
   - Solution: Adjust preprocessing thresholds or use larger dataset

2. **Perfect Accuracy Warning**:
   - May indicate overfitting or too-simple dataset
   - Solution: Use more complex or realistic datasets

3. **Memory Issues**:
   - Large feature counts may cause memory problems
   - Solution: Reduce max_features parameter

## Extensions

Potential lab extensions:
- **Real Datasets**: Replace toy data with actual sentiment datasets
- **Additional Models**: Test SVM, Random Forest, Neural Networks
- **Cross-Validation**: Implement k-fold cross-validation
- **Hyperparameter Tuning**: Add grid search for optimal parameters
- **Metrics Expansion**: Include precision, recall, F1-score

## Educational Value

This lab teaches:
1. **Systematic Approach**: Methodical model comparison
2. **Feature Engineering**: Understanding vectorization impact
3. **Performance Analysis**: Interpreting accuracy results
4. **Practical Skills**: Real-world ML pipeline development

## File Structure

```
├── sentiment_model_comparison.py  # Main lab script
├── README_model_comparison.md     # This documentation
└── sentiment_analysis_lab.py      # Previous lab (state graph)
```

## Dependencies

All packages are standard ML libraries:
- **scikit-learn**: Machine learning algorithms and utilities
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing

## License

This educational lab manual is provided for academic use.

## Author

Generated for educational purposes in sentiment analysis and machine learning model comparison.
