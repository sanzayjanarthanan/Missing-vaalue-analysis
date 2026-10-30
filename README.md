# Task 21 – Missing Value Analysis

**Veda Technology · AI & ML Track · Level 1 · Day 21**

## Description
Identify missing values in a dataset and calculate the number (and percentage) of missing values in each column, before applying any machine learning.

## Objective
Understand why identifying missing data is an important step before building machine learning models.

## Tools Used
- Python
- Pandas
- Jupyter Notebook
- Seaborn / Matplotlib (for visualization)

## Dataset
Titanic Dataset (891 rows × 12 columns)
Source: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

## Approach
1. Loaded the dataset using `pd.read_csv()`.
2. Checked shape and structure with `df.info()`.
3. Detected missing values using `df.isnull()`.
4. Counted missing values per column using `.sum()`.
5. Calculated the percentage of missing values per column.
6. Built a missing-value summary table, sorted by percentage.
7. Visualized missing data with a heatmap.
8. Identified the columns that need treatment.

## Key Findings

| Column   | Missing Count | Missing % | Suggested Treatment                      |
|----------|---------------|-----------|------------------------------------------|
| Cabin    | 687           | ~77.10%   | Drop the column (too many missing values) |
| Age      | 177           | ~19.87%   | Impute with median/mean                  |
| Embarked | 2             | ~0.22%    | Fill with mode (most frequent value)     |

## Deliverables
- Missing-value summary
- Column-wise missing-value counts
- Identification of columns requiring treatment

## Why It Matters
- Most ML algorithms cannot handle `NaN` values and will throw errors.
- Unhandled missing data can lead to biased or inaccurate predictions.
- Knowing *how much* and *where* data is missing helps choose the right strategy (drop, impute, or flag).

## Interview Questions – Quick Answers
1. **What is a missing value?** A cell in a dataset that has no value stored (NaN/null).
2. **Why can missing values be a problem in ML?** Many algorithms fail on NaN, and missing data can bias the model and reduce accuracy.
3. **How can you identify missing values in Pandas?** `df.isnull().sum()` or `df.isna().sum()`.

## How to Run
```bash
pip install pandas seaborn matplotlib jupyter
jupyter notebook Missing_Value_Analysis.ipynb
```

## Author
**J Sanjay** – AI & Data Science Student
GitHub: [sanzayjanarthanan](https://github.com/sanzayjanarthanan)
