This project implements Linear Regression completely from scratch in Python, without relying on machine learning libraries like scikit-learn. 
The goal is to understand the mathematics and mechanics behind one of the most fundamental ML algorithms, while practicing clean and reproducible code.

🚀 Features

Implements Linear Regression using:

Gradient Descent optimization

Mean Squared Error (MSE) loss function

Works on any tabular dataset (CSV format)

Includes data preprocessing (normalization & train-test split)

Compares results with scikit-learn’s implementation

📂 Project Structure

├── data/  

├── linear_regression.py 
 
├── notebook.ipynb  

├── poetry.toml      

└── README.md          

⚡ Usage

Clone the repo:

git clone https://github.com/yourusername/linear-regression-from-scratch.git
cd linear-regression-from-scratch


Install dependencies:

pip install -r requirements.txt


Run example notebook:

jupyter notebook notebook.ipynb


Or train the model on your own dataset:

python linear_regression.py --data data/your_dataset.csv

✅ Results

The implementation achieves comparable performance with scikit-learn’s LinearRegression on test datasets.

Visualization of regression line and loss convergence included in notebook.

🎯 Learning Goals

Strengthen understanding of linear regression math

Learn how gradient descent optimization works in practice

Build intuition for how ML libraries implement algorithms internally

🔮 Future Improvements

Add support for polynomial regression

Implement regularization (Ridge & Lasso)

Extend to multivariate datasets
