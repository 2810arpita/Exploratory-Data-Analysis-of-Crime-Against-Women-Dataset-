# Exploratory-Data-Analysis-of-Crime-Against-Women-Dataset-

📌 Project Overview

This project focuses on performing Exploratory Data Analysis (EDA) on a real-world dataset related to crimes against women in India. The goal is to extract meaningful insights, identify patterns, and understand trends using data science techniques.

The project also includes statistical analysis and a basic machine learning model (Linear Regression) to predict outcomes.

🎯 Objectives
Analyze crime trends over the years
Identify most common crime categories
Perform state-wise (regional) analysis
Evaluate legal process efficiency (reported, pending, convicted cases)
Find relationships between variables
Apply machine learning for prediction

📂 Dataset Information
Dataset Name: Cases under Crime Against Women
Source: Government of India Open Data Platform
Format: CSV

🔹 Features:
Area/State Name
Year
Crime Category (Group & Sub-group)
Cases Reported
Cases Chargesheeted
Cases Convicted
Cases Pending Investigation
Cases Pending Trial

🛠️ Technologies Used
Python
Pandas – Data manipulation
NumPy – Numerical operations
Matplotlib & Seaborn – Data visualization
SciPy – Statistical analysis
Scikit-learn – Machine Learning

⚙️ ETL Process
1. Extract
Loaded dataset using pandas.read_csv()
2. Transform
Removed missing values and duplicates
Selected relevant columns
Grouped data (year-wise, category-wise)
3. Load
Stored cleaned data in DataFrame for analysis

📊 Data Analysis & Visualization
🔹 Visualizations Used:
Line Chart → Year-wise crime trends
Bar Chart → Top crime categories
Histogram → Data distribution
Scatter Plot → Relationship between variables
Heatmap → Correlation analysis

📈 Key Insights
Crime data is highly uneven and skewed
No strong correlation between most variables
Some crime categories are significantly higher than others
Many cases remain pending, indicating legal system gaps
Crime trends show variation over time

🤖 Machine Learning
Model Used: Linear Regression
Evaluation Metrics:
R² Score
Mean Squared Error (MSE)
Result:
High accuracy observed
Possible overfitting detected

📌 Outcomes
Identified crime patterns and trends
Highlighted high-risk categories and regions
Evaluated legal system efficiency
Demonstrated use of EDA + ML on real-world data

🚀 Future Scope
Use advanced models (Random Forest, Decision Tree)
Apply time-series forecasting (ARIMA, LSTM)
Build interactive dashboards (Power BI / Tableau)
Integrate additional datasets (population, literacy, etc.)

📚 References
https://data.gov.in
https://www.kaggle.com
https://pandas.pydata.org
https://scikit-learn.org

👩‍💻 Author

Arpita

B.Tech CSE/IT
Lovely Professional University

⭐ Acknowledgement

I would like to thank my mentor and university for their guidance and support in completing this project.
