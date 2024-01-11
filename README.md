# STOCK-PRICE-PREDICTION-USING-NEWS

# Project Description:

The aim of this project is to develop a predictive model for stock prices by incorporating information from news articles. Stock prices are influenced by a multitude of factors, and news articles can be a valuable source of information that may impact market sentiments and decisions. By leveraging natural language processing (NLP) techniques and machine learning algorithms, this project seeks to predict stock price movements based on relevant news data.

                                        +------------------------+
                                        |                        |
                                        | Historical Stock Price |
                                        |       Data Store       |
                                        |                        |
                                        +-----------+------------+
                                                    |
                       +--------------------------+--------------------+
                       |                                               |
           +-----------v-----------+                   +-------------v------------+
           |                       |                   |                          |
           |   Data Preprocessing   |                   |  News Articles Data Store |
           |                       |                   |                          |
           +-----------+-----------+                   +-------------+------------+
                       |                                               |
                       |                                               |
                       |                                               |
                       |                                               |
                       |                                               |
                       |                                               |
                       |                                               |
           +-----------v------------------+                +-------------v------------+
           |                              |                |                          |
           | Feature Extraction (NLP)    |                | Sentiment Analysis Tool |
           |                              |                |                          |
           +-----------+------------------+                +-------------+------------+
                       |                                               |
                       |                                               |
                       |                                               |
                       |                                               |
                       |                                               |
           +-----------v------------------+                +-------------v------------+
           |                              |                |                          |
           |    Model Development         |                |   News API Integration  |
           |                              |                |                          |
           +-----------+------------------+                +-------------+------------+
                       |                                               |
                       |                                               |
                       |                                               |
                       |                                               |
                       |                                               |
           +-----------v------------------+                +-------------v------------+
           |                              |                |                          |
           |   Model Evaluation & Tuning  |                | Real-time News Fetching  |
           |                              |                |                          |
           +-----------+------------------+                +-------------+------------+
                       |                                               |
                       |                                               |
                       |                                               |
                       |                                               |
                       |                                               |
           +-----------v------------------+                +-------------v------------+
           |                              |                |                          |
           |    Model Deployment          |                |   User Interface (Web)  |
           |                              |                |                          |
           +-----------------------------+                +---------------------------+



# Historical Stock Price Data Store:

Database: MySQL, PostgreSQL, MongoDB

# Data Preprocessing:

Python: pandas, NumPy
Data Cleaning: pandas, NumPy

# Sentiment Analysis Tool:

Python: NLTK, TextBlob, VADER Sentiment

# User Interface (Web):

Web Framework: Flask, Django, Streamlit
Visualization: Matplotlib, Plotly, D3.js (for dynamic charts)

# Python: The primary programming language used for developing the project.
Streamlit: A web framework used for creating interactive and user-friendly data applications.
Setup and Installation:

# Download the Code:

Download the code folder for the Stock Price Prediction using News and Sentimental Analysis project.
Navigate to the project folder containing the main files, datasets, and any additional resources.
Change to the Working Directory:

Open a terminal or command prompt.
Change the directory to the project folder:
bash
Copy code
cd ~/Downloads/stock_price_prediction_using_News
Install Dependencies:

It is recommended to install dependencies in a virtual environment. Create and activate a virtual environment.
Install dependencies using requirements.txt or other provided instructions.
Run the App:

Execute the following command in the terminal to run the Streamlit app:
bash
Copy code
streamlit run "cd ~/Downloads/stock_price_prediction_using_News"


# References:

https://www.kaggle.com/datasets/aaron7sun/stocknews?resource=download&select=upload_DJIA_table.csv
