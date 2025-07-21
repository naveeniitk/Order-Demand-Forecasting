# Order Demand Forecasting (INTERNSHIP - JIO Platforms Limited)

When you're done, you can delete the content in this README and update the file with details for others getting started with your repository.

*We recommend that you open this README in another tab as you perform the tasks below. You can [watch our video](https://youtu.be/0ocf7u76WSo) for a full demo of all the steps in this tutorial. Open the video in a new tab to avoid leaving Bitbucket.*


This repository contains a Flask-based web application that predicts the number of orders based on user input: a category and a date. It uses a pre-trained machine learning model (`modelwithacc.pkl`) to perform predictions.

---

## Features

- Simple web interface using Flask
- Takes **category** and **date** as input
- Predicts and displays the number of orders
- Uses a serialised ML model (`.pkl` file)

---

## Project Structure
```cpp
.
├── modelwithacc.pkl # Pre-trained ML model
├── app.py # Main Flask app
├── templates/
│ └── index.html # HTML form for user input
└── README.md # This documentation


---

## Model Details

The model is trained and saved as `modelwithacc.pkl`. It is loaded using Python’s `pickle` module and used to generate predictions based on category and date inputs.

---
