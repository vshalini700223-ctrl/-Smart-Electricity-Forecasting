# Smart Electricity Forecasting Using LSTM

## 1. Project Overview

**Smart Electricity Forecasting Using LSTM** is a Deep Learning project
that predicts electricity consumption using historical household
electricity data.

The project uses an **LSTM (Long Short-Term Memory)** model because
electricity consumption is time-series data. The model learns patterns
from previous electricity readings and predicts future power
consumption.

The prediction results are displayed through a web-based dashboard built
using **Flask, HTML, CSS, JavaScript, and Chart.js**.

> **Simple concept:**\
> Past electricity data → LSTM learns the pattern → Future electricity
> consumption is predicted → Flask sends the result → Web dashboard
> displays it.

------------------------------------------------------------------------

## 2. Main Objective

The main objective of this project is to:

-   Analyze historical electricity consumption.
-   Use Deep Learning to forecast electricity power.
-   Compare actual and predicted electricity consumption.
-   Display electricity information through an interactive dashboard.
-   Provide a foundation for future smart-meter and IoT integration.

------------------------------------------------------------------------

## 3. Technologies Used

### Machine Learning / Deep Learning

-   Python
-   TensorFlow / Keras
-   LSTM
-   NumPy
-   Pandas
-   Scikit-learn

### Backend

-   Flask
-   Python REST APIs

### Frontend

-   HTML
-   CSS
-   JavaScript
-   Chart.js

### Data

-   Household electricity consumption dataset

------------------------------------------------------------------------

## 4. How the Project Works

The complete project flow is:

``` text
Household Electricity Dataset
            ↓
     Data Preprocessing
            ↓
      LSTM Deep Learning
            ↓
    Electricity Prediction
            ↓
     Model Evaluation
            ↓
      Flask Backend API
            ↓
    HTML/CSS/JavaScript
            ↓
      Web Dashboard
```

------------------------------------------------------------------------

## 5. Dataset

The project uses household electricity consumption data.

Important columns include:

-   Date
-   Time
-   Global active power
-   Global reactive power
-   Voltage
-   Global intensity
-   Sub-metering 1
-   Sub-metering 2
-   Sub-metering 3

The Flask application loads a manageable portion of the dataset for the
dashboard simulation.

The current application loads:

-   **100,000 rows**
-   **99,992 valid rows after preprocessing**

Missing or invalid values are removed before the data is used by the
application.

------------------------------------------------------------------------

## 6. Why LSTM?

LSTM stands for **Long Short-Term Memory**.

Electricity consumption changes over time, so previous readings can
contain useful information about future consumption.

For example:

``` text
Previous electricity readings
          ↓
     LSTM learns patterns
          ↓
Next electricity consumption
```

The project uses a **24-reading input window**.

This means the forecasting system works with a sequence of previous
readings to generate a prediction.

------------------------------------------------------------------------

## 7. Model Evaluation

The project evaluates the LSTM predictions using:

### MAE

**Mean Absolute Error**

MAE measures the average absolute difference between actual and
predicted power.

Lower MAE generally means smaller prediction errors.

### RMSE

**Root Mean Square Error**

RMSE measures prediction error while giving greater influence to larger
errors.

Lower RMSE generally indicates better prediction performance.

The project stores evaluation results in:

``` text
model_evaluation_results.csv
```

The file contains:

``` text
Actual_Power_kW
Predicted_Power_kW
Absolute_Error_kW
```

The current evaluation file contains **5,000 prediction samples**.

------------------------------------------------------------------------

## 8. Web Dashboard

The dashboard provides a visual interface for the electricity
forecasting system.

It displays:

### Current Power

The current electricity power reading in kW.

### Predicted Next Power

The predicted electricity power from the LSTM evaluation results.

### Voltage

The voltage reading from the electricity dataset.

### Current

The electricity current reading.

### Live Electricity Consumption

A graph showing current and predicted power values.

### Energy Sub-Metering

The dashboard displays:

-   Sub Meter 1
-   Sub Meter 2
-   Sub Meter 3

### AI Model Performance

The dashboard provides:

-   MAE
-   RMSE
-   Evaluation sample count
-   LSTM model information
-   Actual vs Predicted Power graph

------------------------------------------------------------------------

## 9. Flask Backend

Flask connects the Python backend with the web dashboard.

Important API endpoints include:

``` text
/
```

Displays the dashboard.

``` text
/api/live
```

Returns electricity readings and prediction information.

``` text
/api/performance
```

Returns model evaluation information such as MAE, RMSE, test samples,
actual values, and predicted values.

``` text
/api/reset
```

Resets the live simulation.

``` text
/api/health
```

Checks whether the backend is running and returns basic system
information.

------------------------------------------------------------------------

## 10. Live Simulation

The current project uses historical electricity data to create a **live
simulation**.

It does not currently connect directly to a physical smart electricity
meter.

When the user presses:

**START LIVE**

the JavaScript frontend requests data from:

``` text
/api/live
```

The Flask backend provides the next electricity reading and prediction.

The dashboard updates approximately every **5 seconds**.

The flow is:

``` text
START LIVE
    ↓
JavaScript calls Flask API
    ↓
Flask gets next dataset reading
    ↓
Prediction information is returned
    ↓
Dashboard values are updated
    ↓
Chart is updated
```

------------------------------------------------------------------------

## 11. Real-World Usage

The current project is a prototype for a smart electricity forecasting
system.

In a real-world system, an IoT-enabled smart meter could continuously
send electricity information to the application.

The future architecture could be:

``` text
Smart Electricity Meter
          ↓
       IoT/API
          ↓
    Data Processing
          ↓
        LSTM
          ↓
 Future Power Prediction
          ↓
    Web/Mobile Dashboard
          ↓
       User Alerts
```

------------------------------------------------------------------------

## 12. How It Can Help People

For household users, the system can help them understand electricity
consumption patterns.

For example, if electricity consumption is increasing, a future version
could provide a warning such as:

``` text
High electricity consumption expected.
```

This could help users identify unnecessary electricity usage and make
better energy-management decisions.

The system can therefore support:

-   Electricity consumption monitoring
-   Future consumption awareness
-   Energy-management decisions
-   High-consumption alerts in future versions

------------------------------------------------------------------------

## 13. Possible Use by Electricity Companies

A larger version of the system could be used for electricity demand
forecasting.

Possible applications include:

-   Demand planning
-   Peak-load monitoring
-   Energy management
-   Smart-grid systems
-   Resource planning

The current project demonstrates the forecasting concept using household
electricity data.

------------------------------------------------------------------------

## 14. Project Files

Important project files include:

``` text
Smart_Electricity_Forecasting/
│
├── app.py
├── app_backup.py
├── predict.py
├── evaluate_model.py
├── electricity_lstm_model.keras
├── scaler.pkl
├── model_evaluation_results.csv
│
├── data/
│   └── household_power_consumption.txt
│
├── templates/
│   └── index.html
│
└── venv/
```

### File descriptions

**app.py**\
Flask backend and API server.

**predict.py**\
Prediction-related Python code.

**evaluate_model.py**\
Evaluates the trained model and generates evaluation results.

**electricity_lstm_model.keras**\
Trained LSTM model.

**scaler.pkl**\
Saved data-scaling object used by the machine-learning workflow.

**model_evaluation_results.csv**\
Actual and predicted electricity values used for model evaluation and
dashboard performance visualization.

**index.html**\
Main web dashboard.

**household_power_consumption.txt**\
Household electricity dataset.

------------------------------------------------------------------------

## 15. How to Run the Project

### Step 1: Open PowerShell

Go to the project folder:

``` powershell
cd C:\Users\saran\Documents\Smart_Electricity_Forecasting
```

### Step 2: Activate the virtual environment

``` powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Flask if required

``` powershell
pip install flask
```

### Step 4: Start the Flask application

``` powershell
python app.py
```

When the application starts successfully, you should see messages
similar to:

``` text
Dataset loaded
Rows: 100000
Valid rows: 99992
Loading LSTM prediction results...
LSTM results loaded
Prediction rows: 5000
Debugger is active!
```

### Step 5: Open the dashboard

Open a browser and visit:

``` text
http://127.0.0.1:5000
```

------------------------------------------------------------------------

## 16. How to Demonstrate the Project

During the demonstration:

1.  Start the Flask application.
2.  Open the dashboard in the browser.
3.  Show the current power card.
4.  Show the predicted power card.
5.  Click **START LIVE**.
6.  Show the graph updating.
7.  Explain voltage and current.
8.  Show the three sub-meter values.
9.  Scroll to **AI Model Performance**.
10. Explain MAE and RMSE.
11. Show the **Actual vs Predicted** graph.
12. Explain that the current version is a historical-data live
    simulation.
13. Explain that actual IoT smart-meter integration is a future
    improvement.

------------------------------------------------------------------------

## 17. Important Point for Presentation

Do not say that the current system is connected to a physical smart
meter.

A correct explanation is:

> "The current implementation uses historical smart-meter data to
> simulate a real-time electricity stream. The architecture can be
> extended in the future to receive live readings from an IoT-enabled
> smart meter."

This accurately describes the current project.

------------------------------------------------------------------------

## 18. Advantages

-   Uses Deep Learning for time-series forecasting.
-   Provides electricity consumption predictions.
-   Provides an interactive web dashboard.
-   Shows actual vs predicted values.
-   Provides model evaluation metrics.
-   Can be extended to IoT and smart-meter systems.
-   Helps users understand electricity consumption patterns.

------------------------------------------------------------------------

## 19. Limitations

-   The current dashboard uses historical data for live simulation.
-   It is not currently connected to a physical electricity meter.
-   Prediction quality depends on the trained model and available data.
-   The current implementation is a prototype rather than a production
    smart-grid system.

------------------------------------------------------------------------

## 20. Future Enhancements

Possible future improvements include:

-   Real-time IoT smart-meter integration.
-   Mobile application.
-   High-consumption notifications.
-   Daily and monthly electricity forecasting.
-   Electricity bill estimation.
-   Appliance-level consumption analysis.
-   Cloud deployment.
-   Database integration.
-   Smart-grid integration.
-   More advanced forecasting models.

------------------------------------------------------------------------

## 21. Simple Explanation for Staff

> "Our project is Smart Electricity Forecasting using LSTM. Historical
> electricity consumption data-va use panni, LSTM deep learning model
> previous electricity usage patterns-ah learn pannum. Previous 24
> readings base panni next electricity consumption predict pannum. Flask
> backend moolama prediction website-ku send pannrom. Dashboard-la
> current power, predicted power, voltage, current and graphs display
> aagum. Currently historical data use panni real-time simulation
> pannirukkom. Future-la actual IoT smart meter connect panni real-time
> prediction and high-consumption alerts provide panna mudiyum."

------------------------------------------------------------------------

## 22. One-Line Concept

**Past electricity data → LSTM learns the pattern → Future power is
predicted → Flask sends the result → Dashboard displays it → Users can
make better energy-management decisions.**

------------------------------------------------------------------------

## 23. Conclusion

The Smart Electricity Forecasting project demonstrates how Deep Learning
can be applied to electricity time-series forecasting.

The project combines:

**LSTM + Python + Flask + HTML + CSS + JavaScript + Chart.js**

to create a complete electricity forecasting prototype with model
evaluation and an interactive dashboard.

The main idea is simple:

> **Use past electricity consumption patterns to predict future
> electricity consumption and present the result in a useful
> dashboard.**
