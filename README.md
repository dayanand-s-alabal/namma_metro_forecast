# Namma Metro Ridership Forecasting 🚇

An AI-powered ridership forecasting system for **Bengaluru's Namma Metro**, designed to predict passenger entries and exits at individual metro stations and across the entire network for the next seven days.

The project uses historical ridership data, statistical forecasting, machine learning, and a tracker-driven forecasting approach to generate future ridership estimates. Results are presented through an interactive dashboard with station-level analysis, hourly patterns, model performance, and data-quality information.

> **Project status:** Academic prototype. This is not an official BMRCL product.

---

## 1. Project Overview

Namma Metro ridership changes depending on weekdays, weekends, holidays, festivals, special events, and passenger travel patterns.

Understanding these patterns can help with:
- Estimating future passenger traffic at individual stations.
- Identifying expected peak and low-ridership hours.
- Understanding daily passenger demand across the metro network.
- Supporting operational planning and resource allocation.
- Analysing historical ridership trends and forecasting performance.

Our project addresses this problem by building a Python-based forecasting pipeline that predicts the next seven days of metro ridership and displays the results through an interactive dashboard.

### Key Features

- **7-day ridership forecasting:** Automatically forecasts the seven days following the date on which the program is executed.
- **Station-wise forecasting:** Predicts hourly entries and exits for individual stations.
- **Network-level forecasting:** Estimates total daily ridership across the metro network.
- **Missing-data imputation:** Handles missing historical observations using a tracker-driven top-down approach.
- **Machine learning and statistical models:** Evaluates multiple forecasting algorithms and selects suitable models.
- **Interactive dashboard:** Provides visualisations, forecast tables, historical trends, and model evaluation.
- **Prediction intervals:** Provides 80% prediction intervals to communicate forecast uncertainty.
- **Automated validation:** Includes data-quality checks, leakage tests, and forecasting acceptance criteria.

---

## 2. How the Project Works

The project follows a data-processing and forecasting pipeline.

```text
        Historical Ridership Datasets
                    |
                    v
           Data Ingestion
                    |
                    v
        Data Cleaning & Validation
                    |
                    v
       Missing Data Identification
                    |
                    v
       Historical Data Preparation
                    |
                    v
        Model Training & Evaluation
                    |
                    v
         Network Ridership Forecast
                    |
                    v
       Station-wise & Hourly Forecast
                    |
                    v
          Forecast Validation
                    |
                    v
          SQLite + CSV Outputs
                    |
                    v
          Interactive Dashboard
```

### Step 1: Data Collection

The project uses two primary ridership data sources:

1. Station-wise hourly entries and exits obtained through RTI data.
2. Daily network-level ridership figures from the Namma Metro ridership tracker.

Additional reference datasets provide station information, station-name mappings, holidays, and significant events.

### Step 2: Data Cleaning and Preparation

The raw datasets are processed to:
- Standardise station names using station-name mappings.
- Identify duplicate records.
- Detect negative passenger counts.
- Validate dates and hourly observations.
- Identify missing observations.
- Maintain a record of data provenance and cleaning operations.

Missing observations are not automatically replaced with zero.

The project distinguishes between:

| Status | Meaning |
|---|---|
| Observed | Actual ridership observation |
| Missing | No data was recorded |
| Not applicable | Station was not operational or the hour had no service |
| Imputed | Value estimated using the missing-data method |

### Step 3: Missing Data Imputation

The project uses daily network ridership figures to estimate missing station-level historical observations.

The network total is distributed across stations and operating hours using historical station shares and hourly passenger profiles.

This is particularly important because station-level data ends on 30 September 2025, whereas network-level tracker data continues until 15 November 2026.

### Step 4: Model Training and Forecasting

Multiple forecasting algorithms are evaluated using historical ridership data.

The selected production approach uses a tracker-driven top-down forecasting method to generate station-level hourly predictions.

### Step 5: Dashboard and Results

Forecasts, historical observations, model performance, and data-quality information are stored and presented through an interactive dashboard.

---

## 3. Datasets

The project uses the following collected datasets.

### 3.1 Hourly Station Data — Core Dataset

**Directory:** `01_hourly_station/`

Contains station-wise hourly passenger entries and exits.

| File | Description |
|---|---|
| `cleaned/station-hourly.csv` | Hourly passenger entries |
| `cleaned/station-hourly.parquet` | Parquet version of entries |
| `cleaned/station-hourly-exits.csv` | Hourly passenger exits |
| `cleaned/station-hourly-exits.parquet` | Parquet version of exits |
| `original_xlsx/` | Original RTI Excel files |

**Coverage:**
- 1 August – 18 August 2025
- 1 September – 30 September 2025
- 48 observed days
- 83 stations

The CSV files are semicolon-delimited.

**Source:** RTI request to BMRCL by Vivek Mathew.

The data is also mirrored on OpenCity.

Licence information is available in `LICENSE_vonter_repo.txt`. The source repository uses ODbL, while OpenCity lists CC BY-NC-SA 4.0. Check the applicable licence before non-academic use.

### 3.2 Station Pair Origin-Destination Data

**Directory:** `02_station_pair_od/`

Contains hourly passenger movement information between station pairs.

| File | Description |
|---|---|
| `stationpair-hourly.csv` | Hourly origin-destination data |
| `stationpair-hourly-enhanced.csv` | Enhanced data including fare and revenue |

**Coverage:** 1–18 August 2025.

This dataset is optional and can support future origin-destination analysis.

### 3.3 Daily Network Ridership Data

**Directory:** `03_daily_network/`

Contains daily network-level ridership figures.

| File | Description |
|---|---|
| `NammaMetro_Ridership_Dataset.csv` | Raw daily BMRCL ridership figures by ticket mode |
| `NammaMetro_Ridership_Processed.csv` | Cleaned and aggregated daily ridership data |

**Coverage:** 26 October 2024 – 15 November 2026 for the raw dataset.

The raw dataset contains 495 observed days and 256 missing days.

This dataset is particularly important because it provides more recent network-level ridership information than the station-level hourly dataset.

### 3.4 Reference, Calendar and Station Data

**Directory:** `04_reference_calendar_stations/`

| File | Description |
|---|---|
| `NammaMetro_Stations_Dataset.csv` | Station names in English and Kannada, metro lines, and interchange information |
| `station-codes.csv` | Station-code mappings |
| `station-names.csv` | Original station-name to canonical-name mappings |
| `significant_dates.csv` | Holidays, festivals, and significant events |

These datasets help standardise station information and identify calendar-related ridership patterns.

### 3.5 Data Not Included

The following information is not currently included:
- Weather data from IMD or Open-Meteo.
- Complete station opening-date information.
- Automatically updated live ridership data.

---

## 4. Forecasting Models

The project evaluates multiple statistical and machine learning approaches.

### Station-Level Models

| Model | Category |
|---|---|
| Seasonal Naive | Baseline |
| 3-week Same-Hour Moving Average | Baseline |
| Simple Exponential Smoothing | Statistical |
| Holt's Method | Statistical |
| Holt-Winters | Statistical |
| SARIMA | Statistical |
| XGBoost | Machine Learning |
| Ensemble | Combined Model |
| `trk_td` | Tracker-driven Hybrid Model |

### Network-Level Models

The network daily forecasting pipeline evaluates:
- Naive forecasting.
- Seasonal Naive.
- Holt-Winters.
- SARIMA.
- Holiday-adjusted Holt-Winters (`hw_h`).
- Holiday-adjusted SARIMA (`sarima_h`).

The selected network forecasting model is `hw_h`, which achieved a WAPE of approximately 6.20% during the documented backtesting.

### Production Forecasting Approach

The selected station-level production approach is `trk_td` (tracker-driven top-down).

It uses the network ridership forecast and distributes it across stations and hours based on historical passenger shares and hourly profiles.

The general formula is:

`Station Hourly Ridership = Network Total × Station-to-Tracker Ratio × Station Share × Hourly Profile`

The historical station shares, ratio, and hourly profiles are learned from the last 35 observed station-data days available up to 30 September 2025.

Entries and exits are forecast separately.

---

## 5. Forecasting Output

For every execution, the project generates forecasts for the seven days following the execution date.

### Forecast Deliverables

1. Station-wise hourly entry forecasts.
2. Station-wise hourly exit forecasts.
3. Station-wise daily ridership totals.
4. Network-level daily ridership forecasts.
5. Prediction intervals representing forecast uncertainty.
6. Historical missing-data imputation results.

Each station receives 168 hourly forecast values per measure:

`7 days × 24 hours = 168 hourly predictions`

The system follows these rules:
- Entries and exits are never added together.
- Forecast values cannot be negative.
- Hours 01:00–03:00 are forecast as zero because metro services are not operating.
- Missing observations are not interpreted as zero.
- Imputed values are identified separately from actual observations.

---

## 6. Interactive Dashboard

The project includes a seven-page interactive dashboard.

| Dashboard Page | Features |
|---|---|
| Network Overview | Network KPIs, historical ridership, and future forecasts |
| Station Analysis | Station-wise historical and forecast ridership |
| Hourly Profile | Heatmap showing the 168 forecast hours |
| 7-Day Forecast | Daily and hourly forecast tables with prediction intervals |
| Model Performance | Comparison of forecasting models and evaluation metrics |
| Data Quality | Missing-data calendar, imputation results, and validation |
| Report and Export | Generated report and downloadable outputs |

### Dashboard Design

The dashboard includes:
- Interactive charts and visualisations.
- Light and dark themes.
- Animated metro-train background.
- Responsive visual elements.
- Colour-blindness-validated chart palette.
- Clear identification of actual, forecasted, and imputed values.

The dashboard is generated as a self-contained HTML file and can be opened locally in a web browser.

---

## 7. Installation and Setup

### Prerequisites

- Python installed on your system.
- Git.
- Access to the project repository.
- Required Python dependencies.

### Step 1: Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <PROJECT_DIRECTORY>
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies specified by the project's Python package configuration.

### Step 4: Run the Project

Make sure you are in the project root directory and that the datasets are available in the expected project data directories.

---

## 8. How to Run the Project

The project provides a command-line interface for running the forecasting pipeline and managing results.

### 8.1 Run the Complete Forecasting Pipeline

```bash
python -m namma_forecast.cli run
```

This command:
- Loads and validates the datasets.
- Performs data preparation and missing-data imputation.
- Executes the forecasting pipeline.
- Generates seven-day ridership forecasts.
- Stores results in SQLite and CSV files.
- Generates the final project report.

The complete pipeline takes approximately 100 seconds in the documented environment.

### 8.2 Run in Fast Mode

```bash
python -m namma_forecast.cli run --fast
```

The `--fast` option skips SARIMA to reduce execution time.

### 8.3 Launch the Dashboard

```bash
python -m namma_forecast.cli dashboard
```

This starts the dashboard server at:

`http://localhost:8765`

The dashboard opens in your default browser.

### 8.4 Refresh the Datasets

```bash
python -m namma_forecast.cli refresh
```

Use this command when newer hourly station or network tracker data becomes available.

It merges newer data files and can optionally rerun the forecasting pipeline.

**Note:** The application does not automatically download live ridership data. New datasets must be provided before refreshing.

### 8.5 Update Historical Forecast Accuracy

```bash
python -m namma_forecast.cli actuals
```

This command fills actual ridership values into previous forecasts so that forecast accuracy can be tracked over time.

### 8.6 Run Automated Tests

```bash
python -m pytest -q tests
```

This executes the project's automated test suite.

The documented implementation includes 25 unit tests covering data masking, missing-value handling, entries and exits, forecasting leakage, and other acceptance requirements.

---

## 9. Generated Files and Outputs

The forecasting pipeline generates several outputs.

### CSV Files

| File | Description |
|---|---|
| `forecast_station_hourly_7day.csv` | Station-wise hourly forecasts |
| `forecast_station_daily_7day.csv` | Station-wise daily forecasts |
| `forecast_network_daily_7day.csv` | Network-level daily forecasts |
| `imputed_station_daily.csv` | Imputed station daily observations |
| `imputed_station_hourly.csv.gz` | Imputed station hourly observations |
| `imputation_validation.csv` | Missing-data validation results |
| `tracker_path_actual_filled_forecast.csv` | Historical, filled, and forecast network tracker series |
| `run_summary.json` | Summary of the forecasting execution |

Additional outputs include:
- Model performance metrics.
- Data cleaning and quality logs.
- SQLite database containing forecast and validation information.
- `final_report.md`, generated automatically.
- Self-contained HTML dashboard.

---

## 10. Model Evaluation and Validation

The project uses rolling-origin backtesting with three folds of seven days each.

The following evaluation metrics are calculated:

- MAE – Mean Absolute Error.
- RMSE – Root Mean Squared Error.
- MAPE – Mean Absolute Percentage Error.
- sMAPE – Symmetric Mean Absolute Percentage Error.
- WAPE – Weighted Absolute Percentage Error.

WAPE is used as the primary evaluation metric because MAPE can become unstable when hourly passenger counts are small.

### Documented Results

| Metric | Result |
|---|---|
| Selected network model | `hw_h` |
| Network WAPE | Approximately 6.20% |
| Network imputation error | Approximately 2–2.5% |
| Station-hour imputation WAPE | Approximately 10–13% |
| Prediction interval | 80% |
| Automated unit tests | 25 passed |
| Acceptance criteria | 14 passed |

The documented results correspond to the 21 November 2026 execution and its historical evaluation.

The system also performs a data-leakage test to ensure that forecasting models do not use information from dates after the forecast origin.

---

## 11. Known Limitations

The current implementation has the following limitations:

1. **Historical station data:** Station-level hourly data is available only until 30 September 2025.
2. **Network tracker gaps:** There are 256 missing days in the daily tracker dataset, which are filled using seasonal imputation.
3. **Station opening dates:** Station availability changes over time, and incorrect opening-date information can affect aggregation.
4. **Frozen station profiles:** Station shares, station-to-tracker ratios, and hourly profiles are based on data up to 30 September 2025.
5. **No live data integration:** The project requires manually refreshed datasets to incorporate newer ridership observations.
6. **No weather integration:** Weather information is not currently used in forecasting.
7. **Holiday classification:** Holiday and festival identification uses keyword-based classification.
8. **Station-level accuracy:** Station-hour forecasts are less accurate than network-level forecasts.
9. **Prediction intervals:** Station-level intervals use a fixed widening factor of 1.5, which is a heuristic rather than a measured coverage guarantee.
10. **Different data definitions:** Station-derived and network tracker totals are not interchangeable.

---

## 12. Future Improvements

Potential future enhancements include:

- Integrating live Namma Metro ridership data.
- Incorporating weather forecasts and historical weather information.
- Updating station-level models with more recent hourly observations.
- Improving holiday and festival detection.
- Incorporating station opening dates and network expansion information.
- Exploring additional machine learning and deep learning models.
- Improving station-level prediction accuracy.
- Adding origin-destination passenger movement analysis.
- Introducing automated data collection and scheduled model retraining.

---

## 13. Project Objective

The ultimate objective of this project is to develop a data-driven forecasting system that helps understand and predict Namma Metro passenger demand.

By combining historical ridership data, statistical forecasting, machine learning, missing-data imputation, and interactive visualisation, the project aims to provide useful insights into future passenger movement across Bengaluru's metro network.

---
