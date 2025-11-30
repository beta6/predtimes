# predtimes
PredTimes is a web-based AI assistant for time series forecasting. It allows users to connect to their database, train machine learning models, and visualize future predictions through a simple interface. Interestingly, this entire project was developed by an AI.
=======
# PredTimes: Time Series Forecasting

## Overview

PredTimes is a web-based application designed for easy time series forecasting. It allows users to connect to their own databases, select data, train machine learning models, and visualize future predictions. The entire process is managed through a user-friendly web interface, with heavy computations like model training handled asynchronously.

## Features

-   **Intuitive Project Wizard:** A step-by-step guide to set up new forecasting projects.
-   **Flexible Data Source:** Connect to your existing PostgreSQL, MySQL, or MS SQL Server databases.
-   **Customizable Data Selection:** Choose the tables and columns for your time series analysis, including datetime, numeric, and grouping columns.
-   **Asynchronous Model Training:** Train LSTM models in the background without blocking the user interface, powered by Celery.
-   **Prediction Generation:** Generate future predictions based on the trained models.
-   **Interactive Visualizations:** View your time series data and predictions on an interactive chart.
-   **Data Export:** Download your prediction data as a CSV file.

## How It Works

1.  **Create a Project:** The user creates a new project using a wizard.
2.  **Connect to Database:** The user provides connection details for their external database.
3.  **Select Data:** The user selects a table and specifies which columns represent the timestamp, the values to be predicted (numeric), and any columns for grouping the data.
4.  **Train Model:** The user initiates the model training process. The application launches a background task to fetch the data and train a model for each data group.
5.  **Generate & View Predictions:** Once the model is trained, the user can generate and view future predictions on a chart.

## Tech Stack

-   **Backend:** Python, Django
-   **Asynchronous Tasks:** Celery, Redis
-   **Machine Learning:** TensorFlow (with Keras), Scikit-learn, Pandas, NumPy
-   **Database Connectivity:** SQLAlchemy for external DB connections.
-   **Frontend:** HTML, JavaScript, Chart.js, Bootstrap
-   **Containerization:** Docker, Docker Compose

## Getting Started

### Prerequisites

-   Docker
-   Docker Compose

### Installation

1.  Clone this repository to your local machine:
    ```bash
    git clone <repository-url>
    cd predtimes
    ```
2.  The application is configured to run in a Docker container. The required Python packages are listed in `predtimes/requirements.txt` and will be installed automatically by Docker.

### Running the Application

1.  Build and start the services using Docker Compose:
    ```bash
    docker-compose up --build -d
    ```
    This will start the web server, the Celery worker, and a Redis instance.

2.  Access the web application by navigating to `http://localhost:8000` in your web browser.

3.  The default login credentials are:
    -   **Username:** `admin`
    -   **Password:** `password`

## Usage

1.  **Log In:** Use the default credentials to log in.
2.  **Create a Project:** On the home page, click on "Create New Project" to launch the wizard.
3.  **Step 1: Project Name:** Give your project a name and description.
4.  **Step 2: Database Connection:** Provide the connection details for your external database. Test the connection before proceeding.
5.  **Step 3: Select Tables:** Choose the tables you want to use for forecasting.
6.  **Configure Project:** After creating the project, you'll be redirected to the project detail page. Here you can:
    -   Select the columns for **Datetime**, **Numeric** values (the values you want to predict), and **Multigroup** (for creating separate models for each group).
    -   Start the model training.
7.  **Train Model:** Configure training parameters and click "Train" to start the training process. You can monitor the status of the training task.
8.  **Generate Predictions:** Once training is complete, you can generate predictions. The predictions will be displayed on the chart.

## Project Structure

-   `predtimes/`: The main Django project directory.
    -   `main/`: The core Django app containing models, views, and tasks.
    -   `templates/`: HTML templates for the web interface.
    -   `tasks.py`: Celery tasks for model training and prediction.
    -   `views.py`: Django views for handling user requests.
    -   `models.py`: Django models for projects, database connections, etc.
-   `docker-compose.yml`: Defines the services, networks, and volumes for the Docker application.
-   `Dockerfile`: The recipe for building the application's Docker image.
-   `trained/`: Directory where trained models and scalers are saved.
-   `temp_data/`: Directory where temporary data (like CSV exports) is stored.

## Author

-   **beta6** - [https://www.tuxrincon.com/](https://www.tuxrincon.com/)

## License

This project is licensed under the GPLv3 License - see the [LICENSE](https://www.gnu.org/licenses/gpl-3.0.html) file for details.
