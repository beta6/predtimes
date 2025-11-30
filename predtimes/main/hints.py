# predtimes/main/hints.py

HINTS = {
    'home': {
        'title': 'Welcome to PredTimes!',
        'content': [
            'This is your central dashboard. Here you can see all your time series prediction projects.',
            'Click on a project to view its details, train models, and generate predictions.',
            'To get started, create a new project by clicking the "Create New Project" button.',
        ],
    },
    'project_wizard_step1': {
        'title': 'Step 1: Connect to Your Database',
        'content': [
            'PredTimes needs to connect to the database where your time series data is stored.',
            'Select your database type and provide the connection details.',
            'Your credentials are stored securely and are only used to access the data for your project.',
        ],
    },
    'project_wizard_step2': {
        'title': 'Step 2: Select Your Data Tables',
        'content': [
            'PredTimes has successfully connected to your database.',
            'Now, choose the table or tables that contain the data you want to use for your predictions.',
            'You can select multiple tables if your data is spread out (e.g., sales data in one table and product metadata in another).',
        ],
    },
    'project_wizard_step3': {
        'title': 'Step 3: Name Your Project',
        'content': [
            'Give your project a descriptive name that will help you identify it later.',
            'This name will be displayed on your home dashboard.',
        ],
    },
    'project_detail': {
        'title': 'Project Configuration',
        'content': [
            'This is the main dashboard for your project.',
            '1. **Configure Columns:** Select the columns from your table(s) that correspond to the date/time of the event, the numeric value you want to predict, and any columns used to group the data.',
            '2. **Train Model:** Once your columns are configured, you can start training your first prediction model.',
            '3. **Generate Predictions:** After a model is trained, you can generate future predictions.',
        ],
    },
    'training_in_progress': {
        'title': 'Model Training is in Progress',
        'content': [
            'The AI is learning from your historical data. This process can take some time, depending on the size of your dataset.',
            'You can leave this page and come back later. The training will continue in the background.',
            'Once completed, you will be able to generate predictions.',
        ],
    },
    'prediction_in_progress': {
        'title': 'Generating Predictions',
        'content': [
            'The trained model is now generating future predictions.',
            'This should be a relatively quick process.',
            'The results will appear in the chart and table below once the task is complete.',
        ],
    },
}