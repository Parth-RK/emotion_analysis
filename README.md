# emotion_analysis

## Prerequisites

Ensure you have the following installed on your system before proceeding:

- Python (>= 3.7)
- pip (Python package manager)
- A compatible IDE or text editor (optional but recommended, e.g., VS Code or PyCharm)

## Setup Instructions

1. **Clone the Repository**
   
   ```bash
   git clone https://github.com/Parth-RK/emotion_analysis.git
   cd emotion_analysis
   ```

2. **Create a Virtual Environment** (Optional but recommended)

   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**

   ```bash
   pip list
   ```

## Running the Project

1. **Set Up Any Configuration**
   
   If the project requires specific configurations (e.g., API keys, environment variables, or input data paths), edit the provided configuration file (if applicable) or set environment variables as needed. Refer to the project documentation for details.

2. **Generate the model file**

   Execut to start the model training:
   ```bash
   python emo.py
   ```
   This will generate the model file in the models/  directory.

   Execute to run the model on the input data:
   ```bash
   python app.py
   ```
   This will run the saved model on the input data('texts' variable in the file) and print the results.

3. **View Outputs**

   Check the console or specified output directory for results such as logs, plots, or saved models.

## Additional Notes

- To deactivate the virtual environment, run:
  ```bash
  deactivate
  ```

