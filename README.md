# Emotion Analysis

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
   ```
   ```bash
   # On Windows
   venv\Scripts\activate
   ```
   ```bash
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

## Additional Notes

- To deactivate the virtual environment, run:
  ```bash
  deactivate
  ```

## My Specs:
Nvidia GeForce MX150 2GB
Nvidia Graphics Driver 572.83
Cuda toolkit 11.7
Python 3.12
Currently not able to find compatible torch version for cu117; and links with same are not working maybe due to python version.
Will try torch-cu118