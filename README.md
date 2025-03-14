 HEAD
# Green-Scan

# Green Scan - Plant Disease Detection and Treatment System

Green Scan is an AI-powered web application for detecting plant diseases and suggesting treatment methods. The system uses a deep learning model (ResNet-18) trained on a variety of plant disease images to accurately identify diseases and provide actionable treatment recommendations.

## Features

- **Disease Detection**: Identifies plant diseases based on uploaded images.
- **Treatment Suggestions**: Provides treatment recommendations alongside disease identification.
- **User-Friendly UI**: A modern, responsive web interface for easy interaction.
- **Backend API**: Fast and optimized API using FastAPI for efficient image processing.
- **Real-Time Predictions**: Predict plant diseases from live images and provide instant feedback.
  
## Technologies Used

- **Frontend**:
  - HTML, CSS, JavaScript
  - Responsive design (Bootstrap or custom styling)
  - AJAX for real-time image submission and results
- **Backend**:
  - Python
  - FastAPI (for building the REST API)
  - PyTorch (for deep learning model)
  - OpenCV (for image preprocessing)
- **Model**:
  - ResNet-18 pre-trained model for plant disease detection
  - Trained using a custom dataset of plant disease images (PlantVillage dataset)

## Setup and Installation

Follow these steps to set up and run the Green Scan project locally.

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

### 1. Clone the Repository

Clone this repository to your local machine:
```bash
git clone https://github.com/yourusername/green-scan.git
cd green-scan
2. Install Dependencies
Create a virtual environment and install the required dependencies:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate    # On Windows, use 'venv\Scripts\activate'
pip install -r requirements.txt
3. Prepare the Dataset
Make sure to place the dataset in the correct folder. The dataset should be organized as follows:

plaintext
Copy
Edit
green-scan/
│
└── backend/
    └── datasets/
        └── PlantVillage/
            ├── healthy/
            ├── disease1/
            ├── disease2/
            └── ...
4. Run the Backend API
To start the backend, use the following command:

bash
Copy
Edit
python backend/api.py
This will start the FastAPI server, which will be accessible at http://127.0.0.1:8000.

5. Run the Frontend
Open the frontend/index.html in a web browser to access the UI and start uploading plant images for detection.

6. Run the API Using Batch File (Optional for Windows)
For Windows users, you can use the provided run_api.bat file for easy backend startup. Just double-click the .bat file to start the backend API automatically.

How It Works
Upload an Image: The user uploads an image of a plant to the web interface.
Model Inference: The backend processes the image, and the trained model identifies the disease in the plant.
Result Display: The detected disease name and suggested treatment are displayed on the frontend.
Treatment Suggestions: The model also provides a treatment recommendation based on the identified disease.
Model Details
Model: ResNet-18, trained on the PlantVillage dataset.
Input: Images of plants (healthy or diseased).
Output: Disease label and treatment suggestions.
Example Output
json
Copy
Edit
{
  "disease": "Early Blight",
  "confidence": 92.5,
  "treatment": "Apply fungicide and remove infected plant parts."
}
Project Structure
The project is organized as follows:

plaintext
Copy
Edit
green-scan/
├── backend/                            # Backend folder containing API and model-related files
│   ├── api.py                          # FastAPI backend script
│   ├── models/                         # Trained model files
│   ├── datasets/                       # Dataset files
│   ├── requirements.txt                # Python dependencies
│   └── utils/                          # Helper scripts (e.g., image processing)
├── frontend/                           # Frontend folder containing UI files
│   ├── index.html                      # Main HTML file for the UI
│   ├── styles.css                      # Stylesheet for the frontend
│   ├── scripts.js                      # JavaScript for interactivity
├── datasets/                           # Folder for storing plant disease images
│   ├── PlantVillage/                   # Disease image dataset
│   └── ...
└── README.md                           # Project documentation
Contributing
If you would like to contribute to this project, please fork the repository, make your changes, and create a pull request. All contributions are welcome!

Acknowledgements
The model is based on the PlantVillage dataset, which contains labeled images of plant diseases.
Special thanks to the developers of FastAPI and PyTorch.
markdown
Copy
Edit







 a99892a0 (Initial commit)
