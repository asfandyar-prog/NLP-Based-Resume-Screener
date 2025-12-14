# Resume Screening App

This application is an AI-powered resume screening system that helps automate the process of resume categorization, job recommendation, and information extraction. It supports PDF and TXT file uploads and provides comprehensive resume insights to streamline candidate screening.

## Features

- **Resume Categorization**: Automatically categorizes resumes based on job titles or industry sectors.
- **Job Recommendation**: Suggests relevant job roles based on the applicant's skills and experience.
- **Information Extraction**: Extracts key details, including:
  - Name
  - Contact Information (Phone and Email)
  - Skills
  - Education
## Code Explanation Video
For a visual overview of the project, watch my video on youtube:  
**[MY_Project_video](https://youtu.be/Eb2BE3IZQX8)**
  
## Technologies Used

- **Frontend**: HTML, CSS (for styling)
- **Backend**: Python (Flask) for handling file uploads and processing.
- **Machine Learning**: Natural Language Processing (NLP) for resume parsing and categorization.
- **Libraries**: 
  - PDF and TXT file parsing
  - NLP libraries for extracting information

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/resume-screening-app.git
   cd resume-screening-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   python app.py
   ```

4. Visit `http://localhost:5000` to access the application.

## Usage

1. **Upload a Resume**: Upload a PDF or TXT file through the provided interface.
2. **View Insights**: The app displays extracted information such as:
   - Recommended job category
   - Contact details (if found)
   - Skills and education section

## Project Structure

```
├── app.py              # Main application file
├── templates/
│   └── index.html      # HTML template for the UI
├── static/
│   ├── style.css       # CSS for styling
├── requirements.txt    # Required libraries
└── README.md           # Project documentation
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

