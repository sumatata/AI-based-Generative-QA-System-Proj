AI-based Generative Email Subject Line System

This project develops and evaluates an AI-powered system that generates subject lines from email body content using natural language processing (NLP). The goal is to assist in summarizing and labeling emails by automatically proposing concise and relevant subject lines.

Project Structure
	•	email.ipynb: Main Jupyter notebook for preprocessing, modeling, and evaluation
	•	email_ui19.py: Script containing UI logic for visualizing and comparing results
  - [Dataset Folder](https://drive.google.com/drive/folders/1O9JKxKKLPeFH-ZMQTC9MmXnynLYi8UcR?usp=sharing)  
  - [Model Folder](https://drive.google.com/drive/folders/1cRVukm2uhfxyHd7nKjqTQXTddLyReLmO?usp=drive_link)

 Features
	•	Parses raw email body and generates subject lines using text summarization techniques
	•	Evaluates generated subjects against human-written annotations using:
	•	ROUGE
	•	BLEU
	•	METEOR
	•	Supports comparison across multiple annotators (ann0, ann1, ann2)
	•	Outputs quality metrics for generated vs. human subjects

 Requirements
	•	Python 3.10+
	•	Jupyter Notebook or Colab
	•	Transformers, NLTK, Pandas, Scikit-learn
	•	ROUGE, METEOR, BLEU metric libraries

 How to Run
	1.	Clone the repo and navigate into it
	2.	Open email.ipynb in Jupyter/Colab
	3.	Follow cells to:
	•	Mount Google Drive (if using Colab)
	•	Load dataset
	•	Generate subject lines
	•	Evaluate against annotations

Output
	•	Model-generated subject lines
	•	Evaluation scores (ROUGE-1/2/L, BLEU, METEOR)
	•	Side-by-side comparison with human-written subjects
