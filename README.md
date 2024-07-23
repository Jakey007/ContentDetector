# Content Detector

## Data
The dataset used in this study comprises over 750,000 text samples, including both AI-generated and human content. Each sample has four attributes: the text content, its word count, a prompt ID, and the source of the text. 

For the purpose of this analysis, only the text content and the source were selected from the dataset. 

This dataset was attained from Kaggle from the following source:

@misc{zachary_grinberg_2024,
	title={Human vs. LLM Text Corpus},
	url={https://www.kaggle.com/dsv/7378735},
	DOI={10.34740/KAGGLE/DSV/7378735},
	publisher={Kaggle},
	author={Zachary Grinberg},
	year={2024}
}

## Training Models
The model was developed using a random sample on the entire dataset. Analyzing the entire dataset using neural networks is computationally intesive. To manage this, a representative sample of 150,000 pieces of text, or around 20% of the entire dataset, was selected for model development. 

Each text sample was then vectorized using the TfidfVectorizer, a technique used to transform text into numerical representations for machine learning. The dataset was then split into training and testing subsets, with the testing subset comprising 20% of the data. This split ensures that the model's performance can be evaluated on unseen data, helping to gauge its accuracy and performance. 

Initially, preprocessing steps such as removing punctuation and stop words were considered to increase model performance. However, after 

The neural network was trained on the dataset using the Adam optimizer due to its efficiency with large datasets. The trained model was then saved for future use, allowing for later usage and predictions. 

## Evaluating Models
The model's performance was evaluated using three main methods, each detailed below. 

Confusion Matrix:
A confusion matrix was used to visualize the performance of the model in predicting the source of text. It was used to analyze the amount of True Positives, True Negatives, False Positives, and False Negatives. The resulting confusion matrix can be seen below:

|             | Predicted AI Generated | Predicted Human Content |
|-------------|------------------------|-------------------------|
| Actual AI Generated     | 15811                  | 1023                    |
| Actual Human Content    | 2082                   | 11084                   |

Classification Report
A classification report was used to provide details on performance metrics of the model. This includes precision, recall, and F1-score for AI Generated and Human Content. The resulting classification report can be seen below:

| Class          | Precision | Recall | F1-Score |
|----------------|-----------|--------|----------|
| AI Generated   | 0.88      | 0.94   | 0.91     |
| Human Content  | 0.92      | 0.84   | 0.88     |

ROC-AUC Score:
The ROC-AUC score was used to quantify the model's ability to distinguish content across thresholds. This score can be seen below:

ROC-AUC Score: 0.9621

The overall accuracy of the model, as evaluated against the testing subset can be seen below:

Test Accuracy: 0.8965

## Using the Model
To use the model to make predictions on new text, create a .txt file with the text you would like to classify. Then, in the terminal, run `python ContentDetector.py [path to txt file]`. The predicted output will then be printed to the terminal. 
