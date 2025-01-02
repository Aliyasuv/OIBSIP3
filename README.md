# **Spam vs. Ham Classifier**
A machine learning project to classify text messages as **Spam** or **Ham (Not Spam)** using **Naive Bayes** and text vectorization.

## **Overview**
This project demonstrates how to build a text classification model to differentiate between spam and non-spam messages. It uses:
- Text preprocessing with `CountVectorizer` to transform text into feature vectors.
- A **Multinomial Naive Bayes** algorithm to classify the messages.
- Evaluation metrics like accuracy, confusion matrix, and classification report.

---

## **Features**
- Train a spam detection model on labeled text data.
- Evaluate the model’s performance using:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
- Visualize Spam vs. Ham distributions.

---

## **Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aliyasuv/OIBSIP3.git
   cd spam-vs-ham-classifier
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

1. **Prepare your dataset:**
   - Place your labeled dataset in a CSV file named `spam_data.csv` with two columns:
     - `text`: The message text.
     - `label`: The target label (`spam` or `ham`).

2. **Run the script:**
   ```bash
   python spam_vs_ham.py
   ```
   spam_data = pd.read_csv(url = "https://storage.googleapis.com/kagglesdsdata/datasets/483/982/spam.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241009%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241009T104303Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=9ec509cd64a8cc1555ff18f62aa0c01610b6ecb5a9b95a8dc01f19b5ada751979570f9a99d5d55595ad2f992275475f3bb1b94b59c7d787ea98f38db513566bb6d4263405d31c1b79a0de6507433dad69b237ff9773a046b4736ff1470cd225583e1cacd428890621009ab4186ae5e8adf450c462ad92d9cf3da5f16c58b7f08739c8e5e7ae5b8f3e9d5a1b80b7956e324e174df3df802ef8822a9fa3c92e2d250a63244062240d47210f2a647568a89fc625c8eea2ac0234727fcedaf13bcada138b218fcba71446c9be90356a5f09cde4c7ab6514a5642902e5881cea6fabb2270395dadc798bef022ec8ecadaf2226162267c7a1b5a7f41ad15d4cad57af2"
)  


3. **Outputs:**
   - Model accuracy and performance metrics.
   - A plot visualizing the distribution of spam vs. ham messages.

---

## **Project Structure**
```
spam-vs-ham-classifier/
│
├── spam_vs_ham.py        # Main script to train and evaluate the model
├── spam_data.csv         # Example dataset (replace with your own data)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation

---

# **Spam vs. Ham Classifier**
## 1. Import Libraries
## 2. Load and Explore Dataset
## 3. Data Preprocessing(In this step, we clean the dataset and convert text data into numerical vectors using `CountVectorizer`.)
## 4. Model Training
## 5. Evaluation
## 6. Visualization


## **Results**
- **Accuracy:** ~85-95% depending on the dataset.
- **Confusion Matrix Example:**
  ```
  [[950   5]
   [ 20  250]]
  ```
- **Classification Report Example:**
  ```
                precision    recall  f1-score   support

        ham       0.98      0.99      0.98       955
       spam       0.98      0.93      0.95       270

    accuracy                           0.98      1225
   macro avg       0.98      0.96      0.97      1225
weighted avg       0.98      0.98      0.98      1225
  ```
##*Requirements.txt**
pandas==1.4.3
numpy==1.22.4
scikit-learn==1.0.2
matplotlib==3.5.1
jupyter==1.0.0


---

## **Technologies Used**
- **Python**: Core programming language.
- **Scikit-learn**: For text vectorization and Naive Bayes modeling.
- **Matplotlib**: For data visualization.
Install dependencies using:
pip install -r requirements.txt


---

## **Future Improvements**
- Integrate more advanced vectorization methods like **TF-IDF** or **word embeddings**.
- Explore deep learning models such as **LSTMs** or **Transformers**.
- Deploy the model as a web app using **Flask** or **Streamlit**.

---

## **Contributing**
Feel free to contribute to this project by:
1. Forking the repository.
2. Creating a new branch.
3. Submitting a pull request.

---

## **License**
All rights reserved.
---

## **Contact**
If you have any questions or suggestions, please feel free to reach out:
- **GitHub**: [Aliyasuv](https://github.com/Aliyasuv)
- **Email**: aliya.ansari1685@gmail.com

---
##**Binder Batch**
https://mybinder.org/v2/gh/Aliyasuv/OIBSIP3/main

Let me know if you’d like adjustments or additional sections!
