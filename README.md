# 😄 Emotion Detection App

A multilingual emotion detection web app built using **Streamlit** and **Scikit-learn**, capable of:
- Predicting emotions from user input text
- Translating non-English input
- Logging predictions with timestamps
- Allowing retraining by adding new sentences
- Supporting `.txt` dataset uploads
- Running in dark mode with custom theming

---

## 🚀 Features

- 🔍 **Emotion Detection** from user-entered sentences
- 🌐 **Multilingual Input Support** via Google Translate
- 🧠 **Machine Learning Model** using Logistic Regression
- 💬 **Add & Retrain Model** on new sentences without restarting
- 📄 **Upload Custom Dataset** in `[vector] sentence` format
- 🧾 **Emotion Log Viewer** with a 🧼 "Clear Log" option
- 🌙 **Dark Mode** for a modern, sleek UI

---

## 📂 Dataset Format

Dataset (`text.txt`) must follow this format:

[1 0 0 0 0 0 0] I'm feeling joyful!

[0 0 0 1 0 0 0] This is so sad.

Each label vector corresponds to:

`[joy, fear, anger, sadness, disgust, shame, guilt]`

---

## 🛠 Installation

### 1. Clone the Repository

git clone https://github.com/Shravika121/emotion-detector.git
cd emotion-detector

### 2. Install Dependencies

Create and activate a virtual environment :

python -m venv .venv

.venv\Scripts\activate  # Windows

source .venv/bin/activate  # macOS/Linux

Then install requirements:

pip install -r requirements.txt

### 3. Run the App

streamlit run app.py

## 📁 Project Structure

emotion-detector/

├── app.py  

├── text.txt

├── emotion_log.txt

├── requirements.txt 

└── README.md             

### 🧪 Dependencies

streamlit

scikit-learn

pandas

deep-translator

langdetect

Install all via:

pip install -r requirements.txt


## 📸 Screenshots

<img width="1354" height="699" alt="Screenshot 2025-07-23 005216" src="https://github.com/user-attachments/assets/5d0a0fb4-e5ee-4c1d-b7f7-00bbad92f28c" />
<img width="1365" height="689" alt="Screenshot 2025-07-22 225725" src="https://github.com/user-attachments/assets/3ef42326-8757-4beb-915a-6b7669f355db" />
<img width="1358" height="672" alt="Screenshot 2025-07-23 005252" src="https://github.com/user-attachments/assets/55980c85-671a-4344-9b28-b7326c9a8535" />
<img width="1351" height="676" alt="Screenshot 2025-07-23 005435" src="https://github.com/user-attachments/assets/c2c8181b-7224-4b3c-b942-62e0838bd2d1" />

🧠 Model Info

Vectorizer: TfidfVectorizer (1-3 grams, English stopwords)

Classifier: LogisticRegression

Training/Test split: 80/20

Dynamic updates: App retrains on refresh if new data is added

🤝 Contributing

Feel free to fork this repo and submit pull requests. For major changes, open an issue first.

📃 License

This project is open-source and available under the MIT License.

💡 Author

Built by Shravika Rajiyahan




