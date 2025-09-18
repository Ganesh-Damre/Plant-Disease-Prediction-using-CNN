# 🌱 Plant Disease Prediction using CNN

A **deep learning-powered image classifier** that detects plant diseases from leaf images using a **Convolutional Neural Network (CNN)**.  
This project includes a **Streamlit web interface** for real-time image classification.

---

## 📂 Kaggle Dataset
We used the **PlantVillage Dataset** for training.  
🔗 **Dataset Link:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

🔗 **Pretrained Model Link:** (https://drive.google.com/file/d/1xcjt31VqWe7e5meD2OFJLm6dDoOJM-q7/view?usp=sharing) 
download this and paste it in app/trained_model/ 


---

## 📁 Project Structure

```
Plant-Disease-Prediction/
│
├── app/
│   ├── main.py                # Streamlit app entry point
│   ├── class_indices.json     # Label-to-class mapping
│   ├── requirements.txt       # Project dependencies
│   ├── trained_model/         # Saved CNN model (.h5 file)
│   ├── config.toml            # Streamlit configuration
│   ├── credentials.toml       # Placeholder for Streamlit credentials
│   └── README.md              # Project documentation (this file)
│
└── Plant_Disease_Prediction_CNN_Image_Classifier.ipynb  # Model training & experimentation notebook
```

---

## 🚀 Features
- **Image Upload & Preview** – Upload plant leaf images (JPG, PNG, JPEG) and preview them before classification.
- **Deep Learning Model** – CNN trained on a dataset of 38 plant disease classes.
- **Real-time Prediction** – Get instant classification results.
- **Interactive Web App** – Built with Streamlit for a simple and clean user interface.
- **Easy to Deploy** – Just install dependencies and run.

---

## 🧠 Supported Plant Diseases
The model can classify **38 categories**, including:
- **Apple**: Apple Scab, Black Rot, Cedar Rust, Healthy  
- **Corn (Maize)**: Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy  
- **Grape**: Black Rot, Esca, Leaf Blight, Healthy  
- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Septoria Leaf Spot, Spider Mites, Target Spot, TYLCV, Mosaic Virus, Healthy  
- And many more crops including **Peach, Pepper, Potato, Strawberry, Orange, Soybean, Squash, Raspberry**.



---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** TensorFlow, NumPy, Pandas, Scikit-learn  
- **Web App:** Streamlit  
- **Model:** CNN (trained on PlantVillage dataset)

---

## ⚡ Installation & Setup

1. **Clone the Repository**
```bash
git clone https://github.com/Ganesh-Damre/Plant-Disease-Prediction-using-CNN.git
cd Plant-Disease-Prediction-using-CNN/app
```

2. **Create Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the App**
```bash
streamlit run main.py
```

5. **Open in Browser**
Visit the link shown in your terminal.

---

## 📊 Model Details
- **Architecture:** CNN
- **Input Shape:** 224×224×3
- **Dataset:** PlantVillage (38 classes)
- **Framework:** TensorFlow / Keras

---

## 🎯 Usage
- Upload a plant leaf image.
- Click **Classify**.
- View the predicted disease instantly.

---

## 🧩 Future Improvements
- Add **confidence score visualization**.
- Show **remedy suggestions** for each detected disease.
- Deploy to **Streamlit Cloud / Hugging Face Spaces**.

---


