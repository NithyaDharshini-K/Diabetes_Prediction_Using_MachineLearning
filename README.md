# **Diabetes Prediction Using Machine Learning**  

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_Vs6hAW1P_BtirGWHDsygDL4ZGEfGZ5S?usp=sharing)  

---

## 🚀 **About the Project**  
Diabetes is a prevalent chronic disease, and early detection is crucial for effective management.  
This project applies **Machine Learning** to predict whether a person has diabetes based on key medical diagnostic parameters.  

The model is trained using **Support Vector Machine (SVM) with a linear kernel**, which is well-suited for binary classification tasks.  

---

## 📂 **Dataset**  
The dataset used is sourced from the **PIMA Indians Diabetes Database** and consists of **768 samples** with the following features:  
- **Glucose Level**  
- **Blood Pressure**  
- **Insulin Level**  
- **BMI (Body Mass Index)**  
- **Age**  
- And more...  

The target variable (`Outcome`) represents:  
- `0` → Non-Diabetic  
- `1` → Diabetic  

---

## ⚙️ **Model Used**  
### **Support Vector Machine (SVM)**  

### 🔹 **Why SVM?**  
✔ Works well for **binary classification**.  
✔ Maximizes the margin for **better generalization**.  
✔ Effective in handling **high-dimensional data**.  
✔ Performs well with **medical datasets**.  

---

## 🔧 **Steps to Run the Project**  
1. Open **`diabetes_prediction.ipynb`** in **Google Colab**.  
2. Upload the dataset (**`diabetes.csv`**).  
3. Run all cells to train and evaluate the model.  
4. The model will predict whether the person is **Diabetic (`1`)** or **Non-Diabetic (`0`)**.  

---

## 📌 **Libraries Used**  
- **Pandas** → Data manipulation  
- **NumPy** → Numerical computations  
- **Scikit-learn** → Machine learning algorithms  
- **Matplotlib & Seaborn** → Data visualization  

---

## 📊 **Model Performance**  
- **Training Accuracy**: **78.66%**  
- **Testing Accuracy**: **77.27%**  

---

## 📖 **Lessons Learned**  
✅ Understanding **SVM for binary classification**.  
✅ Using Python libraries for **data preprocessing & model building**.  
✅ Evaluating model performance using **accuracy metrics**.  
✅ Importance of **standardizing medical data** for better predictions.  

---

## 🎯 **Future Improvements**  
🔹 Experiment with different models (**Random Forest, Neural Networks**) to compare accuracy.  
🔹 Tune hyperparameters for **better predictions**.  
🔹 Deploy the model using **Flask or Streamlit**.  

---

## 📜 **Acknowledgments**  
- **PIMA Indians Diabetes Database** for dataset.  
- **Scikit-learn documentation** for machine learning insights.  

🔥 **If you found this project useful, consider giving it a ⭐!**  
