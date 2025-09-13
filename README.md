# COVID-19 Detection from Chest CT Scans

This project explores **deep learning models** for detecting COVID-19 from chest CT scan images.  
The work compares image classification architectures (DenseNet and ResNet) and includes a simple web app interface for easy usage.

---

## 📌 Introduction
The COVID-19 pandemic highlighted the urgent need for accurate diagnostic tools.  
While object detection and segmentation approaches were initially considered, the lack of annotated bounding box data led us to focus on **image classification**.

Key CT scan features associated with COVID-19 include:
- Ground-glass opacities
- Consolidation
- Fibrotic lesions
- Pleural effusion
- Thoracic lymphadenopathy  
...among others.

---

## 📊 Dataset
- **Total images:** 3,227 CT scans  
  - 1,601 COVID positive  
  - 1,626 COVID negative  
- Sources:  
  - [COVIDNet-CT](https://github.com/haydengunraj/COVIDNet-CT)  
  - [COVID-CT](https://github.com/UCSD-AI4H/COVID-CT)  
- Train/Validation split:  
  - **2,581 training images**  
  - **646 validation images**  

An additional Kaggle dataset ([COVIDx CT](https://www.kaggle.com/datasets/hgunraj/covidxct)) was tested but discarded due to class imbalance.

---

## 🧠 Model Architectures

### DenseNet169
- 169 layers with dense connectivity.
- Achieved **99% training accuracy** and **88% validation accuracy**.
- Challenges:
  - Lack of pretrained weights (trained from scratch).
  - Overfitting (reduced epochs from 100 → 30).

### ResNet50
- 50-layer residual network.
- Achieved **99% training accuracy** and **87% validation accuracy**.
- Also trained from scratch due to unavailability of pretrained weights.

---

## 🔬 Comparison
| Model      | Training Accuracy | Validation Accuracy |
|------------|-------------------|---------------------|
| DenseNet169| 99%               | **88%**             |
| ResNet50   | 99%               | 87%                 |

👉 **DenseNet169 slightly outperforms ResNet50** on this dataset.

---

## 💻 Web Application
A simple web app was developed using **Flask, HTML, and CSS** to make the model accessible for testing.  
Users can upload a CT scan image and get a prediction on whether it indicates COVID-19.

---

## ✅ Conclusion
- **DenseNet169** proved to be the best-performing model.  
- Object detection and segmentation were avoided due to lack of annotated datasets.  
- Image classification with DenseNet provided reliable results for COVID-19 detection.

---

## 📚 References
1. *COVID-19 Detection in Chest X-rays with Convolutional Neural Networks*  
2. *CT image visual quantitative evaluation and clinical classification of coronavirus disease (COVID-19)*  
3. *End-to-End Object Detection with Transformers*  
4. *Focal Loss for Dense Object Detection*  
5. Huang, G., Liu, … *DenseNet*  
6. [COVIDNet-CT Dataset](https://github.com/haydengunraj/COVIDNet-CT)  
7. [COVID-CT Dataset](https://github.com/UCSD-AI4H/COVID-CT)  
8. [COVIDx CT (Kaggle)](https://www.kaggle.com/datasets/hgunraj/covidxct)  

---

## 🚀 Future Work
- Add transfer learning with pretrained medical imaging models.  
- Explore ensemble methods for improved robustness.  
- Expand dataset with more balanced positive/negative cases.  
