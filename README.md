# Neural Additive Models (NAMs): Interpretable Deep Learning for Tabular Data

## 📄 Table of Contents

1. [Motivation & Problem Statement](#motivation--problem-statement)
2. [NAMs vs. GAMs: The Best of Both Worlds](#nams-vs-gams-the-best-of-both-worlds)
3. [How Neural Additive Models Work](#how-neural-additive-models-work)
4. [What This Repo Does](#what-this-repo-does)
5. [Applications](#applications)
6. [Project Structure](#project-structure)
7. [Quickstart](#quickstart)
8. [How To Use for Your Own Data](#how-to-use-for-your-own-data)
9. [Documentation and Notebooks](#documentation-and-notebooks)
10. [Extending the Repo](#extending-the-repo)
11. [Contributing](#contributing)
12. [Citation](#citation)
13. [FAQ](#faq)

---

## ✨ Motivation & Problem Statement

Deep neural networks (DNNs) have revolutionized machine learning due to their predictive power, but they are notorious black boxes—making it hard (or impossible) to understand *why* they make certain predictions. This lack of transparency is a serious limitation in high-stakes domains such as healthcare, finance, and public policy, where interpretability is as crucial as accuracy.

Classical approaches like Generalized Additive Models (GAMs) offer full transparency: they let you visualize and trust every feature’s contribution, but often struggle to match the accuracy of neural networks when relationships are highly nonlinear or data is complex.

**Goal:** Combine the *interpretability* of GAMs with the *power and flexibility* of neural networks.

---

## 🔎 NAMs vs. GAMs: The Best of Both Worlds

**Generalized Additive Models (GAMs):**
- Predict the outcome as a sum of independent feature effects:
 <img width="611" height="66" alt="image" src="https://github.com/user-attachments/assets/7e0b2a6f-5967-40df-8e80-a0efe9eb1649" />

- Each \( f_i(x_i) \) is a simple function (like a spline or low-degree polynomial).
- **Pros:** Easy to interpret—one can plot and analyze each feature’s effect.
- **Cons:** Limited to simple relationships; can’t model intricate, high-dimensional patterns.

**Neural Additive Models (NAMs):**
- Same additive structure as GAMs, but **each \( f_i(x_i) \) is parameterized as a neural network**, allowing much greater complexity:
 <img width="646" height="69" alt="image" src="https://github.com/user-attachments/assets/71061f4a-bb64-4ee8-af11-b9c3693c039d" />

- **Every feature has its own neural sub-network** (often a multilayer perceptron).
- **Each sub-network can learn flexible, nonlinear mappings** from feature input to effects.
- Retains all the interpretability of GAMs: still possible to plot \( f_i \) for each feature!
- **Result:** Nearly the accuracy of black-box DNNs, but each variable's role remains “glass box” explainable.

---

## 🛠️ How Neural Additive Models Work

1. **Additive Decomposition:**  
   The model prediction is the sum of learnable functions, one for each input feature. This means each variable’s contribution is handled independently.

2. **Neural Parameterization:**  
   Instead of fitting splines or polynomials for each \( f_i \), NAMs use *small neural networks* for each. Each sub-network is free to learn any function suitable for the feature.

3. **Training:**  
   All sub-networks are trained together, using the same loss and optimizer as a regular DNN. The outputs of all feature sub-networks are summed to give the total prediction.

4. **Interpretability:**  
   Because each input’s effect is modeled alone, the shape \( f_i(x_i) \) for every feature can be visualized—just like in old-school GAMs, but with more expressive, data-driven patterns.

5. **Use Cases:**  
   Whenever *interpretability + nonlinearity* are key: medicine, law, regulatory compliance, audit trails, business analytics.

---

## 🚀 What This Repo Does

- Provides a **tutorial-style pipeline**: data loading/prep → NAM training → evaluation → interpretation, *all fully reproducible*.
- Lets you **train NAMs on your own tabular data** in minutes.
- Compares NAM performance and transparency to black-box models and classical interpretable models.
- Features **visualizations and notebooks** for exploring individual feature effects.
- (Optional) Includes a **web interface/app** for uploading data, scoring, and seeing live interpretation.

---

## 📈 Applications

- **Healthcare analytics:** E.g., predicting risk of diabetes or heart disease, with doctor-friendly explanations.
- **Credit scoring:** Transparent approval/rejection decisions, aiding fairness and compliance.
- **Predictive maintenance:** Equipment failure prediction with verifiable variable importance.
- **Human resources:** Interpretable attrition and hiring models.
- **Business analytics:** Data-driven, actionable insights for operations and marketing, with transparent reasoning.

---

---

## ⚡ Quickstart

1. **Clone the repo:**

```bash
git clone https://github.com/yourusername/nam-interpretable
cd nam-interpretable
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Prepare data:**
- Place your CSV in `data/`, or use the sample provided (e.g., `PIMA_diabetes.csv`).

4. **Train a NAM:**

```bash
python scripts/train_nam.py --data_file data/PIMA_diabetes.csv
```
( I just put my file location here for reference , change it for yours )


5. **Visualize Feature Effects:**

```bash
python scripts/visualize_nam.py --model_path models/trained_nam.pth
```

6. **(Optional) Start Web App:**

```bash
streamlit run app/demo.py
```

---

## 🧑‍💻 How To Use for Your Own Data

1. Drop your tabular dataset (CSV) into `data/`.
2. Edit config or script flags for your input columns/target.
3. Run the training and visualization steps as above.
4. Use the web app for interactive analysis or predictions.

---

## 📝 Documentation and Notebooks

See `notebooks/` for:
- *Theory and Motivation* (NAM theory, design philosophy)
- *Practical Tutorials* (data prep, NAM training, visualization)
- *Comparisons to Black-Box and Traditional Models*

---

## ⚡ Extending the Repo

- Swap in your dataset or custom NAM architectures
- Add new visualization methods or web components
- Implement new applications: credit scoring, disease screening, HR analytics, etc.
- Contribute improvements via Pull Requests!

---

## 🤝 Contributing

Open to issues, bug reports, and PRs!
- Please follow our [Contributing Guidelines](CONTRIBUTING.md)
- For feature requests, open an issue and label it as `enhancement`

---

## 📜 Citation

If you use this repo or the main NAM technique, please cite:
@inproceedings{
agarwal2021neural,
title={Neural Additive Models: Interpretable Machine Learning with Neural Nets},
author={Rishabh Agarwal et al.},
booktitle={NeurIPS},
year={2021}
}


---

## 🙋 FAQ

### Q: What’s required to run this repo?
**A:** Python 3.8+, PyTorch/TensorFlow (see `requirements.txt`), matplotlib/seaborn, pandas.

### Q: What datasets are supported?
**A:** Any tabular data (CSV) with numerical/categorical features.

### Q: Can I use this for regression?
**A:** Yes, just adjust the output layer and loss function.

---

**Ready to help people trust powerful neural models in practice!**


