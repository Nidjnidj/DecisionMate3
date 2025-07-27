# 📊 DecisionMate3

**DecisionMate3** is a multilingual, professional-grade decision-support web application built with Streamlit. It supports personal, financial, business, and project planning decisions in a visual and structured format.

---

## 🌟 Features

* 🔐 User login (username only)
* 🌐 Multilingual support (English, Azerbaijani, Russian)
* 🌃 Light/Dark theme switcher
* 📂 Modular interface for different decision categories
* 📈 Tools like CAPEX/OPEX, NPV, sensitivity analysis, tornado charts, planning (Gantt/S-Curve), risk matrix (coming soon)

---

## ✨ Run Locally

1. **Clone the repository**:

   ```bash
   git clone https://github.com/YOUR_USERNAME/DecisionMate3.git
   cd DecisionMate3
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:

   ```bash
   streamlit run app3.py
   ```

---

## 🌍 Deploy on Streamlit Cloud

1. Push this project to your GitHub repository.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **“New App”** and choose:

   * Repository: `YOUR_USERNAME/DecisionMate3`
   * Branch: `main` or `master`
   * File: `app3.py`
4. Click **“Deploy”**

You’ll get a free public URL like:

```
https://your-username-decisionmate3.streamlit.app
```

---

## 📁 File Structure

```
DecisionMate3/
├── app3.py               # Main Streamlit app
├── requirements.txt      # Python dependencies
├── nijat_logo.png        # App branding/logo
├── README.md             # This file
```

---

## ⚒ Requirements

```txt
streamlit
pandas
numpy
numpy-financial
plotly
fpdf
networkx
```

---

## 📬 Feedback

For feature requests or issues, please open a [GitHub Issue](https://github.com/YOUR_USERNAME/DecisionMate3/issues).

---

## 📄 License

MIT License. Free for personal and commercial use.
