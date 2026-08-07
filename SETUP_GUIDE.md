# 🚀 Local Model Backend Setup Guide (Run on Any Laptop)

Follow these simple step-by-step instructions to set up and run the CXR-GNN AI Model Backend on **any laptop or computer** (Windows, Mac, or Linux).

---

## 📋 Prerequisites
- **Python 3.11** (or 3.10) installed.
- **Git** and **Node.js** (optional, for localtunnel).

---

## 💻 Step-by-Step Setup

### Step 1: Clone the Repository
Open Terminal / PowerShell on the laptop and run:
```bash
git clone https://github.com/eggshell-coder/X-Ray-app.git
cd X-Ray-app/backend
```

### Step 2: Create a Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt / CMD)**:
  ```cmd
  venv\Scripts\activate.bat
  ```
- **Mac / Linux**:
  ```bash
  source venv/bin/activate
  ```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Start the Model Backend Server
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🌐 Share Backend to Render Website (Tunneling)

Open a **second terminal window** on the laptop and run:
```bash
npx localtunnel --port 8000
```

Copy the generated URL (e.g. `https://xxxx.loca.lt`) and paste it into the **⚙️ Settings Modal** on your Render Website!
