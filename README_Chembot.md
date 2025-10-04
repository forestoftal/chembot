# 🧪 Chembot — Molecular Modeling AI Agent

Chembot is an **AI-driven chemistry assistant** designed to streamline molecular modeling workflows for research and education.  
It converts **natural language and voice commands into SMILES codes** and visualizes them as **2D/3D molecular structures**, enhancing accessibility and efficiency in computational chemistry.

> ⚠️ This project is currently a **prototype AI Agent for research and educational purposes only**, not for commercial use.  
> Future development aims to integrate **reinforcement learning–based molecular modeling tools** and collaborative extensions with other cheminformatics systems.

---

## 🧬 Features

- **Voice & Text Interface:** Automatically detects available microphones for voice input using Whisper API.  
- **Natural Language to Structure:** Uses OpenAI GPT-4 to interpret chemistry-related commands and translate them into SMILES codes.  
- **Molecular Visualization:** Displays both 2D and 3D molecular structures using RDKit.  
- **Cross-Platform Compatibility:** Developed in Linux but includes a `windowenv.yaml` file for Windows-based environments.  
- **Personal API Key Integration:** Users must connect their **own OpenAI API key** in the `.env` file for the application to function properly.  

---

## ⚙️ Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/YOUR_USERNAME/Chembot.git
   cd Chembot
   ```

2. Set up the conda environment (Windows version):  
   ```bash
   conda env create -f windowenv.yaml
   conda activate chembot
   ```

3. Create a `.env` file and add your OpenAI API key:  
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```

4. Run the app:  
   ```bash
   python chembot.py
   ```

---

## 🧰 Tech Stack

- **Languages:** Python  
- **AI Models:** OpenAI GPT-4, Whisper (speech recognition)  
- **Libraries:** RDKit, wxPython, PyAudio, OpenAI SDK  
- **Environment:** Conda (`windowenv.yaml` provided for Windows setup)  

---

## 🚀 Roadmap

- Integration with **reinforcement learning–based molecular modeling systems**.  
- Collaborative compatibility with external tools (e.g., AutoDock, DeepChem).  
- Improved UI and cross-platform optimization.

---

## 📜 License
This project is open for **educational and non-commercial use** only.  
All rights reserved by the original author (Yerim Kim).
