
# 🤖 DEXY – AI Meeting Agent
## AI-Powered Meeting Assistant

DEXY is an AI-based Meeting Assistant built using Python, NLP, LLMs, and Automation.
It automatically participates in meetings, generates transcripts, summarizes discussions, and emails the Minutes of Meeting (MoM) to participants.

### 📌 Project Overview

DEXY acts as a smart virtual meeting agent that:

- 🎙️ Listens to live meetings

- 📝 Converts speech to text (Transcription)

- 🧠 Uses Large Language Models (LLMs) to summarize discussions

- 📧 Automatically sends structured Minutes of Meeting via email

This project helps teams save time, improve productivity, and maintain proper documentation.

### 🚀 Features

- ✅ Automatic Speech-to-Text transcription

- ✅ Intelligent summarization using LLM

- ✅ Action item extraction

- ✅ Structured Minutes of Meeting (MoM) generation

- ✅ Automated email delivery to participants

- ✅ Clean and readable meeting reports

🛠️ Tech Stack
Technology	Purpose
Python	Core development
NLP	Text processing & summarization
LLMs	Context understanding & MoM generation
Speech-to-Text API	Meeting transcription
SMTP / Email API	Automated email delivery
Automation Scripts	Workflow execution
🏗️ System Architecture

Meeting Audio Input

Speech-to-Text Conversion

Text Cleaning & NLP Processing

LLM-based Summarization

MoM Generation

Email Automation

📂 Project Structure
DEXY-Meeting-Agent/
│
├── main.py
├── transcription.py
├── summarizer.py
├── mom_generator.py
├── email_automation.py
├── requirements.txt
└── README.md
⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/yourusername/dexy-meeting-agent.git
cd dexy-meeting-agent
2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Add API Keys

Create a .env file and add:

OPENAI_API_KEY=your_api_key
EMAIL_ID=your_email
EMAIL_PASSWORD=your_password
5️⃣ Run the Project
python main.py
📊 Sample Output (MoM Format)

Meeting Title: Weekly Project Discussion
Date: 12 Feb 2026

Summary:

Discussed dashboard integration

Identified deployment blockers

Action Items:

Vaibhavi to complete API integration

Team to test staging server

Next Meeting: 18 Feb 2026

🎯 Use Cases

Corporate Meetings

Academic Project Discussions

Client Calls

Standup Meetings

Remote Team Collaboration

📈 Future Enhancements

🌍 Multilingual Support

📅 Calendar Integration

🔗 Slack / Teams Integration

📊 Dashboard for meeting analytics

🔐 Secure cloud deployment

👩‍💻 Author

