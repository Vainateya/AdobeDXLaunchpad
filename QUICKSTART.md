# 🚀 Quick Start Guide

Get your Adobe Resources Expert chatbot running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (get one at https://platform.openai.com/api-keys)

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set Up Environment

1. Copy the environment template:

   ```bash
   cp env_example.txt .env
   ```

2. Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

## Step 3: Run the Chatbot

### Option A: Web Interface (Recommended)

```bash
streamlit run rag_chatbot.py
```

Then open your browser to the URL shown (usually http://localhost:8501)

### Option B: Command Line

```bash
python simple_chatbot.py
```

## Step 4: Test It!

Try these example questions:

- "What Adobe Analytics courses are available?"
- "Tell me about Adobe Commerce certifications"
- "What are the prerequisites for Expert level certifications?"
- "How much do Adobe certifications cost?"

## Troubleshooting

### If you get import errors:

```bash
pip install -r requirements.txt
```

### If you get API key errors:

Make sure your `.env` file contains a valid OpenAI API key

### If the setup is slow:

The first run creates a vector database. Subsequent runs will be much faster.

## Need Help?

Run the test script to check your setup:

```bash
python test_setup.py
```

Or use the automated setup:

```bash
python setup.py
```

---

**That's it! You're ready to chat with your Adobe Resources Expert! 🎓**
