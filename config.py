import os

if os.environ.get("LLM") is None:
  os.environ['LLM'] = 'CHATGPT'
LLM = os.getenv("LLM")

if os.environ.get("LLAMA_MODEL_PATH") is None:
  os.environ['LLAMA_MODEL_PATH'] = ''
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH")

if os.environ.get("OPENAI_API_KEY") is None:
  os.environ['OPENAI_API_KEY'] = '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'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if os.environ.get("AI21_API_KEY") is None:
  os.environ['AI21_API_KEY'] = 'JYIIbUMqeZgOkABxGWTGAiqkpTEeuxrP'
AI21_API_KEY = os.getenv("AI21_API_KEY")

