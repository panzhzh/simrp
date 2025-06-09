# 测试脚本 test_models.py
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 测试SentenceTransformer
print("Loading SentenceTransformer...")
sbert = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
print("SentenceTransformer loaded successfully!")

# 测试T5
print("Loading T5...")
tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
print("T5 loaded successfully!")