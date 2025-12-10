from transformers import AutoTokenizer, AutoModel
model_id = "intfloat/multilingual-e5-large"
print("Downloading tokenizer...")
AutoTokenizer.from_pretrained(model_id)
print("Downloading model...")
AutoModel.from_pretrained(model_id)
print("Done")
