from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

model_path = "./Qwen25"

tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path)
# config = AutoConfig.from_pretrained(model_path)

s = "Hello, 世界！"
print(s)
inputs = tokenizer(s, return_tensors="pt")
print(inputs)
# outputs = model.generate(**inputs, max_new_tokens=100)
# print(tokenizer.decode(outputs[0]))

# model = AutoModelForCausalLM.from_config(config)
# print(model)
