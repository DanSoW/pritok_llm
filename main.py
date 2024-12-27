from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import nvmem
import gc

torch.cuda.empty_cache()

nvmem.printInfoCUDA()
nvmem.printMemoryUsed()

model_name = "microsoft/Phi-3-mini-4k-instruct"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")

nvmem.printMemoryUsed()

input_text = "write bubble sort on python"
model_inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

torch.cuda.empty_cache()
gc.collect()

nvmem.printMemoryUsed()

out = model.generate(model_inputs, max_new_tokens=512)[0]

generated_text = list(map(tokenizer.decode, out))[0]
result = str(generated_text).removeprefix(input_text)

print(result)



