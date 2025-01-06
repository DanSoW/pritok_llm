from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import nvmem
import gc
import sys

torch.cuda.empty_cache()

nvmem.printInfoCUDA()
nvmem.printMemoryUsed()

#model_name = "microsoft/Phi-3-mini-4k-instruct"
model_name = "ai-forever/rugpt3small_based_on_gpt2"
#model_name = "Qwen/Qwen2.5-Coder-3B"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

nvmem.printMemoryUsed()

#sys.exit()

input_text = "Как твои дела?"
model_inputs = tokenizer.encode(input_text, return_tensors="pt")
model_inputs = model_inputs.cuda()

#torch.cuda.empty_cache()
#gc.collect()

nvmem.printMemoryUsed()


model.generation_config.pad_token_id = tokenizer.pad_token_id
out = model.generate(model_inputs, max_new_tokens=2048,  max_length=1024, do_sample=True)[0]

generated_text = list(map(tokenizer.decode, out))[0]
#result = str(generated_text).removeprefix(input_text)

print(generated_text)



