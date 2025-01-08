import cog
from cog import BasePredictor, Input, ConcatenateIterator
import os
import time
import torch
import subprocess
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import TextIteratorStreamer
from load_prompt import get_prompt

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MODEL_CACHE = "/home/phi_model"  # "model-cache"

MODEL_URL = "https://weights.replicate.delivery/default/microsoft/Phi-3-mini-4k-instruct/model.tar"

# Структура промпт-запроса
MESSAGES="""<|system|>
{sys_prompt}<|end|>
<|user|>
{user_prompt}<|end|>
<|assistant|>
"""

#print(torch.nn.attention.SDPBackend.FLASH_ATTENTION)

attention = "eager"
# attention = "flash_attention_2"

def download_weights(url, dest):
    start = time.time()

    print("downloading url: ", url)
    print("downloading to: ", dest)

    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        # Загрузка модели в память
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # Загрузка модели по пути
        self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_CACHE, trust_remote_code=True,
                torch_dtype=torch.float16, device_map=self.device, attn_implementation=attention,
        )

        # Создание токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE)

    def predict(self) -> ConcatenateIterator[str]:
        tokens = ["The", "quick", "brown"]
        for token in tokens:
            yield token + " "

    def predict2(self, prompt: str, max_length: int = 2048, temperature: float = 0.1, top_p: float = 1.0, top_k: int = 1, 
                 repetition_penalty: float = 1.1, system_prompt: str = "You are a helpful AI assistant", seed: int = 100): #-> ConcatenateIterator[str]:
        print("predict start")
        # Запуск одиночной генерации
        if seed is None:
            seed = torch.randint(0, 100000, (1,)).item()

        torch.random.manual_seed(seed)

        #0 Определение формата сообщения
        chat_format = MESSAGES.format(sys_prompt=system_prompt, user_prompt=prompt)
        tokens = self.tokenizer(chat_format, return_tensors="pt")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, remove_start_token=True)

        input_ids = tokens.input_ids.to(device=self.device)
        max_length = input_ids.shape[1] + max_length

        generation_kwargs = dict(
            input_ids=input_ids,
            max_length=max_length,
            return_dict_in_generate=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
            do_sample=True
        )

        self.model.generate(generation_kwargs)

        #print("start thread")
        #thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        #thread.start()

        #for _, new_text in enumerate(streamer):
            #print(new_text)
            #yield new_text

        #print("joining thread")
        #thread.join()
        #print("close thread")


msg = get_prompt()
print("Prompt: ", msg)

predictor = Predictor()
print("to setup")
predictor.setup()

print("to predict")
predictor.predict2(msg)

