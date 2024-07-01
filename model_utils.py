import transformers
from torch import cuda, bfloat16
import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)


class ModelLoader:
    def __init__(self, model_id, hf_auth):
        self.model_id = model_id
        self.hf_auth = hf_auth
        self.device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        self.model = self.initialize_model()
        self.tokenizer = self.load_tokenizer()

    def initialize_model(self):
        # Set quantization configuration
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype='bfloat16'
        )

        # Initialize and load the model
        model_config = AutoConfig.from_pretrained(
            self.model_id,
            use_auth_token=self.hf_auth
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=self.hf_auth,
            # load_in_8bit_fp32_cpu_offload=True,  # Enable CPU offloading
        )

        # Enable evaluation mode
        model.eval()

        return model
    

    def load_tokenizer(self):
        # Initialize and load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_auth_token=self.hf_auth
        )
        return tokenizer