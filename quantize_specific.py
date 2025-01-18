from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict

def generate_decoder_map():
    decoder_map: Dict[str, str] = {
        "gpt_neox": "gpt_neox.layers",
    }
    return decoder_map

def get_mixmodel(model_1, model_2, mixconfig):
    DecoderMap = generate_decoder_map()
    try:
        decoders_1 = eval(f"model_1.{DecoderMap[mixconfig['model_family']]}")
        decoders_2 = eval(f"model_2.{DecoderMap[mixconfig['model_family']]}")
    except AttributeError as e:
            raise ValueError(f"Unsupported model family '{mixconfig['model_family']}' or invalid layer attribute: {e}")
        
    for i, model_index in enumerate(mixconfig['layers']):
        if model_index == 0:
            decoders_1[i] = decoders_2[i]
    return model_1

# Test the new model
if __name__ == "__main__":
    model_names = ["EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b", "EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b"]
    # model_names = ["EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b"]
    mixconfig = {}
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_size="left")
        tokenizer.pad_token = tokenizer.eos_token
        # model_1 = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=False, device_map="balanced")
        model_1 = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="cuda:0")
        # model_2 = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=False, device_map="balanced")
        # model_2 = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=False, device_map="cuda:0")
        layers = model_1.gpt_neox.layers
        seg_length = len(layers) // 4

        mixconfig[model_name] = [{
            'model_family': "gpt_neox",
            'layers': [
                0 if j * seg_length <= i and i < (j+1) * seg_length else 1 for i in range(len(layers))
            ],
        } for j in range(4)]
        
        del model_1, tokenizer
        torch.cuda.empty_cache()

        print(mixconfig)

    print(mixconfig)
        # mixmodel = get_mixmodel(model_1, model_2, mixconfig)
        # inputs = tokenizer("Hello, how are you?", return_tensors="pt")
        # input_ids = inputs.input_ids.to(model_1.device)
        # attention_mask = inputs.attention_mask.to(model_1.device)
        
        # with torch.no_grad():
        #     output = model_1.generate(input_ids, attention_mask=attention_mask)

        # print(tokenizer.batch_decode(output, skip_special_tokens=True))