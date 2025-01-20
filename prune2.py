import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
import random
import json

class MemorizationAnalyzer:
    def __init__(self, model_name="EleutherAI/pythia-410m", device="cuda"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.full_precision_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        self.full_precision_model.eval()
        self.device = device
        self.info_list = []
        self.i = 0
        
    def get_completion(self, model, prompt, max_new_tokens=50):
        """Generate completion using greedy decoding"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Use greedy decoding
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        completion = self.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        print("\n ##################################################### \n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {completion}")
        print("\n ##################################################### \n")
        
        self.info_list.append({
            "model_name": self.model_name,
            "prompt": prompt,
            "completion": completion
        })
        self.i += 1

        return completion

    def is_extractable(self, model, sequence, context_length, target_length=50):
        """Check if a sequence is extractable given context_length tokens of prompt"""
        tokens = self.tokenizer.encode(sequence)
        if len(tokens) < context_length + target_length:
            return False
        
        prompt_tokens = tokens[:context_length]
        target_tokens = tokens[context_length:context_length + target_length]
        
        prompt = self.tokenizer.decode(prompt_tokens)
        target = self.tokenizer.decode(target_tokens)
        
        completion = self.get_completion(model, prompt, max_new_tokens=target_length)
        self.info_list[self.i - 1]['truth'] = target
        self.info_list[self.i - 1]['extractable'] = completion.strip() == target.strip()
        return completion.strip() == target.strip()

    def sample_from_pile(self, n_samples=1000, min_length=100):
        """Sample sequences from the Pile dataset"""
        dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        samples = []
        
        for example in dataset:
            text = example['text']
            if len(text.split()) >= min_length:
                samples.append(text)
                if len(samples) >= n_samples:
                    break
                    
        return samples

    def calculate_perplexity(self, model, text):
     """Calculate the perplexity of a given text."""
     model.to(self.device)
     model.eval()

     tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
     input_ids = tokens.input_ids
     with torch.no_grad():
        outputs = model(**tokens, labels=input_ids)
     loss = outputs.loss.item()
     perplexity = torch.exp(torch.tensor(loss)).item()

     return perplexity

    def evaluate_pruned_model(self, context_lengths=[50, 100, 200, 500]): 
     print("Sampling sequences from The Pile...")
     sequences = self.sample_from_pile()

     pruned_results = {}
     perplexity_scores = {}
     self.full_precision_model.to("cuda:0")   

     for context_length in context_lengths:
        print(f"\nEvaluating pruned model with context length {context_length}")
        extractable = 0
        total_perplexity = 0

        for sequence in tqdm(sequences):
            # Calculate extractability
            if self.is_extractable(self.full_precision_model, sequence, context_length):
                extractable += 1
                self.info_list[self.i - 1]['context_length'] = context_length
                self.info_list[self.i - 1]['model'] = 'pruned'
            
            # Calculate perplexity for the sequence
            perplexity = self.calculate_perplexity(self.full_precision_model, sequence)
            total_perplexity += perplexity
            self.info_list[self.i - 1]['perplexity'] = perplexity
        
        torch.cuda.empty_cache()

        # Compute averages
        pruned_results[context_length] = extractable / len(sequences)
        perplexity_scores[context_length] = total_perplexity / len(sequences)

        print(f"Fraction extractable: {pruned_results[context_length]:.3f}")
        print(f"Average perplexity: {perplexity_scores[context_length]:.3f}")

     self.full_precision_model.to("cpu")   
     torch.cuda.empty_cache()

    # Log data
     log_data = {
        "model_name": self.model_name,
        "results": pruned_results,
        "perplexity_scores": perplexity_scores,
        "details": self.info_list
     }

     json.dump(log_data, open(f'logs/info_list_pruned.json', 'w'), indent=4)

     return pruned_results, perplexity_scores


    def prune_model(self, amount=0.15):
        """Prune the model using magnitude-based pruning"""
        import torch.nn.utils.prune as prune
        for name, module in self.full_precision_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
        print("Model pruned with magnitude-based pruning.")

    

if __name__ == "__main__":
    analyzer = MemorizationAnalyzer()
    analyzer.prune_model(amount=0.15)  
    pruned_results, perplexity_scores = analyzer.evaluate_pruned_model()
    
    print("\nSummary of pruned model results:")
    for context_length in pruned_results:
     fraction = pruned_results[context_length]
     perplexity = perplexity_scores[context_length]
     print(f"Context length {context_length}: {fraction:.3f} extractable, Perplexity: {perplexity:.3f}")
