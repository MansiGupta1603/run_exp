from dataclasses import dataclass
from typing import Optional, Iterator, Dict, List
import json
from pathlib import Path
import gc
import argparse
from typing_extensions import Literal
import warnings
from tqdm import tqdm
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging
)
from datasets import (
    load_dataset,
    IterableDataset
)

logging.set_verbosity_error()

@dataclass
class ModelConfig:
    model_name: str
    model_family: str
    model_path: Optional[Path]


@dataclass
class QuantConfig:
    load_in_4bit: Optional[bool]
    load_in_8bit: Optional[bool]
    bnb_4bit_quant_type: Optional[str]
    bnb_4bit_use_double_quant: Optional[bool]
    bnb_4bit_quant_storage: Optional[torch.dtype]
    bnb_4bit_compute_dtype: Optional[torch.dtype]


@dataclass
class LayerSwapConfig:
    skip_layers: Optional[List[int]]
    

def create_dtype_map() -> Dict[str, torch.device]:
    mapping = {
        ("float16", "fp16") : torch.float16,
        ("bfloat16",)       : torch.bfloat16,
        ("float32", "fp32") : torch.float32,
    }
    dtype_map = {}
    for keys, value in mapping.items():
        for key in keys:
            dtype_map[key] = value
    return dtype_map


def load_config_from_json(
    json_file: Path,
    config_type: Literal["model", "quant", "layer_swap"]
) -> Iterator[QuantConfig]:
    with open(json_file, "r", encoding="utf-8") as f:
        configs = json.load(f)
    match config_type:
        case "model":
            for config in configs:
                yield ModelConfig(**config)
        case "quant":
            for config in configs:
                yield QuantConfig(**config)
        case "layer_swap":
            for config in configs:
                yield LayerSwapConfig(**config)

def generate_decoder_map():
    decoder_map: Dict[str, str] = {
        "gpt_neox": "gpt_neox.layers",
    }
    return decoder_map

DecoderMap = generate_decoder_map()

class MemorizationAnalyser:
    def __init__(
        self,
        model_config: ModelConfig,
        quant_config: QuantConfig,
        quant_config_swap: Optional[QuantConfig],
        layer_swap_config: Optional[LayerSwapConfig],
        swap_every: Optional[int],
        dataset_name: str = "monology/pile-uncopyrighted",
        batch_size: int = 128,
        device_map: Literal["cpu", "auto"] = "auto",
        dtype_map: Dict = create_dtype_map()
    ):
        if layer_swap_config is not None and swap_every is not None:
            raise ValueError(f"Please specify only one of layer_swap_config or swap_every")
        self.model_name = model_config.model_name
        self.dataset_name = dataset_name
        self.dataset = None
        self.batch_size = batch_size
        self.device_map = args.device_map
        self.dtype_map = dtype_map
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit = quant_config.load_in_4bit,
            bnb_4bit_quant_type = quant_config.bnb_4bit_quant_type,
            load_in_8bit = quant_config.load_in_8bit,
            bnb_4bit_use_double_quant = quant_config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype = self.dtype_map[quant_config.bnb_4bit_compute_dtype]
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config = self.quant_config,
            device_map=self.device_map
        )
        self.model.eval()

        if quant_config_swap and (layer_swap_config or swap_every):
            self.quant_config_swap = BitsAndBytesConfig(
                load_in_4bit = quant_config_swap.load_in_4bit,
                bnb_4bit_quant_type = quant_config_swap.bnb_4bit_quant_type,
                load_in_8bit = quant_config_swap.load_in_8bit,
                bnb_4bit_use_double_quant = quant_config_swap.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype = self.dtype_map[quant_config_swap.bnb_4bit_compute_dtype]
            )
            self.model_swap = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config = self.quant_config_swap,
                device_map=self.device_map
            )
            self.model_swap.eval()
            self.decoder_map = generate_decoder_map()
            
            try:
                decoders_1 = eval(f"self.model.{DecoderMap[model_config.model_family]}")
                decoders_2 = eval(f"self.model_swap.{DecoderMap[model_config.model_family]}")
            except AttributeError as e:
                    raise ValueError(f"Unsupported model family '{model_config.model_family}' \
                        or invalid layer attribute: {e}")

            if layer_swap_config:
                self.layer_swap_config = layer_swap_config
                for layer in self.layer_swap_config.skip_layers:
                        decoders_1[layer] = decoders_2[layer]
            elif swap_every:
                self.swap_every = swap_every
                for layer in range(len(decoders_1)):
                    if (layer + 1) % self.swap_every == 0:
                        decoders_1[layer] = decoders_2[layer]
                        
        if (quant_config_swap is None and (layer_swap_config is not None or swap_every is not None)) or \
            (quant_config_swap is not None and layer_swap_config is None and swap_every is None):
            raise ValueError(f"Please provide both quant_config_swap and either layer_swap_config or swap_every, but not both.")
 

    def get_completion(
        self,
        max_new_tokens: int = 50,
        context_lengths: List[int]= [50, 100, 200, 500],
        target_length: int = 50,
        num_samples: int = 1000,
    ):  
        self.memorized: int = 0
        self.prompts: int = 0
        for context_length in tqdm(context_lengths, desc="Context Lengths"):
            print(f"Model: {self.model_name}, Dataset: {self.dataset_name}, \
                      Context Length: {context_length}, Target Length: {target_length}")
            for i, prompts in enumerate(tqdm(self.dataset, desc=f"Processing Context {context_length}", leave=False)): 
                
                inputs = self.tokenizer(
                    prompts["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length= max(context_lengths)+target_length,
                ).to(self.model.device)
                
                print(f"inputs: {inputs}")
                
                prompt_tokens = inputs["input_ids"][:, :context_length]
                attention_mask = inputs["attention_mask"][:, :context_length]
                target_tokens = inputs["input_ids"][:, context_length:context_length + target_length]
                
                # print(f"target tokens: {target_tokens}")
                # print(f"prompt tokens: {prompt_tokens}")
                # print(f"prompt tokens attention mask: {attention_mask}")
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        prompt_tokens,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    
                      #print(f"output_ids: {output_ids}")
                       print(f"decoded output: {self.tokenizer.batch_decode(output_ids)}")
        
                
                # print((target_tokens == output_ids[:, context_length:context_length + target_length]).all(dim=1).sum().item())
                # print(target_tokens.size())
                self.memorized += (target_tokens == output_ids[:, context_length:context_length + target_length]).all(dim=1).sum().item()
                self.prompts += len(output_ids)
                
                # print(f"prompt tokens device: {prompt_tokens.device}")
                # print(f"target tokens device: {target_tokens.device}")
                # print(f"attention_mask device: {prompt_tokens.device}")
                # print(f"inputs device: {inputs.input_ids.device}")
                
                if i * len(output_ids) >= num_samples: 
                    print(f"Memorized: {self.memorized / self.prompts}")
                    break
                
                del inputs, prompt_tokens, target_tokens, attention_mask, output_ids 
                gc.collect()
                
                    
    def sample_from_pile(self, min_length=100, batch_size=128):
        
        self.dataset : IterableDataset = load_dataset(
            self.dataset_name,
            split="train",
            streaming=True
        ).filter(
            lambda prompt: len(prompt["text"].split()) >= min_length
        ).batch(
            batch_size
        )

def run_analysis(
    model_config: ModelConfig,
    quant_config: QuantConfig,
    dataset_name: str,
    batch_size: int,
    device_map: str,
    max_new_tokens: int,
    context_lengths: List[int],
    target_length: int,
    num_samples: int,
    quant_config_swap: Optional[QuantConfig] = None,
    layer_swap_config: Optional[LayerSwapConfig] = None,
    swap_every: Optional[int] = None
) -> None:
    
    analyzer = MemorizationAnalyser(
        model_config=model_config,
        quant_config=quant_config,
        quant_config_swap=quant_config_swap,
        layer_swap_config=layer_swap_config,
        swap_every=swap_every,
        dataset_name=dataset_name,
        batch_size=batch_size,
        device_map=device_map,
    )
    
    print(f"Loaded model: {analyzer.model}")
    
    # Prepare the dataset
    analyzer.sample_from_pile(batch_size=batch_size)
    
    # Run the analysis
    analyzer.get_completion(
        max_new_tokens=max_new_tokens,
        context_lengths=context_lengths,
        target_length=target_length,
        num_samples=num_samples,
    )
    
    # Cleanup
    del analyzer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Analyze degree of memorization for a specified model and dataset, \
            varying quantization parameters"
    )
    parser.add_argument(
        "--model-config", 
        type=str,
        required = True,
        help = "Path to the model JSON configuration file"
    )
    parser.add_argument(
        "--quant-config", 
        type = str,
        required = True,
        help = "Path to the quantization JSON configuration file"
    )
    parser.add_argument(
        "--quant-config-swap", 
        type = str,
        help = "Path to the swap quantization JSON configuration file"
    )
    parser.add_argument(
        "--layer-swap-config", 
        type = str,
        help = "Path to the layer swap JSON configuration file"
    )
    parser.add_argument(
        "--dataset",
        type = str,
        default= "monology/pile-uncopyrighted",
        help = "Benchmark dataset name"
    )
    parser.add_argument(
        "--num-samples",
        type = int,
        default= 100,
        help = "Number of samples analyzed from dataset"
    )
    parser.add_argument(
        "--batch-size",
        type = int,
        default= 128,
        help = "Batch size"
    )
    parser.add_argument(
        "--swap-every",
        type = int,
        default= 1_000,
        help = "Batch size"
    )
    parser.add_argument(
        "--device-map",
        type = str,
        default="auto",
        help = "BitsAndBytes Device Map configuration"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate during analysis"
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500],
        help="List of context lengths to analyze"
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=50,
        help="Target length for analysis"
    )

    args = parser.parse_args()

    model_configs = load_config_from_json(Path(args.model_config), config_type="model")
    for model_config in model_configs:
        print(f"Model: {model_config}")
        quant_configs = load_config_from_json(Path(args.quant_config), config_type="quant")
        for quant_config in quant_configs:
            print(f"Quantization Config: {quant_config}")
            
            if args.quant_config_swap:
                quant_config_swaps = load_config_from_json(
                    Path(args.quant_config_swap),
                    config_type="quant"
                )
                
                for quant_config_swap in quant_config_swaps: 
                    
                    if args.layer_swap_config:
                        layer_swap_configs = load_config_from_json(
                            Path(args.layer_swap_config), 
                            config_type="layer_swap"
                        )
                
                        for layer_swap_config in layer_swap_configs:
                                run_analysis(
                                    model_config=model_config,
                                    quant_config=quant_config,
                                    dataset_name=args.dataset,
                                    batch_size=args.batch_size,
                                    device_map=args.device_map,
                                    max_new_tokens=args.max_new_tokens,
                                    context_lengths=args.context_lengths,
                                    target_length=args.target_length,
                                    num_samples=args.num_samples,
                                    quant_config_swap=quant_config_swap,
                                    layer_swap_config=layer_swap_config,
                                )
                    elif args.swap_every:
                        run_analysis(
                            model_config=model_config,
                            quant_config=quant_config,
                            dataset_name=args.dataset,
                            batch_size=args.batch_size,
                            device_map=args.device_map,
                            max_new_tokens=args.max_new_tokens,
                            context_lengths=args.context_lengths,
                            target_length=args.target_length,
                            num_samples=args.num_samples,
                            swap_every=args.swap_every,
                            quant_config_swap=quant_config_swap,
                        )
                        
                        
            if args.layer_swap_config:
                layer_swap_configs = load_config_from_json(
                    Path(args.layer_swap_config), 
                    config_type="layer_swap"
                )
                
                for layer_swap_config in layer_swap_configs:
                    quant_config_swaps = load_config_from_json(
                        Path(args.quant_config_swap),
                        config_type="quant"
                    )
                    
                    for quant_config_swap in quant_config_swaps:
                        print(f"Layer Swap Config: {layer_swap_config}")
                        
                        run_analysis(
                            model_config=model_config,
                            quant_config=quant_config,
                            dataset_name=args.dataset,
                            batch_size=args.batch_size,
                            device_map=args.device_map,
                            max_new_tokens=args.max_new_tokens,
                            context_lengths=args.context_lengths,
                            target_length=args.target_length,
                            num_samples=args.num_samples,
                            quant_config_swap=quant_config_swap,
                            layer_swap_config=layer_swap_config,
                        )
                
                        # analyzer = MemorizationAnalyser(
                        #     model_config=model_config,
                        #     quant_config=quant_config,
                        #     quant_config_swap=quant_config_swap,
                        #     layer_swap_config=layer_swap_config,
                        #     dataset_name=args.dataset,
                        #     batch_size=args.batch_size,
                        #     device_map=args.device_map   
                        # )
                        # print(f"Loaded model: {analyzer.model}")
                        # analyzer.sample_from_pile(
                        #     batch_size=args.batch_size
                        # )
                        # completion = analyzer.get_completion(
                        #     max_new_tokens=args.max_new_tokens,
                        #     context_lengths=args.context_lengths,
                        #     target_length=args.target_length,
                        #     num_samples=args.num_samples
                        # )
                        
                        # del analyzer
                        # gc.collect()
                        # torch.cuda.empty_cache()
            else:
                run_analysis(
                    model_config=model_config,
                    quant_config=quant_config,
                    dataset_name=args.dataset,
                    batch_size=args.batch_size,
                    device_map=args.device_map,
                    max_new_tokens=args.max_new_tokens,
                    context_lengths=args.context_lengths,
                    target_length=args.target_length,
                    num_samples=args.num_samples,
                )
                # analyzer = MemorizationAnalyser(
                #     model_config=model_config,
                #     quant_config=quant_config,
                #     dataset_name=args.dataset,
                #     batch_size=args.batch_size,
                #     device_map=args.device_map   
                # )
                # print(f"Loaded model: {analyzer.model}")
                # analyzer.sample_from_pile(
                #     batch_size=args.batch_size
                # )
                # completion = analyzer.get_completion(
                #     max_new_tokens=args.max_new_tokens,
                #     context_lengths=args.context_lengths,
                #     target_length=args.target_length,
                #     num_samples=args.num_samples
                # )
                
                # del analyzer
                # gc.collect()
                # torch.cuda.empty_cache()
