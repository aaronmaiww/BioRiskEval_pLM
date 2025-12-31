"""
Common utility functions for BioRiskEval evaluation scripts.
"""

from typing import Optional, Tuple
import os
import torch
from transformers import AutoTokenizer, EsmForMaskedLM


FACEBOOK_CONFIG = {
    "8M":   "facebook/esm2_t6_8M_UR50D",
    "35M":  "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
}


def parse_model_tier(model_name: str) -> str:
    """
    Parse model tier from HuggingFace model name.
    
    Args:
        model_name (str): Model name like "given131/8M_T1" or "facebook/esm2_t6_8M_UR50D"
    Returns:
        str: Tier number (e.g., '1', '2', '3', 'H', 'F')
    """
    if "T1" in model_name:
        return "1"
    elif "T2" in model_name:
        return "2"
    elif "T5" in model_name:
        return "5"
    elif "T6" in model_name:
        return "6"
    elif "H" in model_name:
        return "H"
    elif "F" in model_name:
        return "F"
    else:
        # Facebook 모델의 경우 tier를 추출
        if "t6" in model_name.lower():
            return "6"
        elif "t12" in model_name.lower():
            return "12"
        elif "t30" in model_name.lower():
            return "30"
        # 기본값으로 "unknown" 반환 (에러 대신)
        return "unknown"


def parse_model_size(model_name: str) -> str:
    """
    Parse model size from HuggingFace model name.
    
    Args:
        model_name (str): Model name like "given131/8M_T1" or "facebook/esm2_t6_8M_UR50D"
    Returns:
        str: Model size key ("8M", "35M", "150M")
    """
    if "8M" in model_name:
        return "8M"
    elif "35M" in model_name:
        return "35M"
    elif "150M" in model_name:
        return "150M"
    else:
        raise ValueError(f"Cannot determine model size from: {model_name}")

def load_esm2_model(ckpt_path: str, custom_weights_path: Optional[str] = None, 
                    use_flash_attn: bool = True) -> tuple:
    """
    Load ESM2 model using HuggingFace transformers.
    
    Args:
        ckpt_path (str): HuggingFace model name (e.g., "given131/8M_T1" or "facebook/esm2_t6_8M_UR50D")
        custom_weights_path (str, optional): Path to custom weights file (.pt or .pth)
        use_flash_attn (bool): Use Flash Attention 2 if available (requires flash-attn package)
    Returns:
        model: HuggingFace EsmForMaskedLM model
        tokenizer: HuggingFace ESM2 tokenizer
    """
    # Check Flash Attention 2 availability
    attn_implementation = None
    if use_flash_attn:
        try:
            import flash_attn
            attn_implementation = "flash_attention_2"
            print(f"Flash Attention 2 available (version {flash_attn.__version__})")
        except ImportError:
            print("Flash Attention 2 not installed. Install with: pip install flash-attn --no-build-isolation")
            print("Falling back to SDPA (still fast on modern GPUs)")
            attn_implementation = "sdpa"  # Use PyTorch's native SDPA as fallback
    
    # Determine the base Facebook model architecture
    if ckpt_path.startswith("given131/"):
        # Parse model size and get corresponding Facebook config
        model_size = parse_model_size(ckpt_path)
        facebook_model = FACEBOOK_CONFIG[model_size]
        print(f"Using custom model {ckpt_path} with architecture from {facebook_model}")
        
        # Initialize tokenizer and model from Facebook architecture
        tokenizer = AutoTokenizer.from_pretrained(facebook_model)
        
        # Load model with attention implementation
        model_kwargs = {}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        
        try:
            model = EsmForMaskedLM.from_pretrained(facebook_model, **model_kwargs)
            if attn_implementation:
                print(f"Model loaded with {attn_implementation} attention")
        except Exception as e:
            print(f"Failed to load with {attn_implementation}: {e}")
            print("Falling back to default attention")
            model = EsmForMaskedLM.from_pretrained(facebook_model)
        
        # Load custom weights from HuggingFace model
        try:
            print(f"Loading custom weights from HuggingFace model: {ckpt_path}")
            from huggingface_hub import hf_hub_download
            
            # Download model.bin file
            model_bin_path = hf_hub_download(repo_id=ckpt_path, filename="model.bin")
            custom_state_dict = torch.load(model_bin_path, map_location='cpu')
            
            # Extract model_state_dict from the ordered_dict
            if 'model_state_dict' in custom_state_dict:
                model_weights = custom_state_dict['model_state_dict']
                print("Found 'model_state_dict' in checkpoint")
            else:
                print("Warning: 'model_state_dict' not found, using full state dict")
                model_weights = custom_state_dict
            
            # Load the state dict (strict=False to allow partial loading)
            missing_keys, unexpected_keys = model.load_state_dict(model_weights, strict=False)
            
            if missing_keys:
                print(f"Missing keys when loading custom weights: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"Unexpected keys when loading custom weights: {unexpected_keys[:5]}...")  # Show first 5
                
            print("Custom weights loaded successfully from HuggingFace")
        except (ImportError, OSError, RuntimeError) as e:
            print(f"Warning: Failed to load custom weights from HuggingFace: {e}")
            print("Continuing with pretrained weights...")
    
    else:
        # Standard Facebook model
        print(f"Using standard Facebook model: {ckpt_path}")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        
        # Load model with attention implementation
        model_kwargs = {}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        
        try:
            model = EsmForMaskedLM.from_pretrained(ckpt_path, **model_kwargs)
            if attn_implementation:
                print(f"Model loaded with {attn_implementation} attention")
        except Exception as e:
            print(f"Failed to load with {attn_implementation}: {e}")
            print("Falling back to default attention")
            model = EsmForMaskedLM.from_pretrained(ckpt_path)
        
        # Load additional custom weights if provided
        if custom_weights_path:
            print(f"Loading additional custom weights from: {custom_weights_path}")
            try:
                custom_state_dict = torch.load(custom_weights_path, map_location='cpu')
                
                # Handle different state dict formats
                if 'model_state_dict' in custom_state_dict:
                    custom_state_dict = custom_state_dict['model_state_dict']
                elif 'model' in custom_state_dict:
                    custom_state_dict = custom_state_dict['model']
                elif 'state_dict' in custom_state_dict:
                    custom_state_dict = custom_state_dict['state_dict']
                
                # Load the state dict (strict=False to allow partial loading)
                missing_keys, unexpected_keys = model.load_state_dict(custom_state_dict, strict=False)
                
                if missing_keys:
                    print(f"Missing keys when loading custom weights: {missing_keys[:5]}...")  # Show first 5
                if unexpected_keys:
                    print(f"Unexpected keys when loading custom weights: {unexpected_keys[:5]}...")  # Show first 5
                    
                print("Additional custom weights loaded successfully")
            except (ImportError, OSError, RuntimeError) as e:
                print(f"Warning: Failed to load additional custom weights: {e}")
                print("Continuing with model weights...")
    
    model.eval()
    return model, tokenizer

