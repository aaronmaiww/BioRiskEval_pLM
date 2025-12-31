"""
Common utility functions for BioRiskEval evaluation scripts.
"""


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

