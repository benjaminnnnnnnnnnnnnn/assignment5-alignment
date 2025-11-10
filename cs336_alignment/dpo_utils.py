import torch
import torch.nn.functional as F
import numpy as np

from typing import Any, Callable, Literal
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from cs336_alignment.sft_utils import tokenize_prompt_and_output, get_response_log_probs, masked_normalize


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    # raise NotImplementedError
    chosen_sequence   = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n{response_chosen}"""
    rejected_sequence = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n{response_rejected}"""
    chosen_sequence   += tokenizer.eos_token    
    rejected_sequence += tokenizer.eos_token

    # 对格式化后的序列进行编码
    chosen_input_ids = tokenizer.encode(chosen_sequence, return_tensors="pt")
    rejected_input_ids = tokenizer.encode(rejected_sequence, return_tensors="pt")

    # 计算训练模型的 log-probabilities
    with torch.no_grad():
        chosen_logits = lm(chosen_input_ids).logits
        rejected_logits = lm(rejected_input_ids).logits

    # 计算参考模型的 log-probabilities

    chosen_ref_logits = lm_ref(chosen_input_ids).logits
    rejected_ref_logits = lm_ref(rejected_input_ids).logits

    # 计算 log-probability 差值
    chosen_log_probs = torch.log_softmax(chosen_logits, dim=-1)
    rejected_log_probs = torch.log_softmax(rejected_logits, dim=-1)
    chosen_ref_log_probs = torch.log_softmax(chosen_ref_logits, dim=-1)
    rejected_ref_log_probs = torch.log_softmax(rejected_ref_logits, dim=-1)
    
    """
    (Pdb) chosen_log_probs.shape
    torch.Size([1, 48, 50257])
    (Pdb) chosen_input_ids.shape
    torch.Size([1, 48])
    """
    chosen_log_prob = chosen_log_probs[:, :-1, :].gather(-1, chosen_input_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)
    rejected_log_prob = rejected_log_probs[:, :-1, :].gather(-1, rejected_input_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)
    chosen_ref_log_prob = chosen_ref_log_probs[:, :-1, :].gather(-1, chosen_input_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)
    rejected_ref_log_prob = rejected_ref_log_probs[:, :-1, :].gather(-1, rejected_input_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)

    # 计算 DPO 损失
    chosen_ratios, rejected_ratios = chosen_log_prob - chosen_ref_log_prob, rejected_log_prob - rejected_ref_log_prob
    
    dpo_loss = -F.logsigmoid(beta * (chosen_ratios - rejected_ratios) )

    return dpo_loss