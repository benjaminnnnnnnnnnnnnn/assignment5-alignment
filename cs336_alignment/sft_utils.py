import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase
) -> dict[str, torch.Tensor]:
    # 151643 for Qwen/Qwen2.5-Math-1.5B
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100

    prompt_input_ids = []
    output_input_ids = []

    for prompt in prompt_strs:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_input_ids.append(torch.tensor(tokens))
    for output in output_strs:
        tokens = tokenizer.encode(output, add_special_tokens=False)
        output_input_ids.append(torch.tensor(tokens))

    seq_lengths = [len(p_ids) + len(o_ids) for p_ids, o_ids in zip(prompt_input_ids, output_input_ids)]
    # Pad to max length
    max_length = max(seq_lengths)

    concatenated_input_ids = []
    concatenated_labels = []
    response_masks = []

    for (
        p_ids,
        o_ids,
    ) in zip(prompt_input_ids, output_input_ids):
        input_ids = torch.cat([p_ids, o_ids], dim=0)
        pad_length = max_length - input_ids.shape[0]
        padded_input_ids = F.pad(input_ids, (0, pad_length), value=pad_token_id)

        response_mask = torch.cat([torch.zeros_like(p_ids).bool(), torch.ones_like(o_ids).bool()], dim=0)
        padded_response_mask = F.pad(response_mask, (0, pad_length), value=False)

        concatenated_input_ids.append(padded_input_ids[:-1])
        concatenated_labels.append(padded_input_ids[1:])
        response_masks.append(
            padded_response_mask[1:]
        )  ### set prompt and padding to be false, will not calculate loss

    input_ids_tensor = torch.stack(concatenated_input_ids)
    label_tensor = torch.stack(concatenated_labels)
    response_mask_tensor = torch.stack(response_masks)

    return {"input_ids": input_ids_tensor, "labels": label_tensor, "response_mask": response_mask_tensor}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # logits has shape: (B, S, V)
    log_probs = F.log_softmax(logits, dim=-1)  # log p(x)
    probs = torch.exp(log_probs)

    return -torch.sum(probs * log_probs, dim=-1)


def get_response_log_probs(
    model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor, return_token_entropy: bool = False
) -> dict[str, torch.Tensor]:
    """
    input:
        - model,
        - input_ids: (B, S)
        - labels: (B, S)
    output: log_softmax(B S) and entropy(B S)
    """

    # (B, S, V)
    logits = model(input_ids).logits

    # First way
    # log_probs = F.log_softmax(logits, dim=-1)
    # cond_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    # Second way
    # nll = F.cross_entropy(
    #     logits.view(-1, logits.size(-1)),  # (B*T, V)
    #     labels.view(-1),  # (B*T,)
    #     reduction="none",
    #     # ignore_index=-100,  # optional, if you mark pads as -100
    # ).view(labels.size())  # reshape back to (B, T)

    # cond_log_probs = -nll  # convert NLL to log probs

    # if return_token_entropy:
    #     token_entropy = compute_entropy(logits)
    #     return {"log_probs": cond_log_probs, "token_entropy": token_entropy}

    # return {"log_probs": cond_log_probs}
    log_prob = F.log_softmax(logits, dim=-1)  # B, S, V
    label_token_log_softmax = log_prob.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # B, S

    if return_token_entropy:
        entropy = compute_entropy(logits)
        return {"log_probs": label_token_log_softmax, "token_entropy": entropy}
    else:
        return {"log_probs": label_token_log_softmax}


def masked_normalize(
    tensor: torch.Tensor, mask: torch.Tensor, normalize_constant: float = 1.0, dim: int = -1
) -> torch.Tensor:
    assert normalize_constant != 0, "Normalization constant must not be zero"

    masked_tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
    return torch.sum(masked_tensor, dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Perform a training step on a microbatch of data.

    Args:
        policy_log_probs (torch.Tensor): (B, S)
        response_mask (torch.Tensor): (B, S)
        gradient_accumulation_steps (int): The number of gradient accumulation steps.
        normalize_constant (float, optional): The normalization constant. Defaults to 1.0.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: The loss and metadata.
    """

    assert gradient_accumulation_steps > 0, "Gradient accumulation steps must be positive"
    assert policy_log_probs.shape == response_mask.shape, (
        "policy_log_probs and response_mask must have same shape"
    )

    # Create loss
    # One thing to notice is that, the loss return by the masked_normalize function is
    # NOT averaged over the batch
    raw_loss = masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1)

    # average of the loss and scale by gradient accumulation steps
    loss = -raw_loss.mean()
    loss = loss / gradient_accumulation_steps

    # Backprop for this microbatch
    loss.backward()

    # Metadata for logging (detach to avoid holding graph)
    num_resp_tokens = response_mask.sum()  # The total number of tokens in the response

    metadata: dict[str, torch.Tensor] = {
        "loss": loss.detach(),
        "num_response_tokens": num_resp_tokens.detach(),
    }

    return loss, metadata
