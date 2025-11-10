import torch
import torch.nn.functional as F
import numpy as np

from typing import Any, Callable, Literal
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def compute_group_normalized_rewards( 
    reward_fn, 
    rollout_responses,
    repeated_ground_truths,
    group_size,
    normalize_by_std, 
    advantage_eps,
):
    """
    Compute rewards for each group of rollout responses, normalized by the group size.
    Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against the ground truths, 
            producing a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str] Rollouts from the policy. The length of this list is 
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str] The ground truths for the examples. 
            The length of this list is rollout_batch_size, because the ground truth for each example is repeated group_size times.
        group_size: int Number of responses per question (group).
        advantage_eps: float Small constant to avoid division by zero in normalization.
        normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise subtract only the group mean.
    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
            advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout response.
            raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout response.
            metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    assert len(rollout_responses) == len(repeated_ground_truths)
    
    raw_rewards = []
    for rollout_response, gt_response in zip(rollout_responses, repeated_ground_truths):
        curr_reward = reward_fn(rollout_response, gt_response)["reward"]
        raw_rewards.append(curr_reward)

    raw_rewards = torch.tensor(raw_rewards)  # prompts * group_size, 1
    rewards_per_group = raw_rewards.reshape((-1, group_size))  # prompts * group_size
    mean_reward_per_group = torch.mean(rewards_per_group, dim=-1, keepdim=True)  # prompts * 1
    advantage = rewards_per_group - mean_reward_per_group  # prompts * group_size

    if normalize_by_std:
        std_reward_per_group = torch.std(rewards_per_group, dim=-1, keepdim=True)
        advantage /= std_reward_per_group + advantage_eps  # prompts
    advantage = advantage.flatten()  # prompts * group_size, 1

    metadata = {
        "mean": torch.mean(raw_rewards).item(),
        "std": torch.std(raw_rewards).item(),
        "max": torch.max(raw_rewards).item(),
        "min": torch.min(raw_rewards).item(),
    }

    return advantage, raw_rewards, metadata
    
    
def compute_naive_policy_gradient_loss( 
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages 
    is either the raw reward or an already-normalized advantage.
    
    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for each token.
    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss 
        (to be aggregated across the batch and sequence dimensions in the training loop).
    
    Implementation tips:
        • Broadcast the raw_rewards_or_advantages over the sequence_length dimension.
    """
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
            probs from the policy being trained.
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
            from the old policy.
        cliprange: float Clip parameter ϵ (e.g. 0.2).
    Returns: tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped loss.
        metadata dict containing whatever you want to log. We suggest logging whether each
            token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
            the min was lower than the LHS.
        Implementation tips:
        • Broadcast advantages over sequence_length    
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    smooth_adv = ratio * advantages
    clip_smooth_adv = torch.clip(ratio, 1 - cliprange, 1 + cliprange) * advantages
    loss = -torch.minimum(smooth_adv, clip_smooth_adv)
    
    clip_flag = (ratio < 1 - cliprange) | (ratio > 1 + cliprange)
    rhs_min_flag = clip_smooth_adv < smooth_adv
    metadata = dict(
        clip_flag = clip_flag,
        rhs_min_flag = rhs_min_flag,
    )
    
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.
    
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the policy being trained.
        loss_type One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages Required for "reinforce_with_baseline" and "grpo_clip"; shape (batch_size, 1).
        old_log_probs Required for "grpo_clip"; shape (batch_size, sequence_length).
        cliprange Required for "grpo_clip"; scalar ϵ used for clipping.
    Returns: tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss (batch_size, sequence_length), per-token loss.
        metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
        
    Implementation tips:
        • Delegate to compute_naive_policy_gradient_loss or compute_grpo_clip_loss.
        • Perform argument checks (see assertion pattern above).
        • Aggregate any returned metadata into a single dict.   
    """
    assert loss_type in ["no_baseline", "reinforce_with_baseline", "grpo_clip"]
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards is required for no_baseline loss"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )
        metadata = {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages is required for reinforce_with_baseline loss"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
        metadata = {}
    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages is required for grpo_clip loss"
        assert old_log_probs is not None, "old_log_probs is required for grpo_clip loss"
        assert cliprange is not None, "cliprange is required for grpo_clip loss"
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
        
    return loss, metadata


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where mask == 1.
    
    Args:
        tensor: torch.Tensor The data to be averaged.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
        dim: int | None Dimension over which to average. If None, compute the mean over all masked elements.
    
    Returns:
        torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics    
    """
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)
    

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[float, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the policy being trained.
        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps Number of microbatches per optimizer step.
        loss_type One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
        raw_rewards Needed when loss_type == "no_baseline"; shape (batch_size, 1).
        advantages Needed when loss_type != "no_baseline"; shape (batch_size, 1).
        old_log_probs Required for GRPO-Clip; shape (batch_size, sequence_length).
        cliprange Clip parameter ϵ for GRPO-Clip.
    
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return this so we can log it.
            metadata Dict with metadata from the underlying loss call, and any other statistics you might want to log.
        Implementation tips:
        • You should call loss.backward() in this function. Make sure to adjust for gradient
        accumulation
    """
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    loss = masked_mean(loss, response_mask)
    loss = loss / gradient_accumulation_steps
    
    # Backpropagate the loss
    loss.backward()

    return loss, metadata