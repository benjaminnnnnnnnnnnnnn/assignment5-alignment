import logging
import math
import os
import time
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Callable, Literal, List
from tqdm import tqdm

import dotenv
import pdb
import torch
import torch.nn as nn
import wandb

from unittest.mock import patch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.dynamic_module_utils import get_imports
from vllm import LLM, SamplingParams

from cs336_alignment.data_utils import extract_reference_answer, load_and_format_prompts
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.evaluate import get_response, get_all_response
from cs336_alignment.sft_utils import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)
from cs336_alignment.grpo_utils import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,   
)
from cs336_alignment.utils import (
    get_run_name,
    print_rich_dict,
    save_model_and_tokenizer,
)
from cs336_alignment.vllm_utils import init_vllm, load_model_into_vllm_instance

logging.getLogger("vllm").setLevel(logging.WARNING)


@dataclass
class TrainConfig:
    experiment_name_base: str = "experiments"
    experiment_name: str = "GRPO-qwen2.5-gsm8k-grpo_clip"
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    data_path: str = "./data/gsm8k/train.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"

    mixed_precision_training: bool = True
    learning_rate: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.98)
    train_device: str = "cuda:2"

    # # Log print:
    # log_print_steps = 16

    # For evaluation
    eval_device: str = "cuda:2"
    eval_interval_steps: int = 5
    
    # random seed
    seed = 42
    
    # GRPO
    n_grpo_steps: int = 40
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 1024
    group_size: int = 8
    epochs_per_rollout_batch: int = 1  # On-policy
    train_batch_size: int = 256  # On-policy
    gradient_accumulation_steps: int = 64  # microbatch size is 2, will fit on H100
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = "grpo_clip"
    use_std_normalization: bool = True
    cliprange: float = 0.2
    
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    
    
@dataclass
class EvaluateConfigDebug:
    data_path: str = "./data/gsm8k/test_samples.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 512
    min_tokens: int = 4


@dataclass
class EvaluateConfig:
    data_path: str = "./data/gsm8k/test.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1024
    min_tokens: int = 4
    
    
def sft_collate_fn(batch, tokenizer):
    prompts, cot, answers = zip(*batch)  # each is a tuple of strings
    prompts = list(prompts)
    cot = list(cot)

    batch_enc = tokenize_prompt_and_output(prompts, cot, tokenizer)

    return {**batch_enc, "answers": answers}


def grpo_collate_fn(batch, tokenizer):
    prompts, cot, adv, raw_reward, old_policy = zip(*batch)  # each is a tuple of strings
    prompts = list(prompts)
    cot = list(cot)
    old_policy = list(old_policy)

    batch_enc = tokenize_prompt_and_output(prompts, cot, tokenizer)
    adv = torch.stack(adv)[..., None]
    raw_reward = torch.stack(raw_reward)[..., None]
    
    maxLen = batch_enc["response_mask"].shape[1]
    for i, p in enumerate(old_policy):
        if p.shape[1] < maxLen:
            # pad to maxLen
            pad_len = maxLen - p.shape[1]
            pad_tensor = torch.zeros((p.shape[0], pad_len), dtype=p.dtype)
            old_policy[i] = torch.cat([p, pad_tensor], dim=1)
    old_policy = torch.cat(old_policy, dim=0)  # (B, S)

    return {
        **batch_enc, 
        "adv": adv, "raw_reward": raw_reward, 
        "old_policy": old_policy
    }


def to_float(x):
    if isinstance(x, torch.Tensor):
        return x.float().item()
    elif isinstance(x, str):
        return float(x.strip())

    return float(x)


class SFTDataset(Dataset):
    def __init__(self, train_prompts, train_cot, train_answers):
        self.train_prompts = train_prompts
        self.train_cot = train_cot
        self.train_answers = train_answers

    def __len__(self):
        return len(self.train_prompts)

    def __getitem__(self, idx: int) -> tuple[str, str, float]:
        prompt = self.train_prompts[idx]
        cot = self.train_cot[idx]
        answer = to_float(self.train_answers[idx])

        return prompt, cot, answer


class GrpoDataset(Dataset):
    def __init__(
        self, 
        train_prompts, train_cot, train_answers, 
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        advantages: torch.Tensor, 
        raw_rewards: torch.Tensor,
        train_config: TrainConfig,
    ):
        self.train_prompts = train_prompts
        self.train_cot = train_cot
        self.train_answers = train_answers      
        self.advantages = advantages  # (N, )
        self.raw_rewards = raw_rewards  # (N, )          

        # generate old policy log probs for grpo_clip
        self.old_policy_log_probs = []
        dataset = SFTDataset(train_prompts, train_cot, train_answers)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,  # 在 macOS 下避免 multiprocessing/pickle 问题
            pin_memory=True,
            collate_fn=lambda batch: sft_collate_fn(batch, tokenizer),
        )
        with torch.no_grad():
            for data in tqdm(dataloader, desc="Generating old policy log probs"):
                input_ids = data["input_ids"].to(train_config.eval_device)
                labels = data["labels"].to(train_config.eval_device)
                
                log_prob = get_response_log_probs(model=model, input_ids=input_ids, labels=labels)
                log_prob = log_prob["log_probs"].cpu()
                self.old_policy_log_probs.append(log_prob) 

    def __len__(self):
        return len(self.train_prompts)

    def __getitem__(self, idx: int) -> tuple[str, str, float]:
        prompt = self.train_prompts[idx]
        cot = self.train_cot[idx]
        adv = self.advantages[idx]
        raw_reward = self.raw_rewards[idx]
        old_policy = self.old_policy_log_probs[idx]

        return prompt, cot, adv, raw_reward, old_policy


def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    if it >= max_steps:
        return min_lr
    # pure cosine decay
    decay_ratio = it / max_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
):
    responses = get_response(vllm_model, prompts, eval_sampling_params)
    allinfo_dict_list = []
    for response, answer, prompt in zip(responses, answers, prompts):
        # extracted_answer = extract_reference_answer(response)
        reward_dict = reward_fn(response, answer)
        allinfo_dict_list.append(reward_dict)

    overview = {"correct": 0, "format_wrong": 0, "answer_wrong": 0, "count": 0}
    for reward in allinfo_dict_list:
        overview["count"] += 1
        if reward["reward"] == 1:
            overview["correct"] += 1
        elif reward["format_reward"] == 1:
            overview["answer_wrong"] += 1
        else:
            overview["format_wrong"] += 1

    return overview


def evaluate_sft_model(config: EvaluateConfig, vllm: LLM, eval_step: int):
    prompts, cot, answers = load_and_format_prompts(config.data_path, config.prompt_path)

    sp = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        min_tokens=config.min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    results = evaluate_vllm(vllm, r1_zero_reward_fn, prompts, answers, sp)

    wandb.log(
        {
            "eval/count": results["count"],
            "eval/correct": results["correct"],
            "eval/answer_wrong": results["answer_wrong"],
            "eval/format_wrong": results["format_wrong"],
            "eval_step": eval_step,
        }
    )


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def main(debug_mode):
    dotenv.load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if debug_mode:
        train_config = TrainConfigDebug()
        eval_config = EvaluateConfigDebug()
    else:
        train_config = TrainConfig()
        eval_config = EvaluateConfig()

    # Login Wandb
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project="cs336-alignment-grpo",
        config={
            "train": asdict(train_config),
            "eval": asdict(eval_config),
        },
        name=train_config.experiment_name + f"-{time.strftime("%m%d-%H%M%S")}",
        reinit=True,
        mode="online",
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    vllm = init_vllm(
        model_id=train_config.model_name, 
        device=train_config.eval_device, 
        seed=train_config.seed, 
        gpu_memory_utilization = 0.5,
    )

    all_prompts, all_cot, all_answers = load_and_format_prompts(train_config.data_path, train_config.prompt_path)

    # ---------------------
    # Load Model and tokenizer
    # ---------------------
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=train_config.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        ).to(train_config.train_device)
        tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    else:
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=train_config.model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
            )
            tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    print(f"[train] Tokenizer {train_config.model_name} loaded")
    print(f"[train] Model {train_config.model_name} loaded on {train_config.train_device}")
    
    # init vllm to collect samples
    load_model_into_vllm_instance(model, vllm)
    grpo_sp = SamplingParams(
        temperature=eval_config.temperature,
        top_p=eval_config.top_p,
        max_tokens=eval_config.max_tokens,
        min_tokens=eval_config.min_tokens,
        n=train_config.group_size,  # rollout counts
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    
    global_step = 0
    for gi in range(train_config.n_grpo_steps):
        print("\n====================================================================")
        print(f"======================= GRPO Step: {gi} =======================")
        print("====================================================================\n")
        # 随机选取部分数据进行筛选
        random_indices = random.sample(range(len(all_prompts)), k=train_config.n_prompts_per_rollout_batch)
        sampled_prompts = [all_prompts[i] for i in random_indices]
        sampled_cot = [all_cot[i] for i in random_indices]
        sampled_answers = [all_answers[i] for i in random_indices]

        # 采样回答和计算reward
        print("Sampling responses and compute rewards...")
        responses = get_all_response(
            vllm_model=vllm,
            prompts=sampled_prompts,
            sampling_params=grpo_sp,
        )
        print("Sampling completed.")

        repeated_sampled_prompts, repeated_sampled_cot, repeated_sampled_answers = [], [], []
        for prompt, cot, ans in zip(sampled_prompts, sampled_cot, sampled_answers):
            repeated_sampled_prompts.extend([prompt] * train_config.group_size)
            repeated_sampled_cot.extend([cot] * train_config.group_size)
            repeated_sampled_answers.extend([ans] * train_config.group_size)
        advantages, raw_rewards, raw_rewards_stats = compute_group_normalized_rewards( 
            reward_fn=r1_zero_reward_fn,
            rollout_responses=responses,
            repeated_ground_truths=repeated_sampled_answers,
            group_size=train_config.group_size,
            normalize_by_std=train_config.use_std_normalization, 
            advantage_eps=train_config.advantage_eps,
        )

        # Data Preparation
        # ---------------------
        dataset = GrpoDataset(
            repeated_sampled_prompts, responses, repeated_sampled_answers, 
            tokenizer, model, 
            advantages, raw_rewards,
            train_config
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=train_config.micro_train_batch_size,
            shuffle=True,
            num_workers=0 if debug_mode else 8,  # 在 macOS 下避免 multiprocessing/pickle 问题
            pin_memory=True,
            collate_fn=lambda batch: grpo_collate_fn(batch, tokenizer),
        )
        
        # ---------------------
        # Mixed Precision Context
        # ---------------------
        ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if train_config.mixed_precision_training
            else nullcontext()
        )
        
        # ---------------------
        # Optimizer
        # ---------------------
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
        print("[train] Optimizer initialized")
        
        # ---------------------
        # Training Process
        # ---------------------
        batch_loss, cur_step = 0, 0
        n_steps_per_rollout = math.floor(train_config.epochs_per_rollout_batch
            * len(responses) / train_config.train_batch_size)
        for _epoch in range(train_config.epochs_per_rollout_batch):
            for _i, data in enumerate(dataloader):
                global_step += 1
                input_ids = data["input_ids"].to(train_config.train_device)
                labels = data["labels"].to(train_config.train_device)
                response_mask = data["response_mask"].to(train_config.train_device)
                cur_advantages = data["adv"].to(train_config.train_device)
                cur_raw_rewards = data["raw_reward"].to(train_config.train_device)
                old_policy = data["old_policy"].to(train_config.train_device)

                with ctx:
                    log_prob = get_response_log_probs(model=model, input_ids=input_ids, labels=labels)
                    log_prob = log_prob["log_probs"]
                    loss, metadata = grpo_microbatch_train_step(
                        policy_log_probs = log_prob,  # (2, 401)
                        response_mask = response_mask,  # (2, 401)
                        gradient_accumulation_steps = train_config.gradient_accumulation_steps,
                        loss_type = train_config.loss_type,
                        raw_rewards = cur_raw_rewards,  # (16, )
                        advantages = cur_advantages,
                        old_log_probs = old_policy,
                        cliprange = train_config.cliprange,
                    )
                    
                    if train_config.loss_type == "grpo_clip":
                        clip_flag = metadata["clip_flag"]
                        wandb.log(
                            {
                                "train/clip_ratio": clip_flag.sum().item() / clip_flag.numel(),
                                "train_step": global_step,
                            }
                        )

                batch_loss += loss
                if global_step % train_config.gradient_accumulation_steps == 0:
                    cur_step += 1
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    adj_lr = get_lr(cur_step, train_config.learning_rate, n_steps_per_rollout)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = adj_lr

                    optimizer.step()
                    optimizer.zero_grad()

                    print(
                        f"[train] Step {cur_step} | Loss: {batch_loss / train_config.gradient_accumulation_steps:.4f} | LR: {adj_lr:.7f}"
                    )

                    wandb.log(
                        {
                            "train/loss": batch_loss / train_config.gradient_accumulation_steps,
                            "train/lr": adj_lr,
                            "train_step": global_step,
                        }
                    )

                    batch_loss = 0
        
        if (gi + 1) % train_config.eval_interval_steps == 0:
            # Run evaluatoin
            print(f"\n[eval] at step {gi}")
            load_model_into_vllm_instance(model, vllm)
            evaluate_sft_model(eval_config, vllm, eval_step=gi+1)
            print(f"[eval] Evaluation completed for step {gi}")
            
            # save model
            save_model_and_tokenizer(model, tokenizer, train_config)
        
    wandb.finish()


if __name__ == "__main__":
    main(debug_mode=False)

