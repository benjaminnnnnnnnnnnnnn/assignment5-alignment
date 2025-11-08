import logging
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Callable, List

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
from cs336_alignment.utils import (
    get_run_name,
    print_rich_dict,
    save_model_and_tokenizer,
)
from cs336_alignment.vllm_utils import init_vllm, load_model_into_vllm_instance

logging.getLogger("vllm").setLevel(logging.WARNING)


@dataclass
class TrainConfigMac:
    experiment_name_base: str = "experiments"
    experiment_name: str = "sft-qwen2.5-gsm8k"
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    data_path: str = "./data/gsm8k/train.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"

    batch_size: int = 1
    gradient_accumulation_steps: int = 2  # default 16
    training_steps: int = 512
    mixed_precision_training: bool = False  # default True
    learning_rate: float = 5e-6
    betas: tuple[float, float] = (0.9, 0.98)
    train_device: str = 'cpu'  # "cuda:0"

    # Log print:
    log_print_steps = 12  # default 12

    # For evaluation
    eval_device: str = 'cpu'  # "cuda:1"
    eval_interval_steps: int = 16  # default 32
    
    # random seed
    seed = 42
    
    # Expert Iteration
    expert_iterations: int = 5
    rollout_counts = 4  # number of samples per prompt
    sft_epochs_per_ei = 1
    
    
@dataclass
class TrainConfigDebug:
    experiment_name_base: str = "experiments"
    experiment_name: str = "EI-qwen2.5-gsm8k-debug"
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    data_path: str = "./data/gsm8k/train.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"

    batch_size: int = 2
    gradient_accumulation_steps: int = 2  # default 16
    training_steps: int = 32
    mixed_precision_training: bool = False  # default True
    learning_rate: float = 5e-6
    betas: tuple[float, float] = (0.9, 0.98)
    train_device: str = "cuda:0"

    # Log print:
    log_print_steps = 4  # default 12

    # For evaluation
    eval_device: str = "cuda:0"
    eval_interval_steps: int = 16  # default 32
    
    # random seed
    seed = 42
    
    # Expert Iteration
    expert_iterations: int = 5
    rollout_counts = 4  # number of samples per prompt
    sft_epochs_per_ei = 2
    num_ei_samples = 100


@dataclass
class TrainConfig:
    experiment_name_base: str = "experiments"
    experiment_name: str = "EI-qwen2.5-gsm8k_num=2k_G=8_ep=2"
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    data_path: str = "./data/gsm8k/train.jsonl"
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt"

    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    training_steps: int = 512
    mixed_precision_training: bool = True
    learning_rate: float = 5e-6
    betas: tuple[float, float] = (0.9, 0.98)
    train_device: str = "cuda:0"

    # Log print:
    log_print_steps = 16

    # For evaluation
    eval_device: str = "cuda:1"
    eval_interval_steps: int = 64
    
    # random seed
    seed = 42
    
    # Expert Iteration
    expert_iterations: int = 5
    rollout_counts = 8  # number of samples per prompt
    sft_epochs_per_ei = 2
    num_ei_samples = 2000
    
    
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


def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    if it >= max_steps:
        return min_lr
    # pure cosine decay
    decay_ratio = it / max_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def log_generate(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    cot: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
    cur_step: int,
    num_example=2,
):
    random_indices = random.sample(range(len(prompts)), k=num_example)
    sampled_prompts = [prompts[i] for i in random_indices]
    sampled_cot = [cot[i] for i in random_indices]
    sampled_answers = [answers[i] for i in random_indices]

    responses = get_response(vllm_model, sampled_prompts, eval_sampling_params)

    for i in range(num_example):
        print(f"======= Step: {cur_step}; Example {i} =======")
        response = responses[i]
        answer = sampled_answers[i]
        prompt = sampled_prompts[i]
        extracted_answer = extract_reference_answer(response)
        true_label = sampled_cot[i]

        reward_dict = reward_fn(response, answer)

        info_dict = {
            "prompt": prompt,
            "true_label": true_label,
            "response": response,
            "true_answer": answer,
            "extracted_answer": extracted_answer,
            **reward_dict,
        }

        # print_formatted_dict(info_dict)
        print_rich_dict(info_dict)
        print("==============================================\n")


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


def evaluate_sft_model(config: EvaluateConfig, vllm: LLM, eval_step: int, global_pre_step: int):
    prompts, cot, answers = load_and_format_prompts(config.data_path, config.prompt_path)

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        min_tokens=config.min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    results = evaluate_vllm(vllm, r1_zero_reward_fn, prompts, answers, sampling_params)

    wandb.log(
        {
            "eval/count": results["count"],
            "eval/correct": results["correct"],
            "eval/answer_wrong": results["answer_wrong"],
            "eval/format_wrong": results["format_wrong"],
            "eval_step": eval_step + global_pre_step,
        }
    )


def train_sft_model(
    model,
    tokenizer,
    train_config: TrainConfig,
    eval_config: EvaluateConfig,
    train_prompts,
    train_cot,
    train_answers,
    vllm: LLM,
    evaluate: bool = True,
    global_pre_step: int = 0,
):
    # Data Preparation
    # ---------------------
    dataset = SFTDataset(train_prompts, train_cot, train_answers)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 0,  # 在 macOS 下避免 multiprocessing/pickle 问题
        pin_memory=True,
        collate_fn=lambda batch: sft_collate_fn(batch, tokenizer),
    )
    print(f"[train] Dataloader initialized with batch size {train_config.batch_size}")

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
    cur_step = 0  # Initialize current step
    batch_loss = 0
    total_micro_steps = 0
    while True:
        for i, data in enumerate(dataloader):
            total_micro_steps += 1
            input_ids = data["input_ids"].to(train_config.train_device)
            labels = data["labels"].to(train_config.train_device)
            response_mask = data["response_mask"].to(train_config.train_device)

            with ctx:
                log_prob = get_response_log_probs(model=model, input_ids=input_ids, labels=labels)
                log_prob = log_prob["log_probs"]
                loss, _ = sft_microbatch_train_step(
                    log_prob, response_mask, train_config.gradient_accumulation_steps
                )

            batch_loss += loss
            if total_micro_steps % train_config.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                adj_lr = get_lr(cur_step, train_config.learning_rate, train_config.training_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = adj_lr

                optimizer.step()
                optimizer.zero_grad()

                print(
                    f"[train] Step {cur_step} | Loss: {batch_loss / train_config.gradient_accumulation_steps:.4f} | LR: {adj_lr:.8f}"
                )

                wandb.log(
                    {
                        "train/loss": batch_loss / train_config.gradient_accumulation_steps,
                        "train/lr": adj_lr,
                        "train_step": cur_step + global_pre_step,
                    }
                )

                batch_loss = 0
                cur_step += 1

                if cur_step % train_config.log_print_steps == 0 and evaluate:
                    load_model_into_vllm_instance(model, vllm)
                    log_generate(
                        vllm,
                        reward_fn=r1_zero_reward_fn,
                        prompts=dataset.train_prompts,
                        cot=dataset.train_cot,
                        answers=dataset.train_answers,
                        eval_sampling_params=SamplingParams(
                            temperature=eval_config.temperature,
                            top_p=eval_config.top_p,
                            max_tokens=eval_config.max_tokens,
                            min_tokens=eval_config.min_tokens,
                            stop=["</answer>"],
                            include_stop_str_in_output=True,
                        ),
                        cur_step=cur_step,
                        num_example=3,
                    )

                if cur_step >= train_config.training_steps:
                    break

        if cur_step >= train_config.training_steps:
            break

    save_model_and_tokenizer(model, tokenizer, train_config)
    print(f"[train] Training finished at step {cur_step}")
    
    # Run evaluatoin
    print(f"\n[eval] at step {cur_step}")
    load_model_into_vllm_instance(model, vllm)
    evaluate_sft_model(eval_config, vllm, eval_step=cur_step, global_pre_step=global_pre_step)
    print(f"[eval] Evaluation completed for step {cur_step}")

    return model


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
        project="cs336-alignment-sft",
        config={
            "train": asdict(train_config),
            "eval": asdict(eval_config),
        },
        name=get_run_name("EI", train_config),
        reinit=True,
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    vllm = init_vllm(
        model_id=train_config.model_name, 
        device=train_config.eval_device, 
        seed=train_config.seed, 
        gpu_memory_utilization = 0.5 if debug_mode else 0.9,
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
    
    global_pre_step = 0
    for ei in range(train_config.expert_iterations):
        print("\n====================================================================")
        print(f"======================= Expert Iteration {ei} =======================")
        print("====================================================================\n")
        # 随机选取部分数据进行筛选
        train_prompts, train_cot, train_answers = [], [], []
        random_indices = random.sample(range(len(all_prompts)), k=train_config.num_ei_samples)
        sampled_prompts = [all_prompts[i] for i in random_indices]
        sampled_cot = [all_cot[i] for i in random_indices]
        sampled_answers = [all_answers[i] for i in random_indices]
        
        # 使用vllm生成rollout_counts个samples
        load_model_into_vllm_instance(model, vllm)
        eval_sampling_params = SamplingParams(
            temperature=eval_config.temperature,
            top_p=eval_config.top_p,
            max_tokens=eval_config.max_tokens,
            min_tokens=eval_config.min_tokens,
            n=train_config.rollout_counts,  # rollout counts
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
        print("[Expert Iteration] Generating responses for filtering...")
        responses = get_all_response(vllm, sampled_prompts, eval_sampling_params)
        
        # 根据reward筛选数据
        for di in range(train_config.num_ei_samples):
            for ri in range(train_config.rollout_counts):
                eval_dict = r1_zero_reward_fn(responses[di * train_config.rollout_counts + ri], sampled_answers[di])
                if eval_dict['reward'] >= 1.0:
                    train_prompts.append(sampled_prompts[di])
                    train_cot.append(responses[di * train_config.rollout_counts + ri])
                    train_answers.append(sampled_answers[di])

        # 动态调整train_config.training_steps，以控制sft训练epochs
        train_config.training_steps = len(train_prompts) * train_config.sft_epochs_per_ei // (train_config.batch_size * train_config.gradient_accumulation_steps) + 1
        print("ei/num_filtered_data:", len(train_prompts))
        print("ei/training_steps_per_ei:", train_config.training_steps)
        wandb.log({
            "ei/num_filtered_data": len(train_prompts),
            "ei/training_steps_per_ei": train_config.training_steps,
        })
        train_sft_model(
            model,
            tokenizer,
            train_config,
            eval_config=eval_config,
            train_prompts=train_prompts,
            train_cot=train_cot,
            train_answers=train_answers,
            vllm=vllm,
            global_pre_step=global_pre_step,
        )
        
        global_pre_step += train_config.training_steps

    wandb.finish()


if __name__ == "__main__":
    main(debug_mode=False)

