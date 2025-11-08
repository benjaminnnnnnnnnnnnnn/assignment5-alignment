import json
import torch
import logging
from pathlib import Path
from typing import Callable, List, Union

import fire
from vllm import LLM, SamplingParams

from cs336_alignment.data_utils import extract_reference_answer, load_and_format_prompts, parse_response_gsm8k, parse_response_mmlu
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from cs336_alignment.utils import print_color, safe_slug

logging.getLogger("vllm").setLevel(logging.WARNING)


def get_response(vllm_model, prompts, sampling_params) -> List[str]:
    result = vllm_model.generate(prompts, sampling_params)
    outputs = [output.outputs[0].text.strip() for output in result]
    return outputs


def get_all_response(vllm_model, prompts, sampling_params) -> List[str]:
    result = vllm_model.generate(prompts, sampling_params)
    outputs = []
    for output in result:
        for out in output.outputs:
            outputs.append(out.text.strip())
    return outputs


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    cot: List[str],
    true_answers: List[str],
    eval_sampling_params: SamplingParams,
    dataset: str,
):
    assert dataset in ["math", "mmlu", "gsm8k", "alpaca", "safety"], "Unsupported dataset"

    responses = get_response(vllm_model, prompts, eval_sampling_params)
    allinfo_dict_list = []
    for response, true_answer, prompt in zip(responses, true_answers, prompts):
        if dataset == "mmlu":
            extracted_answer = parse_response_mmlu(response)
        elif dataset == "gsm8k":
            extracted_answer = parse_response_gsm8k(response)
        else:
            extracted_answer = extract_reference_answer(response)   
        reward_dict = reward_fn(response, true_answer)

        info_dict: dict[str, Union[str, float]] = {
            "prompt": prompt,
            "response": response,
            "true_answer": true_answer,
            "extracted_answer": extracted_answer,
            **reward_dict,
        }

        allinfo_dict_list.append(info_dict)

    return allinfo_dict_list


def dump_vllm_outputs(
    vllm_model: LLM,
    data_path: str,
    eval_sampling_params: SamplingParams,
    dataset: str,
    save_path: str,
):
    assert dataset in ["math", "mmlu", "gsm8k", "alpaca", "safety"], "Unsupported dataset"

    datasets, prompts = [], []

    if dataset == "alpaca":
        with open(data_path, "r") as file:
            for line in file:
                data = json.loads(line)
                datasets.append(data)
                prompts.append(data["instruction"])
    
        responses = get_response(vllm_model, prompts, eval_sampling_params)
        for response, data in zip(reponses, datasets):
            data["output"] = response
            data["generator"] = "llama-3.1-8b-base"

    elif dataset == "safety":
        with open(data_path, "r") as file:
            for line in file[1:]:
                prompts.append(line.split(',')[-1])
    
        responses = get_response(vllm_model, prompts, eval_sampling_params)
        
        for prompt, response in zip(prompts, reponses):
            data = {
                "prompts_final": prompt,
                "output": response,
            }
            datasets.append(data)
    
    with open(save_path, 'w') as fout:
        json.dump(datasets, fout)
        

def main(
    dataset: str = "gsm8k",
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    data_path: str = "./data/gsm8k/test_samples.jsonl",
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
):
    assert dataset in ["math", "mmlu", "gsm8k", "alpaca", "safety"], "Unsupported dataset"
    
    if dataset == "gsm8k":
        model_name = "Llama/Llama-3.1-8B"
        data_path = "./data/gsm8k/test.jsonl"
        prompt_path = "cs336_alignment/prompts/zero_shot_system_prompt.prompt"
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["<|end_of_text|>", "```"],
            include_stop_str_in_output=True,
        )
        reward_func = lambda response, ground_truth: question_only_reward_fn(response, ground_truth, dataset=dataset)
        
    elif dataset == "mmlu":
        model_name = "Llama/Llama-3.1-8B"
        data_path = "data/mmlu/dev/high_school_world_history_dev.csv"
        prompt_path = "cs336_alignment/prompts/mmlu.prompt"
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["<|end_of_text|>", "```"],
            include_stop_str_in_output=True,
        )
        reward_func = lambda response, ground_truth: question_only_reward_fn(response, ground_truth, dataset=dataset)

    elif dataset == "alpaca":
        model_name = "Llama/Llama-3.1-8B"
        data_path = "assignment5-alignment/data/alpaca_eval/alpaca_eval.jsonl"
        prompt_path = "cs336_alignment/prompts/question_only.prompt"
        save_path = "assignment5-alignment/data/alpaca_eval/alpaca_eval_llama_8B.jsonl"

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["<|end_of_text|>", "```"],
            include_stop_str_in_output=True,
        )

    elif dataset == "safety":
        model_name = "Llama/Llama-3.1-8B"
        data_path = "assignment5-alignment/data/simple_safety_tests/simple_safety_tests.csv"
        prompt_path = "cs336_alignment/prompts/question_only.prompt"
        save_path = "assignment5-alignment/data/simple_safety_tests/safety_tests_llama_8B.csv"

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["<|end_of_text|>", "```"],
            include_stop_str_in_output=True,
        )
        
    else:  # MATH dataset
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
        reward_func = r1_zero_reward_fn
        
    print_color(f"Evaluating {model_name} on {data_path}")
    
    vllm_model = LLM(
        model=model_name,
        device="cuda:0",
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.85,
    )

    if dataset in ["alpaca", "safety"]:
        dump_vllm_outputs(vllm_model, data_path, sampling_params, dataset=dataset, save_path=save_path)
        return

    prompts, cot, true_answers = load_and_format_prompts(data_path, prompt_path, dataset=dataset)

    results = evaluate_vllm(vllm_model, reward_func, prompts, cot, true_answers, sampling_params, dataset=dataset)

    # Save the results
    model_tag = safe_slug(model_name)
    data_stem = Path(data_path).stem
    out_dir = Path("evaluations")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"evaluate_{model_tag}_{data_stem}.jsonl"

    format_rewards = 0
    answer_reward = 0
    reward = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for i in results:
            format_rewards += i["format_reward"]
            answer_reward += i["answer_reward"]
            reward += i["reward"]
            json.dump(i, f)
            f.write("\n")

    print_color(f"Format rewards: {format_rewards}/{len(results)}", "green")
    print_color(f"Answer rewards: {answer_reward}/{len(results)}", "green")
    print_color(f"Total rewards: {reward}/{len(results)}", "green")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    fire.Fire(main)
