import json

from tqdm import tqdm
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

if __name__ == "__main__":
    # generate prompts.    
    eval_num = 10
    testset = {
        # GSM8K
        "gsm8k": "/Users/benjamin/Documents/stanford-cs336/assignment5-alignment/data/gsm8k/test.jsonl",
        # MATH
        "algebra": "/Users/benjamin/Documents/stanford-cs336/assignment5-alignment/data/MATH/test-algebra.jsonl",
        "count": "/Users/benjamin/Documents/stanford-cs336/assignment5-alignment/data/MATH/test-count.jsonl",
    }
    r1_zero_path = "/Users/benjamin/Documents/stanford-cs336/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    prompts, answers = [], []
    
    # generate prompts
    with open(r1_zero_path, 'r') as fin:
        system_prompt = fin.read()
    with open(testset["algebra"], 'r') as fin:
        lines = fin.readlines()
    for line in tqdm(lines):
        qa = json.loads(line[:-1])
        question = qa["question"] if "question" in qa else qa["problem"]
        prompt = system_prompt.replace("{question}", question)
        prompts.append(prompt)
        
        answer = qa["answer"] if "answer" in qa else qa["solution"]
        answers.append(answer)
    
    
    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams( 
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    
    # Create an LLM.
    llm = LLM(model="/Users/benjamin/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-1.5B-Instruct/")
    # llm = LLM(
    #     model="/Users/benjamin/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B/",
    #     max_model_len=4096,              # 降低上下文长度
    #     max_num_batched_tokens=4096       # 或者调高到 131072+
    # )

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information. 
    outputs = llm.generate(prompts[:eval_num], sampling_params)
    
    format_reward_cnt = 0
    answer_reward_cnt = 0
    reward_cnt = 0
    for output, answer in zip(outputs, answers):
        prompt = output.prompt
        generated_text = output.outputs[0].text 
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        
        # eval
        """
        return {
            "format_reward": 1.0,
            "answer_reward": 1.0,
            "reward": 1.0
        }
        """
        res = r1_zero_reward_fn(response=generated_text, ground_truth=answer, fast=True)
        format_reward_cnt += res["format_reward"]
        answer_reward_cnt += res["answer_reward"]
        reward_cnt += res["reward"]

    print(f"format acc: {(format_reward_cnt / len(outputs)):.2f}")
    print(f"answer acc: {(answer_reward_cnt / len(outputs)):.2f}")
    