import json
import regex as re


# --- Chain-of-thought to <think> <answer> conversion utility ---
def convert_cot_to_think_answer(text: str) -> str:
    """
    Convert a chain-of-thought style answer that ends with a line like
    "#### 5" into the desired format by replacing that trailer with
    " </think> <answer> 5 </answer>".

    Examples
    --------
    >>> s = (
    ...     "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\\n"
    ...     "Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\\n"
    ...     "This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\\n"
    ...     "#### 5"
    ... )
    >>> convert_cot_to_think_answer(s)
    "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. </think> <answer> 5 </answer>"

    If no trailing "#### <ans>" is found, this function will try to extract a
    terminal number at the end of the string and use that as the answer. If that
    also fails, the input text is returned unchanged.
    """
    # Match a final line that looks like: #### 5 (possibly with spaces/newline)
    m = re.search(r"####\s*([^\n]+)\s*$", text)
    if m:
        ans = m.group(1).strip()
        prefix = text[: m.start()].rstrip()
        return f"{prefix} </think> <answer>{ans}</answer>"

    # Fallback: try to capture a trailing number at end of text
    m_num = re.search(r"(-?\d+(?:\.\d+)?)\s*$", text)
    if m_num:
        ans = m_num.group(1)
        prefix = text[: m_num.start()].rstrip()
        return f"{prefix} </think> <answer>{ans}</answer>"

    return text


def extract_reference_answer(response: str) -> str:
    from cs336_alignment.drgrpo_grader import extract_answer

    model_answer = response.split("<answer>")[-1].replace("</answer>", "")
    if "\\boxed" in model_answer:
        model_answer = extract_answer(model_answer)

    return model_answer


def extract_gsm8k_answer(answer: str) -> str:
    ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"


def wrap_prompt(text: str, prompt_path: str):
    with open(prompt_path, "r") as file:
        prompt = file.read()
    return prompt.format(question=text)


def load_and_format_prompts(data_path: str, prompt_path: str, dataset: str = "gsm8k") -> tuple[list[str], list[str], list[str]]:
    assert dataset in ["math", "mmlu", "gsm8k", "alpaca", "safety"], "Unsupported dataset"
    with open(prompt_path, "r") as file:
        prompt = file.read()

    prompts = []
    cot = []
    answers = []
    with open(data_path, "r") as file:
        if dataset == "mmlu":
            mmlu_format_data = {
                "question": "",
                "answer": "",
            }
            opt_re = re.compile(r",[A-D]\s")
            subject = data_path.split('/')[-1].replace('.csv', '')

        for line in file:
            if dataset == "mmlu":
                matches = opt_re.findall(line)
                if not matches:
                    mmlu_format_data["question"] += line.strip()
                else:
                    splits = line.split(',')
                    question_tail = ",".join(splits[:-5])
                    mmlu_format_data["question"] += question_tail
                    mmlu_format_data["answer"] = splits[-1].strip()
                    prompts.append(prompt.format(
                        subject=subject,
                        question=mmlu_format_data["question"],
                        option_A=splits[-5],
                        option_B=splits[-4],
                        option_C=splits[-3],
                        option_D=splits[-2],
                    ))
                    cot.append(f"The correct answer is {mmlu_format_data["answer"]}")
                    answers.append(mmlu_format_data["answer"])

            # elif dataset == "alpaca":
            #     prompts.append(prompt.format(question=data["instruction"]))
            #     cot.append(data["output"])
            #     answers.append(data["output"])                     

            elif dataset == "gsm8k":
                data = json.loads(line)
                prompts.append(prompt.format(instruction=data["question"]))
                cot.append(data["answer"])
                answers.append(parse_response_gsm8k(data["answer"]))     
                
            else:  # r1_zero_prompt, also for gs8mk because of invalid MATH dataset
                data = json.loads(line)
                prompts.append(prompt.format(question=data["question"]))
                cot.append(convert_cot_to_think_answer(data["answer"]))
                answers.append(extract_gsm8k_answer(data["answer"]))

    return prompts, cot, answers


def load_json_to_list(file_path: str) -> list[dict]:
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line.strip()))

    return data_list


##############################################################################
### CS336 Assignment 5 Supplement (alignment): Instruction Tuning and RLHF ###
##############################################################################
def parse_response_mmlu(response: str) -> str:
    if response and response[0] in ['A', 'B', 'C', 'D']:
        return response[0]
    
    opt_re = re.compile(r" [A-D][\,,\.,\s, ]?")
    matches = opt_re.findall(response)
    if not matches:
        return None
    return  matches[-1][1]

def parse_response_gsm8k(response: str) -> str | None:
    # find all candidate numbers (allow sign, digits, commas and dots), pick the last one
    num_re = re.compile(r"[-+]?[0-9][0-9,\.]*")
    matches = num_re.findall(response)
    if not matches:
        return None
    # clean commas from the chosen match
    ans = matches[-1].replace(",", "")
    if ans[-1] == ".":
        ans = ans[:-1]
    return ans



