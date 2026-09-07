import re
import json
import pandas as pd


_3dthinker10k_path = "/project/aip-wangcs/indrisch/huggingface/hub//datasets--cvis-tmu--3dthinker-10k-mcq/snapshots/c0392e4172ddf9c106b7066c584724dd7ae04144/3dthinker10k_cot.with_question_id.jsonl"
spatialssrl_path = "/project/aip-wangcs/indrisch/huggingface/hub//datasets--cvis-tmu--Spatial-SSRL-81k/snapshots/c6bce21bad8cb7d751a47f7bb91dca7875115c96/SFT-coldstart.with_question_id.json"
scene30k_path = "/project/aip-wangcs/indrisch/huggingface/hub//datasets--cvis-tmu--Scene30K/snapshots/4be0f2eadaf440b9fe9392fdeca790c4edfd68fd/data/train-00000-of-00001.with_question_id.parquet"


_3dthinker10k_path_out = "3dthinker10k_cot.with_question_id.formatted.jsonl"
spatialssrl_path_out = "SFT-coldstart.with_question_id.formatted.json"
scene30k_path_out = "train-00000-of-00001.with_question_id.formatted.parquet"


general_formatting_instruction = "Your output must be formatted as '<think>thought process as you decide on an answer</think><answer>final answer</answer>. Start your response with the '<think>' tag and then write your thought process as you work to answer the question based on the image contents. Once you are confident that you have the correct answer, close your thought process with </think>, open your answer with <answer>, copy in only your final answer, then close your answer (and your output in general) with '</answer>'."

general_formatting_instruction_multiplechoice = "Your output must be formatted as '<think>thought process as you decide on an answer</think><answer>final answer</answer>', where for the answer you must select the *ONE* correct answer from the options listed. For example, if the possible multiple choice answer options are 'A. Above B. Under C. Front D. Behind', start your response with the '<think>' tag and then write your thought process as you work to answer the question based on the image contents. Once you are confident that you have the correct answer (perhaps it is 'A. Above'), close your thought process with </think>, open your answer with <answer>, copy in 'A. Above', then close your answer (and your output in general) with '</answer>'."

spatialssrl_multiple_choice_string = "The final answer should be chosen from"

with open(_3dthinker10k_path, 'r') as f:
    _3dthinker10k = [json.loads(line) for line in f]

with open(spatialssrl_path, 'r') as f:
    spatialssrl = json.load(f)

# scene30k = json.loads(pd.read_parquet(scene30k_path).to_json(orient="records"))
scene30k = pd.read_parquet(scene30k_path) # we load as a table.

# ---- 3dthinker-10k ----

# TODO: adjust the formatting of the 'system' entry.
# "You only need to provide *ONE* correct answer selecting from the options listed below. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should **only** be '<answer>A. Above</answer>'."
_3dthinker10k_answerinstruction = re.match(r".*\n\[Answer Instruction\]\n(.*)\n\n\[Question\]\n.*", _3dthinker10k[0]["system"], re.DOTALL).group(1) # confirmed to be in all 10000 prompts
# _3dthinker10k_answerinstruction_new = "\n[Answer Instruction]\n" + general_formatting_instruction_multiplechoice + "\n\n[Question]\n"

for i in range(len(_3dthinker10k)):
    # fix the system entry
    _3dthinker10k[i]["system"] = _3dthinker10k[i]["system"].replace(_3dthinker10k_answerinstruction, general_formatting_instruction_multiplechoice)
    # TODO: remove the <output_3D>\n at the beginning, and also remove anything between </think> and <answer>.
    _3dthinker10k[i]["output"] = _3dthinker10k[i]['output'][12:]
    extra_content = re.search(r"</think>(.*?)<answer>", _3dthinker10k[i]['output'], re.DOTALL).group()
    _3dthinker10k[i]["output"] = _3dthinker10k[i]["output"].replace(extra_content, "</think><answer>")

# ---- spatialssrl ----

spatialssrl_answerformat = "You FIRST think about the reasoning process as an internal monologue and put your final answer in '\\boxed{}'."
spatialssrl_answerformat_2 = "You FIRST provide a DETAILED analysis and put your final answer in '\\boxed{}'."
spatialssrl_answerformat_3 = "You FIRST provide a detailed reasoning process and put your final answer in '\\boxed{}'."
spatialssrl_answerformat_4 = "You FIRST think about the reasoning process as an internal monologue and put your final answer in '\\boxed{}'."

# NOTE: should we add different ones for different tasks? Likely not, since the rest of the string can describe the required output.

for i in range(len(spatialssrl)):

    # TODO: ensure that we get the multiple choice formatting and extract the full answer string also.

    # ensure that the instruction is for putting think and answer tags. Add the extra space.
    new_instruction = spatialssrl[i]['instruction'].replace(spatialssrl_answerformat, " " + general_formatting_instruction)
    new_instruction = new_instruction.replace(spatialssrl_answerformat_2, " " + general_formatting_instruction)
    new_instruction = new_instruction.replace(spatialssrl_answerformat_3, " " + general_formatting_instruction)
    new_instruction = new_instruction.replace(spatialssrl_answerformat_4, " " + general_formatting_instruction)
    new_instruction = new_instruction.replace("image<image>", "image <image>") # NOTE: does this need to be added as a special tag?
    spatialssrl[i]['instruction'] = new_instruction

    # replace the output from the \boxed format to think and answer
    spatialssrl_output = re.match(r"(.*)\\boxed{(.*)}.*", spatialssrl[i]['output'], re.DOTALL)
    thinking = spatialssrl_output.group(1)
    answer = spatialssrl_output.group(2)


    # if we are in the case where we have a multiple choice, we replace the single char answer with the full answer.
    try:
        r = re.match(r".*(A\..*)\s(B\..*)\s(C\..*)\s(D\.\s[^\.]+)\..*", spatialssrl[i]['instruction'], re.DOTALL)
        answers = [r.group(i) for i in range(1, 5)] # put the extracted possible answers in a list
        answer = answers[ord(answer)-ord('A')] # this will map the answer \in {'A', 'B', 'C', 'D'} to the full extracted answer. This has been confirmed to copy correctly.
    except AttributeError:
        # the case where it is not multiple choice
        pass

    spatialssrl[i]['output'] = "<think>" + thinking + "</think><answer>" + answer + "</answer>"

# ---- scene30k ----

scene30k["formatting_instruction"] = general_formatting_instruction # add a new column which we can use as a system prompt


# === output ===

with open(_3dthinker10k_path_out, 'w') as f:
    for entry in _3dthinker10k:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

with open(spatialssrl_path_out, 'w') as f:
    json.dump(spatialssrl, f, indent=4)

scene30k.to_parquet(scene30k_path_out)
