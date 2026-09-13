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
#_3dthinker10k_answerinstruction = re.match(r".*\n\[Answer Instruction\]\n(.*)\n\n\[Question\]\n.*", _3dthinker10k[0]["system"], re.DOTALL).group(1) # confirmed to be in all 10000 prompts
# _3dthinker10k_answerinstruction_new = "\n[Answer Instruction]\n" + general_formatting_instruction_multiplechoice + "\n\n[Question]\n"

for i in range(len(_3dthinker10k)):
    _3dthinker10k_answerinstruction = re.match(r".*\n\[Answer Instruction\]\n(.*)\n\n\[Question\]\n.*", _3dthinker10k[i]["system"], re.DOTALL).group(1)
    # fix the system entry
    _3dthinker10k[i]["system"] = _3dthinker10k[i]["system"].replace(_3dthinker10k_answerinstruction, general_formatting_instruction_multiplechoice)
    _3dthinker10k[i]["system"] = "".join(_3dthinker10k[i]['system'].split('.')[:["[Question]" in x for x in _3dthinker10k[i]['system'].split('.')].index(True)]) # remove the question from the system prompt
    # TODO: remove the <output_3D>\n at the beginning, and also remove anything between </think> and <answer>. -> done!
    _3dthinker10k[i]["output"] = _3dthinker10k[i]['output'][12:]
    extra_content = re.search(r"</think>(.*?)<answer>", _3dthinker10k[i]['output'], re.DOTALL).group()
    _3dthinker10k[i]["output"] = _3dthinker10k[i]["output"].replace(extra_content, "</think><answer>")
    # take the preceeding <image> tag section from system, remove the '\n's and put it at the start of the question
    image_portion = _3dthinker10k[i]['system'][::-1][_3dthinker10k[i]['system'][::-1].index("<image>"[::-1]):][::-1].replace("\n", "") # get the image tags (without the newlines)
    _3dthinker10k[i]["system"] = _3dthinker10k[i]['system'][::-1][:_3dthinker10k[i]['system'][::-1].index("<image>"[::-1])][::-1] # remove the image part
    _3dthinker10k[i]["instruction"] = image_portion + " " + _3dthinker10k[i]["instruction"] # add the image part

# ---- spatialssrl ----

spatialssrl_answerformat = "You FIRST think about the reasoning process as an internal monologue and put your final answer in '\\boxed{}'."
spatialssrl_answerformat_2 = "You FIRST provide a DETAILED analysis and put your final answer in '\\boxed{}'."
spatialssrl_answerformat_3 = "You FIRST provide a detailed reasoning process and put your final answer in '\\boxed{}'."
spatialssrl_answerformat_4 = "You FIRST think about the reasoning process as an internal monologue and put your final answer in '\\boxed{}'."

# NOTE: should we add different ones for different tasks? Likely not, since the rest of the string can describe the required output.

for i in range(len(spatialssrl)):

    # rename instruction to the unused 'input' to be the query (which includes the image tags, as the queery should). Ensure that the input is for putting think and answer tags. Add the extra space.
    new_instruction = spatialssrl[i]['instruction'].replace(spatialssrl_answerformat, "")
    new_instruction = new_instruction.replace(spatialssrl_answerformat_2, "")
    new_instruction = new_instruction.replace(spatialssrl_answerformat_3, "")
    new_instruction = new_instruction.replace(spatialssrl_answerformat_4, "")
    new_instruction = new_instruction.replace("image<image>", "image <image>") # NOTE: does this need to be added as a special tag? ANS: no.
    spatialssrl[i]['input'] = new_instruction

    # put the instruction as a separate column 'instruction' to be the system prompt.
    spatialssrl[i]['instruction'] = general_formatting_instruction 

    # replace the output from the \boxed format to think and answer
    spatialssrl_output = re.match(r"(.*)\\boxed{(.*)}.*", spatialssrl[i]['output'], re.DOTALL)
    thinking = spatialssrl_output.group(1)
    answer = spatialssrl_output.group(2)


    # if we are in the case where we have a multiple choice, we replace the single char answer with the full answer.
    try:
        r = re.match(r".*(A\..*)\s(B\..*)\s(C\..*)\s(D\.\s[^\.]+)\..*", spatialssrl[i]['input'], re.DOTALL)
        answers = [r.group(i) for i in range(1, 5)] # put the extracted possible answers in a list
        answer = answers[ord(answer)-ord('A')] # this will map the answer \in {'A', 'B', 'C', 'D'} to the full extracted answer. This has been confirmed to copy correctly.
    except AttributeError:
        # the case where it is not multiple choice
        pass

    spatialssrl[i]['output'] = "<think>" + thinking + "</think><answer>" + answer + "</answer>"

# ---- scene30k ----

scene30k["formatting_instruction"] = general_formatting_instruction # add a new column which we can use as a system prompt

# ensure that the question_with_image_tags is formatted as "<image>...<image> <question>" rather than "<question> <image>...<image>"
def scene30k_move_image_tags(question_with_image_tags):
    tags_start_idx = question_with_image_tags.index("<image>")
    return question_with_image_tags[tags_start_idx:] + " " + question_with_image_tags[:tags_start_idx].strip()
    
scene30k['question_with_image_tags'] = scene30k['question_with_image_tags'].map(lambda x : scene30k_move_image_tags(x))

# === output ===

with open(_3dthinker10k_path_out, 'w') as f:
    for entry in _3dthinker10k:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

with open(spatialssrl_path_out, 'w') as f:
    json.dump(spatialssrl, f, indent=4)

scene30k.to_parquet(scene30k_path_out)
