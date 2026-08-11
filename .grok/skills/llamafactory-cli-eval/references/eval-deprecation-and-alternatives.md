# **LLaMA-Factory Eval Deprecation and Alternatives**

[https://gemini.google.com/app/7f7d085d43ac3f1d](https://gemini.google.com/app/7f7d085d43ac3f1d)

*User prompt: Apptainer\> llamafactory-cli eval Traceback (most recent call last):   File "/opt/conda/bin/llamafactory-cli", line 7, in \<module\>     sys.exit(main())              ^^^^^^   File "/scratch/indrisch/LLaMA-Factory-copy/src/llamafactory/cli.py", line 24, in main     launcher.launch()   File "/scratch/indrisch/LLaMA-Factory-copy/src/llamafactory/launcher.py", line 146, in launch     raise NotImplementedError("Evaluation will be deprecated in the future.") NotImplementedError: Evaluation will be deprecated in the future. Apptainer\> llamafactory-cli eval \--help Traceback (most recent call last):   File "/opt/conda/bin/llamafactory-cli", line 7, in \<module\>     sys.exit(main())              ^^^^^^   File "/scratch/indrisch/LLaMA-Factory-copy/src/llamafactory/cli.py", line 24, in main     launcher.launch()   File "/scratch/indrisch/LLaMA-Factory-copy/src/llamafactory/launcher.py", line 146, in launch     raise NotImplementedError("Evaluation will be deprecated in the future.") NotImplementedError: Evaluation will be deprecated in the future.*

Response: The llamafactory-cli eval command was deprecated in recent versions of LLaMA-Factory, which is why launcher.py explicitly raises a NotImplementedError.

To run model evaluations now, pass stage: eval to **llamafactory-cli train** or use the Web UI.

## **Solutions**

### **Option 1: YAML Config (*Recommended*)**

Create an evaluation configuration file named eval.yaml:

`### Model`  
`model_name_or_path: path/to/your/model`  
`# adapter_name_or_path: path/to/lora/adapter  # Uncomment if evaluating a LoRA checkpoint`

`### Evaluation Stage`  
`stage: eval`

`### Task Configuration`  
`task: mmlu                  # Supported tasks: mmlu, ceval, gsm8k, humaneval, etc.`  
`task_dir: evaluation`  
`template: default           # Set to your model's chat template (e.g., llama3, qwen, mistral)`  
`lang: en`  
`n_shot: 0`

`### Output`  
`output_dir: saves/eval_results`

Execute the evaluation:

`llamafactory-cli train eval.yaml`

### **Option 2: Direct CLI Arguments**

Run the evaluation directly from the terminal without creating a YAML file:

`llamafactory-cli train \`  
    `--stage eval \`  
    `--model_name_or_path path/to/your/model \`  
    `--task mmlu \`  
    `--task_dir evaluation \`  
    `--template default \`  
    `--lang en \`  
    `--n_shot 0 \`  
    `--output_dir saves/eval_results`

### **Option 3: Web UI (LLaMA Board)**

Launch the interactive interface inside your Apptainer container:

`llamafactory-cli webui`

Once open in your browser, switch to the **Evaluate** tab to select your model and target benchmark visually.

---

