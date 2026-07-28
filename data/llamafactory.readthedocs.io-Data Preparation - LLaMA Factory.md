# Data Preparation - LLaMA Factory

[llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html](https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html)

---

# Data Preparation

[dataset\_info.json](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dataset_info.json/) contains all pre-processed **local datasets** and **online datasets**. If you wish to use a custom dataset, you **must** add the definition of the dataset and its content to the `dataset_info.json` file.

Currently, we support datasets in [Alpaca](#alpaca) format and [ShareGPT](#sharegpt) format.

## Alpaca

The dataset format requirements for different tasks are as follows:

-   [Supervised Fine-Tuning](#id4)

-   [Pre-training](#id6)

-   [Preference Training](#id9)

-   [KTO](#kto)

-   [Multimodal](#id13)

### Supervised Fine-Tuning Datasets

**Sample Dataset**: [SFT Sample Dataset](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/alpaca_zh_demo.json/)

Instruct Tuning optimizes the model’s performance on specific instructions by enabling it to learn from detailed instructions and their corresponding responses.

The content of the `instruction` column corresponds to human instructions, the `input` column corresponds to human input, and the `output` column corresponds to the model’s response. Below is an example:

"alpaca\_zh\_demo.json"
{
  "instruction": "计算这些物品的总费用。 ",
  "input": "输入：汽车 - $3000，衣服 - $100，书 - $20。",
  "output": "汽车、衣服和书的总费用为 $3000 + $100 + $20 = $3120。"
},

During supervised fine-tuning, the content of the `instruction` column is concatenated with the content of the `input` column to form the final human input, i.e., `instruction input`. The content of the `output` column serves as the model’s response. In the example above, the final human input is:

计算这些物品的总费用。
输入：汽车 - $3000，衣服 - $100，书 - $20。

The model’s response is:

汽车、衣服和书的总费用为 $3000 + $100 + $20 = $3120。

If specified, the content of the `system` column will be used as the system prompt.

The `history` column is a list consisting of multiple string tuples, representing the instruction and response for each turn in the history. Note that during supervised fine-tuning, the responses in the history messages are also used for model learning.

The **format requirements** for Supervised Fine-Tuning datasets are as follows:

\[
  {
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "output": "模型回答（必填）",
    "system": "系统提示词（选填）",
    "history": \[
      \["第一轮指令（选填）", "第一轮回答（选填）"\],
      \["第二轮指令（选填）", "第二轮回答（选填）"\]
    \]
  }
\]

Below is an example of multi-turn conversation in Alpaca format. For single-turn conversations, simply omit the `history` column.

\[
  {
    "instruction": "今天的天气怎么样？",
    "input": "",
    "output": "今天的天气不错，是晴天。",
    "history": \[
      \[
        "今天会下雨吗？",
        "今天不会下雨，是个好天气。"
      \],
      \[
        "今天适合出去玩吗？",
        "非常适合，空气质量很好。"
      \]
    \]
  }
\]

For data in the above format, the **dataset description** in `dataset_info.json` should be:

"dataset\_name": {
  "file\_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history"
  }
}

### Pre-training Datasets

**Sample Dataset**: [Pre-training Sample Dataset](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/c4_demo.json/)

Large Language Models (LLMs) are pre-trained by learning from unlabeled text to acquire language representations. Typically, pre-training datasets are obtained from the internet, as it provides a vast amount of text information from different domains, which helps improve the model’s generalization ability.

The text description format for pre-training datasets is as follows:

\[
  {"text": "document"},
  {"text": "document"}
\]

During pre-training, only the **content** in the `text` column (i.e., the document) is used for model learning.

For data in the above format, the **dataset description** in `dataset_info.json` should be:

"dataset\_name": {
  "file\_name": "data.json",
  "columns": {
    "prompt": "text"
  }
}

### Preference Datasets

Preference datasets are used for Reward Model training, DPO training, and ORPO training. For a given system instruction and human input, the preference dataset provides a better response and a worse response.

[Some research](https://openai.com/index/instruction-following/) suggests that teaching models “what is better” can make them more aligned with human needs. It can even enable models with fewer parameters to outperform those with more parameters.

Preference datasets need to provide the better response in the `chosen` column and the worse response in the `rejected` column. The format for a single turn is as follows:

\[
  {
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "chosen": "优质回答（必填）",
    "rejected": "劣质回答（必填）"
  }
\]

For data in the above format, the **dataset description** in `dataset_info.json` should be:

"dataset\_name": {
  "file\_name": "data.json",
  "ranking": true,
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}

### KTO Datasets

KTO datasets are similar to preference datasets, but instead of providing a better and a worse response, KTO datasets only provide a true/false `label` for each turn.

In addition to the final human input (formed by `instruction` and `input`) and the model response `output`, KTO datasets require an additional `kto_tag` column (true/false) to represent human feedback.

**The format for a single turn is as follows:**
\[
  {
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "output": "模型回答（必填）",
    "kto\_tag": "人类反馈 \[true/false\]（必填）"
  }
\]

For data in the above format, the **dataset description** in `dataset_info.json` should be:

"dataset\_name": {
  "file\_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "kto\_tag": "kto\_tag"
  }
}

### Multimodal Datasets

Currently, we support [multimodal image datasets](#id15), [video datasets](#id17), and [audio datasets](#id19).

#### Image Datasets

Multimodal image datasets require an additional `images` column containing the paths to the input images. Note that the number of images must strictly match the number of <image> tags in the text.

\[
  {
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "output": "模型回答（必填）",
    "images": \[
      "图像路径（必填）"
    \]
  }
\]

For data in the above format, the **dataset description** in `dataset_info.json` should be:

"dataset\_name": {
  "file\_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "images": "images"
  }
}

#### Video Datasets

Multimodal video datasets require an additional `videos` column containing the paths to the input videos. Note that the number of videos must strictly match the number of <video> tags in the text.

\[
  {
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "output": "模型回答（必填）",
    "videos": \[
      "视频路径（必填）"
    \]
  }
\]

For data in the above format, the **dataset description** in `dataset_info.json` should be:

"dataset\_name": {
  "file\_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "videos": "videos"
  }
}

#### Audio Datasets

Multimodal audio datasets require an additional `audio` column containing the paths to the input audios. Note that the number of audios must strictly match the number of <audio> tags in the text.

\[
  {
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "output": "模型回答（必填）",
    "audios": \[
      "音频路径（必填）"
    \]
  }
\]

For data in the above format, the **dataset description** in `dataset_info.json` should be:

"dataset\_name": {
  "file\_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "audios": "audios"
  }
}

## ShareGPT

The dataset format requirements for different tasks are as follows:

-   [Supervised Fine-Tuning](#id22)

-   [Preference Training](#id24)

-   [OpenAI Format](#openai)

Note

-   KTO datasets ([sample](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/kto_en_demo.json/)) and multimodal datasets ([sample](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/mllm_demo.json/)) in ShareGPT format are similar to those in Alpaca format.

-   Pre-training datasets do not support ShareGPT format.

### Supervised Fine-Tuning Datasets

**Sample Dataset**: [SFT Sample Dataset](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/glaive_toolcall_zh_demo.json/)

Compared to `alpaca` format datasets, `sharegpt` format supports **more** role types, such as human, gpt, observation, function, etc. They form a list of objects presented in the `conversations` column. Below is an example of `sharegpt` format:

{
  "conversations": \[
    {
      "from": "human",
      "value": "你好，我出生于1990年5月15日。你能告诉我我今天几岁了吗？"
    },
    {
      "from": "function\_call",
      "value": "{\\"name\\": \\"calculate\_age\\", \\"arguments\\": {\\"birthdate\\": \\"1990-05-15\\"}}"
    },
    {
      "from": "observation",
      "value": "{\\"age\\": 31}"
    },
    {
      "from": "gpt",
      "value": "根据我的计算，你今天31岁了。"
    }
  \],
  "tools": "\[{\\"name\\": \\"calculate\_age\\", \\"description\\": \\"根据出生日期计算年龄\\", \\"parameters\\": {\\"type\\": \\"object\\", \\"properties\\": {\\"birthdate\\": {\\"type\\": \\"string\\", \\"description\\": \\"出生日期以YYYY-MM-DD格式表示\\"}}, \\"required\\": \[\\"birthdate\\"\]}}\]"
}

Note that human and observation must appear at odd positions, while gpt and function must appear at even positions.

\[
  {
    "conversations": \[
      {
        "from": "human",
        "value": "人类指令"
      },
      {
        "from": "function\_call",
        "value": "工具参数"
      },
      {
        "from": "observation",
        "value": "工具结果"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      }
    \],
    "system": "系统提示词（选填）",
    "tools": "工具描述（选填）"
  }
\]

For data in the above format, the **dataset description** in `dataset_info.json` should be:

"dataset\_name": {
  "file\_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "system": "system",
    "tools": "tools"
  }
}

### Preference Datasets

**Sample Dataset**: [Preference Sample Dataset](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dpo_zh_demo.json/)

Preference datasets in ShareGPT format also need to provide the better message in the `chosen` column and the worse message in the `rejected` column. Below is an example:

{
  "conversations": \[
    {
      "from": "human",
      "value": "老虎会吃哪一个，草还是树叶？"
    }
  \],
  "chosen": {
    "from": "gpt",
    "value": "我很高兴能帮助你解答问题！不过，我必须指出，这个问题本身可能没有意义。老虎是食肉动物，通常不吃草或树叶。它们的饮食主要包括大型有蹄动物、小型哺乳动物和鸟类。\\n\\n如果你提出这个问题是出于好奇或有特定原因，我很乐意帮助你找到更多关于老虎饮食习惯的信息，或者建议资源让你更多了解这些壮丽的生物。请告诉我如何进一步协助你！"
  },
  "rejected": {
    "from": "gpt",
    "value": "一只老虎是食肉动物，这意味着它的饮食主要是肉类。老虎主要猎捕和吃其他动物，如鹿和野猪。它们不会选择吃草或树叶，因为这些不是它们自然饮食的一部分。"
  }
}

The format is as follows:

\[
  {
    "conversations": \[
      {
        "from": "human",
        "value": "人类指令"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      },
      {
        "from": "human",
        "value": "人类指令"
      }
    \],
    "chosen": {
      "from": "gpt",
      "value": "优质回答"
    },
    "rejected": {
      "from": "gpt",
      "value": "劣质回答"
    }
  }
\]

For data in the above format, the **dataset description** in `dataset_info.json` should be:

"dataset\_name": {
  "file\_name": "data.json",
  "formatting": "sharegpt",
  "ranking": true,
  "columns": {
    "messages": "conversations",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}

### OpenAI Format

OpenAI format is essentially a special case of the `sharegpt` format, where the first message may be a system prompt.

\[
  {
    "messages": \[
      {
        "role": "system",
        "content": "系统提示词（选填）"
      },
      {
        "role": "user",
        "content": "人类指令"
      },
      {
        "role": "assistant",
        "content": "模型回答"
      }
    \]
  }
\]

For data in the above format, the **dataset description** in `dataset_info.json` should be:

"dataset\_name": {
  "file\_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages"
  },
  "tags": {
    "role\_tag": "role",
    "content\_tag": "content",
    "user\_tag": "user",
    "assistant\_tag": "assistant",
    "system\_tag": "system"
  }
}
