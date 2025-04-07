# Lecture 1 How Machine Learning Gave Rise to Generative AI

Full course syllabus reference to [Machine Learning 2025](https://course.ntu.edu.tw/courses/113-2/41735)  
Note for lecture （Hung-yi Lee YouTube）  
(1) [【生成式AI時代下的機器學習(2025)】第一講：一堂課搞懂生成式人工智慧的技術突破與未來發展](https://www.youtube.com/watch?v=QLiKmca4kzI&list=PLJV_el3uVTsNZEFAdQsDeOdzAaHTca2Gi)  


## Generative AI

Generative AI refers to systems capable of producing content such as text, images, or audio.

The machine exhibits a process of reasoning.

Given an input `x`, it produces an output `y`, which could be a sequence of text, an image, or sound.

Let the output be represented as:

$$
y = \{y_1, y_2, \ldots, y_i, \ldots\}
$$

Each `y_i` is typically called a **token**.

A token can represent:
- a character
- a pixel
- a sampled audio point

Each token has a finite number of possible values, yet from these, infinite possibilities can be composed.

> "Those tokens were words, some of the tokens of course could now be images, or charts, or tables, songs... speech, videos. Those tokens could be anything."  
> — *Jensen Huang*


## Autoregressive Generation

Tokens are generated one at a time in a fixed order:

Given input:

$$
x_1, x_2, \ldots, x_j, \ldots \rightarrow y_1
$$


Then:

$$
x_1, x_2, \ldots, x_j, \ldots, y_1 \rightarrow y_2
$$

Then:

$$
x_1, x_2, \ldots, x_j, \ldots, y_1, y_2 \rightarrow y_3
$$

...

Until:

$$
x_1, x_2, \ldots, x_j, \ldots, y_1, y_2, \ldots, y_{T-1} \rightarrow y_T
$$

And finally:

$$
x_1, x_2, \ldots, x_j, \ldots, y_1, y_2, \ldots, y_T \rightarrow \text{end}
$$

This process is known as **Autoregressive Generation**:  
Token generation is done sequentially, one token at a time.


We can describe the generative process without distinguishing between input `x` and output `y`.  
Even if the base domains of `x` and `y` are different, they can be unified into a single sequence:

$$
\{z_1, z_2, \ldots, z_{t-1}\} \rightarrow z_t
$$

This can be represented as a function (a Neural Network) that outputs a probability distribution over the next token.


## Deep Learning

Using multiple layers leads to better learning performance.

Letting the machine "think" multiple times is another form of depth.  
While the number of layers is limited, the model can reason multiple times.  
This concept is called **Testing Time Scaling**.

**Implementation:**  
Replace the language model's `end` token with a `wait` token to control the output length of the language model.


- **Transformer**: the **Self-Attention layer** requires access to all tokens before generating an answer.  
  Therefore, the sequence length must be limited.

- **Mamba:** (To be discussed later)


### Architecture and Parameters

- The **architecture** (including internal hyperparameters) is defined by humans.
- The **parameters** are learned from training data.

When a model is described as **7B** or **70B**, it refers to the number of parameters.

- The number of parameters is part of the model's architecture.
- The **values** of the parameters must be learned from training data.

The generative function can be written as:

$$
z_t = f_\theta(z_1, z_2, \ldots, z_{t-1})
$$

We aim to find a parameter set \( \theta \) such that the function \( f \) produces a probability distribution where the correct token has the highest probability according to the training data.

## Multi-Task Model

Language translation no longer requires training for each language pair.  
The model learns to convert inputs into an internal language representation.  
Various functionalities can be integrated into a single model.

### 1. Encoder-based Models (2018–2019)

- Input: Text  
- Processing:  
  Text → **Encoder** → Vector → Specialized Model → Summary / Translation  
- Well-known models: **ELMO**, **BERT**, **ERNIE**


### 2. Full Text Generation (2020–2022)

- Input: Text  
- Processing:  
  Text → **Model** \( f_theta \) → Text  
- Fine-tuning is common:
  - Text → Model \( f_theta' \) → Summary  
  - Text → Model \( f_theta'' \) → Translation  
- Well-known model: **GPT-3**


### 3. Instruction-following Models (2023–)

- Same architecture, same parameters  
- Input: Text + Prompt  
- Processing:  
  Text → Model (with prompt) → Summary / Translation  
- Well-known models: **ChatGPT**, **LLaMA**, **Claude**, **Gemini**, **DeepSeek**


## Life-long Learning

Ways to modify a model’s behavior:

### Temporary Modification
- **Using Prompts**: Temporarily alters the model’s behavior by guiding its response.

### Permanent Modification
1. **Fine-tuning**:  
   Add new training data. May degrade the model's original capabilities.

2. **Model Editing**:  
   Manually adjust specific parameters of the model.

3. **Model Merging**

