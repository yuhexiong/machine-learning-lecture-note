# Lecture 1 How Machine Learning Gave Rise to Generative AI

Full course syllabus reference to [Machine Learning 2025](https://course.ntu.edu.tw/courses/113-2/41735)  
Note for lecture(Hung-yi Lee YouTube)  
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

---

## Autoregressive Generation

Tokens are generated one at a time in a fixed order:

- Given input:
  $$
  x_1, x_2, \ldots, x_j, \ldots \rightarrow y_1
  $$
- Then:
  $$
  x_1, x_2, \ldots, x_j, \ldots, y_1 \rightarrow y_2
  $$
- Then:
  $$
  x_1, x_2, \ldots, x_j, \ldots, y_1, y_2 \rightarrow y_3
  $$
- ...
- Until:
  $$
  x_1, x_2, \ldots, x_j, \ldots, y_1, y_2, \ldots, y_{T-1} \rightarrow y_T
  $$
- And finally:
  $$
  x_1, x_2, \ldots, x_j, \ldots, y_1, y_2, \ldots, y_T \rightarrow \text{end}
  $$

This process is known as **Autoregressive Generation**:  
Token generation is done sequentially, one token at a time.