# 第一講 機器學習如何創造出了生成式AI

完整課程大綱請參考 [Machine Learning 2025](https://course.ntu.edu.tw/courses/113-2/41735)  
筆記為以下課程 （Hung-yi Lee YouTube）  
(1) [【生成式AI時代下的機器學習(2025)】第一講：一堂課搞懂生成式人工智慧的技術突破與未來發展](https://www.youtube.com/watch?v=QLiKmca4kzI&list=PLJV_el3uVTsNZEFAdQsDeOdzAaHTca2Gi)  

## 生成式 AI（Generative AI）

生成式 AI 是指能夠產生內容（例如文字、圖像或音訊）的系統。

機器展現出一種推理的過程。

給定一個輸入 `x`，它會產生一個輸出 `y`，這個輸出可以是一段文字、一張圖像，或是聲音。

令輸出表示為：

$$
y = \{y_1, y_2, \ldots, y_i, \ldots\}
$$

每個 `y_i` 通常被稱為一個 **token**。

一個 token 可以代表：
- 一個字元
- 一個像素
- 一個取樣的音訊點

每個 token 有有限的可能值，但從這些值中可以組合出無限的可能性。

> "Those tokens were words, some of the tokens of course could now be images, or charts, or tables, songs... speech, videos. Those tokens could be anything."  
> — *Jensen Huang*

## 自回歸式生成（Autoregressive Generation）

Token 會依序一個一個地被產生：

給定輸入：

$$
x_1, x_2, \ldots, x_j, \ldots \rightarrow y_1
$$

然後：

$$
x_1, x_2, \ldots, x_j, \ldots, y_1 \rightarrow y_2
$$

然後：

$$
x_1, x_2, \ldots, x_j, \ldots, y_1, y_2 \rightarrow y_3
$$

……

直到：

$$
x_1, x_2, \ldots, x_j, \ldots, y_1, y_2, \ldots, y_{T-1} \rightarrow y_T
$$

最後：

$$
x_1, x_2, \ldots, x_j, \ldots, y_1, y_2, \ldots, y_T \rightarrow \text{end}
$$

這個過程稱為 **Autoregressive Generation**：  
token 的產生是依序、一個接著一個地進行。

我們也可以不區分輸入 `x` 與輸出 `y`，來描述這個生成過程。  
即使 `x` 和 `y` 的基本領域不同，也可以合併為一個序列：

$$
\{z_1, z_2, \ldots, z_{t-1}\} \rightarrow z_t
$$

這可以表示為一個函數（神經網路），它輸出下一個 token 的機率分布。

## 深度學習（Deep Learning）

使用多層會帶來更好的學習效果。

讓機器「思考」多次也是一種深度。  
雖然層數有限，但模型可以進行多次推理。  
這個概念稱為 **Testing Time Scaling**。

**實作：**  
將語言模型的 `end` token 替換為 `wait` token，以控制語言模型的輸出長度。

- **Transformer**：**Self-Attention 層** 需要存取所有的 token 才能產生答案。  
  因此，序列長度必須受到限制。

- **Mamba:**（稍後討論）

### 架構（Architecture）與參數設定（Parameters）

- **architecture**（包括內部的超參數）由人類定義。
- **parameters** 是從訓練資料中學習出來的。

當一個模型被描述為 **7B** 或 **70B**，表示它的參數數量。

- 參數的數量是模型架構的一部分。
- 參數的 **數值** 必須從訓練資料中學習出來。

生成函數可表示為：

$$
z_t = f_\theta(z_1, z_2, \ldots, z_{t-1})
$$

我們的目標是找出一組參數 \( \theta \)，使得函數 \( f \) 產生的機率分布中，正確的 token 擁有最高的機率（根據訓練資料）。


## 多任務模型（Multi-Task Model）

語言翻譯不再需要針對每一組語言對進行訓練。  
模型學會將輸入轉換為一種內部語言表示。  
各種功能可以整合至單一模型中。

### 1. 編碼器為基礎的模型（2018–2019）

- 輸入：文字  
- 處理方式：  
  文字 → **編碼器** → 向量 → 專用模型 → 摘要 / 翻譯  
- 著名模型：**ELMO**、**BERT**、**ERNIE**

### 2. 全文生成模型（2020–2022）

- 輸入：文字  
- 處理方式：  
  文字 → **模型** \( f_\theta \) → 文字  
- 常見的作法是微調：
  - 文字 → 模型 \( f_\theta' \) → 摘要  
  - 文字 → 模型 \( f_\theta'' \) → 翻譯  
- 著名模型：**GPT-3**

### 3. 指令導向模型（2023–）

- 相同的架構、相同的參數  
- 輸入：文字 + 提示詞（Prompt）  
- 處理方式：  
  文字 → 模型（加上提示詞）→ 摘要 / 翻譯  
- 著名模型：**ChatGPT**、**LLaMA**、**Claude**、**Gemini**、**DeepSeek**

## 終身學習（Life-long Learning）

調整模型行為的方式：

### 暫時性的修改
- **使用提示詞（Prompts）**：透過引導模型的回答，暫時性改變其行為。

### 永久性的修改
1. **微調（Fine-tuning）**：  
   加入新的訓練資料。可能會降低模型原本的能力。

2. **模型編輯（Model Editing）**：  
   手動調整模型的特定參數。

3. **模型合併（Model Merging）**

