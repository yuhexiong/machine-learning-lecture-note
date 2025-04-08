# Lecture 2 From Tuning Parameters to Orchestrating AI Agents' Workflow

Full course syllabus reference to [Machine Learning 2025](https://course.ntu.edu.tw/courses/113-2/41735)  
Note for lecture （Hung-yi Lee YouTube）  
(1) [【生成式AI時代下的機器學習(2025)】第二講：一堂課搞懂 AI Agent 的原理 (AI如何透過經驗調整行為、使用工具和做計劃)](https://www.youtube.com/watch?v=M2Yg1kwPpts&list=PLJV_el3uVTsNZEFAdQsDeOdzAaHTca2Gi&index=2)


## AI Agent

Humans do not provide explicit actions, steps, or instructions—only a **goal**.  
The AI figures out how to accomplish the goal by itself.

AI continuously cycles through the following loop:

goal → observe (observation 1) → take action 1 → observe (observation 2) → ... → until the goal is reached.

This is essentially **a chain reaction**.

- **No model training** is involved in this loop.
- It's purely the **application** of models.


## Advantages of Running AI Agents with LLMs

### 1. **Actions** (Example: Playing Go)

- **Typical Agent (e.g., AlphaGo)**:  
  Predefined finite action space, can only play a move on the board.

- **LLM Agent**:  
  Nearly **infinite possibilities** in actions.

### 2. **Optimization** (Example: AI Programmer)

- **Typical Agent**:  
  Must define a reward function (e.g., whether a program compiles), which is subjective and requires manual tuning.

- **LLM Agent**:  
  Can **read logs directly** and reason from them.


## Turn-Based vs Real-World Interaction

- Conceptually:  
  `observation → action → observation → action`

- Real-world:  
  A new **observation may arrive before the action ends**,  
  e.g., **spoken dialogue** – people may interrupt mid-response.


## Key Abilities of an AI Agent

### 1. **Adjust Behavior Based on Experience**

- **Feedback = observation**  
  → Use it to produce the next action.

- When the chain becomes too long, the agent must **review all past observations and actions**.

To handle this, AI uses:

#### (1) **Reading from Memory**
- Stores information in **Agent’s Memory**.
- Uses **retrieval-based methods** to only fetch **relevant** memories.
- Action is generated based on both the **relevant memory** and the **current observation**.

**Negative feedback has almost no effect** on current LLMs.

#### (2) **Selective Memory Storage**
- AI asks itself:  
  *Is this information important enough to remember?*
- Only stores **important** events in memory.

#### (3) **Reflection**
- Reorganize memory.
- Generate **new ideas**.
- Construct **knowledge graphs** from memories.


### 2. **Using Tools (Function Calls)**

- Examples of tools:
  1. Search engines
  2. Code execution
  3. Other AIs

- The **System Prompt** tells the model **how to use tools** (higher priority).
- The **User Prompt** contains the user’s request (lower priority).

The language model **generates text** (e.g., code).  
When certain trigger phrases appear, the Agent executes the corresponding **tool**.


### 3. **Planning Capability**

