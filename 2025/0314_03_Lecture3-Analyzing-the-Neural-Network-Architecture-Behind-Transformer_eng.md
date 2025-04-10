# Lecture 3 Analyzing the Neural Network Architecture Behind Transformer

Full course syllabus reference to [Machine Learning 2025](https://course.ntu.edu.tw/courses/113-2/41735)  
Note for lecture （Hung-yi Lee YouTube）  
(1) [【生成式AI時代下的機器學習(2025)】第三講：AI 的腦科學 — 語言模型內部運作機制剖析 (解析單一神經元到整群神經元的運作機制、如何讓語言模型說出自己的內心世界)](https://www.youtube.com/watch?v=Xnil63UDW2o&list=PLJV_el3uVTsNZEFAdQsDeOdzAaHTca2Gi&index=3)


## Transformer

![transformer](./images/0314/01_transformer.png)


### Single Neuron

Within a Transformer layer, there are sub-layers:
- Self-attention layer  
- Feed-forward layer (applies independently to each token)

Single neuron performs:  
```
Vector → Weighted Sum → Activation Function → Output Vector
```

#### Analyze what a neuron does:
- Observe what happens when the neuron is activated  
- Remove the neuron and see what effect disappears  
   - How to remove it: set its value to 0 or to an average  
- Activate the neuron at different levels and observe changes


#### Conclusion
- Modifying a single neuron usually does not change the overall model output, only shifts some probabilities  
- A task is typically managed by multiple neurons
  - Tasks are likely handled by combinations of neurons. With 4096 neurons, there are up to 2⁴⁰⁹⁶ possible combinations.
  
- A single neuron often participates in multiple tasks  



### Layer of Neurons

**Function Vector**: A vector representing a specific semantic or functional feature.

The output of a layer is called a **representation**.  
If the representation is close to a function vector, the function may be triggered.

```
Representation ≈ Function Vector + Other Function Vectors
```


#### Estimate a function vector

Observe cases where the function is triggered = `function vector` + `other function vectors`

Observe cases where the function is not triggered = `other function vectors'`

Assume average of `other function vectors` and average of `other function vectors'` are similar, so they cancel out.

Then:  

```
Function Vector ≈ (Average of activated representations) − (Average of non-activated representations)
```

#### Validation

Inject the derived `function vector` into the network and check if the intended function is activated.

