# Lecture 1 Introduction of Deep Learning Extra

Full course Syllabus reference to [Machine Learing 2022 Spring](https://speech.ee.ntu.edu.tw/~hylee/ml/2022-spring.php?fbclid=IwAR2rE3UFymIOeTEoEzyZBhO-5vbpYpyw1Ho_KHO8cmwVd0_f7nI3iYunW4A)  
Note for lecture(Hung-yi Lee YouTube)  
(1) [ML Lecture 6: Brief Introduction of Deep Learning](https://www.youtube.com/watch?v=Dr-WRlEFefw)  
(2) [ML Lecture 7: Backpropagation](https://www.youtube.com/watch?v=ibJpTrp5mcE)  
(3)  [ML Lecture 1: Regression - Case Study](https://www.youtube.com/watch?v=fegAeph9UaA)  
(4) [ML Lecture 4: Classification](https://www.youtube.com/watch?v=fZAZUYEeIMg)  
(5) [ML Lecture 5: Logistic Regression](https://www.youtube.com/watch?v=hSXFuypLukA)  

## Deep Learning

(1) Define a set of function  
(2) Goodness of function  
(3) Pick the best function  

### Fully Connected Feedforward

Layer 1: input layer  
Layer 2: hidden layer  
...  
Layer N-1: hidden layer (feature extractor replacing feature engineering)  
Layer N: output layer  

<p align="center">
  <img src="./images/0218/08_fully_connected.png" alt="Fully Connected"/>
</p>

Write the calculation as a matrix:  

<p align="center">
  <img src="./images/0218/09_matrix.png" alt="Matrix"/>
</p>

<!-- $$
\sigma \left(
\begin{bmatrix}
1 & -2 \\
-1 & 1
\end{bmatrix}
\begin{bmatrix}
1 \\
-1
\end{bmatrix}
+
\begin{bmatrix}
1 \\
0
\end{bmatrix}
\right)
=
\begin{bmatrix}
0.98 \\
0.12
\end{bmatrix}
$$ -->

Example: Handwriting Digit Recognition  
Input 16*16: 256 dim, output 0~9: 10 dim  

### Cross Entropy

$$
C \Rightarrow \mathcal{L} = \sum C^n \quad \text{gradient descent}
$$

#### Universal Approximation Theorem
Any continuous f

$$
f: \mathcal{R}^N \rightarrow \mathcal{R}^M
$$

can by realized by a network with one hidden layer(with enough hidden neurons)  

## Backpropagation

$$
\mathcal{L}(\theta) = \sum_{n=1}^{N} C^n(\theta)
$$

$$
\frac{\partial}{\partial w} \mathcal{L}(\theta) = \sum_{n=1}^{N} \frac{\partial C^n(\theta)}{\partial w}
$$

<p align="center">
  <img src="./images/0218/10_backpropagation.png" alt="Backpropagation"/>
</p>


The derivative of the cost \( c \) with respect to weight \( w \):

$$
\frac{\partial c}{\partial w} = \frac{\partial z}{\partial w} \cdot \frac{\partial c}{\partial z}
$$

**Forward pass**

$$
z = x_1w_1 + x_2w_2 + b
$$

Partial derivatives:

$$
\frac{\partial z}{\partial w_1} = x_1, \quad \frac{\partial z}{\partial w_2} = x_2
$$

**Backward pass**

$$
\frac{\partial c}{\partial z} = \frac{\partial a}{\partial z} \cdot \frac{\partial c}{\partial a}
$$


$$
\text{where} \quad \frac{\partial a}{\partial z} \quad \text{ is the sigmoid derivative.}
$$

Gradient of \( c \) with respect to \( a \):

$$
\frac{\partial c}{\partial a} = \frac{\partial z'}{\partial a} \cdot \frac{\partial c}{\partial z'} + \frac{\partial z''}{\partial a} \cdot \frac{\partial c}{\partial z''}
$$

Given:

$$
\frac{\partial z'}{\partial a} = w_3, \quad \frac{\partial z''}{\partial a} = w_4
$$

Final expression:

$$
\frac{\partial c}{\partial z} = \sigma'(z) \cdot \left[w_3 \cdot \frac{\partial c}{\partial z'} + w_4 \cdot \frac{\partial c}{\partial z''}\right]
$$


$$
\text{where} \quad  \sigma'(z) \quad  \text{ is constant.}
$$

**Case 1. Output Layer**  

$$
 \frac{\partial c}{\partial z'} =  \frac{\partial y_1}{\partial z'} \cdot \frac{\partial c}{\partial y_1}
$$


$$
 \frac{\partial c}{\partial z''} =  \frac{\partial y_2}{\partial z''} \cdot \frac{\partial c}{\partial y_2}
$$



**Case 2. Not Output Layer**  
Continue to the next layer until reaching the Output Layer.  


## Regularization

If we hope function is smooth, smaller wi is better  

$$
y = b + \sum w_i x_i
$$


so define loss function as:  

$$
L = \sum_n \left(\hat{y}^n - \left( b + \sum w_i x_i \right)\right)^2 + \lambda \sum (w_i)^2
$$

## Classifier

![Classifier](./images/0218/11_classifier.png)  

we have
$$
P(x) =P(x \mid C_1) \cdot P(C_1) + P(x \mid C_2) \cdot P(C_2)
$$


so according Bayes' theorem:  
$$
P(C_1 \mid x) = \frac{P(x \mid C_1) \cdot P(C_1)}{P(x \mid C_1) \cdot P(C_1) + P(x \mid C_2) \cdot P(C_2)}
$$

If x not in C1 of training data but x definitly in Class 1, use C1 to find its Gaussian Distribution.  

$$
L(\mu, \sigma) = f_{\mu, \sigma}(x^1) \cdot f_{\mu, \sigma}(x^2) \cdot \ldots \cdot f_{\mu, \sigma}(x^N)
$$



$$
\mu^*, \sigma^* = \text{arg} \underset{\mu, \sigma}{\text{max}} \, L(\mu, \sigma)
$$


