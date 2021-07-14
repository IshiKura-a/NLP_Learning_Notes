# S1 Classification and Vector Spaces

## Introduction

1. Represent texts as a vector and build a classifier by **Logistic Regression**.
2. Use **Naive Bayes** on the same problem.
3. **Vector Space Model**

## CH1 Logistic Regression

Supervised ML (training):

![image-20210712142955828](.\imgs\image-20210712142955828.png)

This chapter's target: use **logistic regression** to classify whether a tweet is positive or not, i.e. to do **sentiment analysis**.

![image-20210712143308071](.\imgs.\imgs\image-20210712143308071.png)

The first step is to represent a tweet into a vector.

First, build a frequency dictionary like this:

![image-20210712144639768](.\imgs.\imgs\image-20210712144639768.png)

It's build by the positive tweets and negative tweets separately.

Next, we conduct the feature extraction, by represent the $m^{\text{th}}$ tweet with the vector:
$$
X_m=[1, \sum_{w}\text{freq}(w,1), \sum_{w}\text{freq}(w,0)]
$$
where 1 is a bias item,  $\sum_{w}\text{freq}(w,1)$ is the sum of the positive frequency of all the words in the tweet and $\sum_{w}\text{freq}(w,0)$ is the sum of the negative frequency of all the words in the tweet.

For example, the tweet "I am sad, I am not learning NLP" is represent as $[1,8,10]$, where 8 = 3+3+1+1, and 11 = 3+3+1+1+1+1.

From examples listed above, we found the word "am" got a quite high frequency, while the word per se seems to have no influence on the classification. Thus, before we represent the tweet into a vector, we need to do some preprocessing, for example **remove the words that are not that important.**

Basically, we have **Stop words** and **Punctuation** e.g.

![image-20210712150407404](.\imgs.\imgs\image-20210712150407404.png)

Meanwhile, a tweet usually have **Handle**s and **URL**s, which also have no influence on sentiment analysis and can be removed.

![image-20210712151213231](.\imgs.\imgs\image-20210712151213231.png)

Also, a word may have multiple formats like word "tune", may be "tuning", "tunes", "TUNE" and so on. To address this problem, we use **Stemming** and **Lowercase**.

![image-20210712151821949](.\imgs.\imgs\image-20210712151821949.png)

Now, we get the vectors of input data, and then we will use a function $f$, together with its parameter $\theta$, to get the predict label $\hat{Y}$, i.e. $\hat{Y}=f(\theta,X)$.

In this chapter, we use **sigmoid function**:
$$
\hat{Y}=\frac{1}{1+e^{-\theta^T{X}}}
$$
To train the parameter $\theta$, we use **gradient descent**:

![image-20210712153651510](.\imgs.\imgs\image-20210712153651510.png)

## CH2 Naive Bayes

Bayes Rule:
$$
P(Y|X)=\frac{P(X,Y)}{P(X)}=P(X|Y)\times\frac{P(Y)}{P(X)}
$$
Let's go back to the vector generation again. Now, we do not use any pre-processing techniques. For training set:

Positive:

* I am happy because I am learning NLP
* I am happy not sad

Negative:

* I am sad I am not learning NLP
* I am sad not happy

We get the frequency as follows:

![image-20210712160855685](.\imgs.\imgs\image-20210712160855685.png)

We find that in Pos, we have 13 classes while in Neg, there's only 12. Then, we compute the possibility $P(w_i|class)$ , for example, $P(I|Pos)=\frac{3}{13}=0.24$ and get another table.

In this table, some words have nearly identical conditional probability like "**I**", "**am**" and "**learning**". They don't add anything to the sentiment.

In contrast, words like "**happy**", "**sad**" and "**not**" have significant difference between probabilities. They are **power words** which express one sentiment or the other. 

Additionally, words like "**because**" only appears in one sentiment, which could not be computed in sentiment analysis. To solve this, we need to **smooth our probability function**.

Say we get a new tweet "I am happy today; I am learning." Now we use the probability table above to predict the sentiment of the tweet with the formula:
$$
\prod_{i=1}^{m}\frac{P(w_i|pos)}{P(w_i|neg)}=\frac{0.24}{0.25}\times\frac{0.24}{0.25}\times\frac{0.15}{0.08}\times\frac{0.24}{0.25}\times\frac{0.24}{0.25}\times\frac{0.08}{0.08}=1.59>1
$$
 Since the result is greater than 1, the sentiment is **positive**.

### Laplacian Smoothing

From the process above, we found that, we cannot deal with words like "**because**", as they only appears in one category. To address this, we introduce **Laplacian Smoothing**.

Previously, we define class as $\text{class}\in\{\text{Positive}, \text{Negative}\}$，and define the conditional probability to be:
$$
P(w_i|class)=\frac{\text{freq}(w_i, class)}{N_{class}}
$$
where $N_{class}$ is the frequency of all words in the class. This conditional probability could be one, therefore, we add 1 to the numerator and becomes:
$$
P(w_i|class)=\frac{\text{freq}(w_i, class)+1}{N_{class}}
$$
But now the sum of it is not 1. Therefore, we add a $V_{class}$ to the denominator:
$$
P(w_i|class)=\frac{\text{freq}(w_i, class)+1}{N_{class}+V_{class}}
$$
which is the number of **unique** words in the class. Now the probabilities sum to 1.

![image-20210712165407160](.\imgs\image-20210712165407160.png)

This is called **Laplacian Smoothing**.

### Log Likelihood

Now, let's compute the ratio of the words above, with the formular:
$$
\text{ratio}(w_i)=\frac{P(w_i|pos)}{P(w_i|neg)}
$$
![image-20210713131639206](.\imgs\image-20210713131639206.png)

The relationship between the ratio and the sentiment is listed on the left. Approximately, we have
$$
\text{ratio}(w_i)\approx\frac{\text{freq}(w_i,1)+1}{\text{freq}(w_i,0)+1}
$$
Previously, we use the product of the ratios to decide if a tweet is positive or not. This is called the **likelihood**. But that's only the case we have balanced data set. Actually, we have to take the **prior probability** into account:
$$
\frac{P(pos)}{P(neg)}\cdot\prod_{i=1}^{m}\frac{P(w_i|pos)}{P(w_i|neg)}>1
$$
However, the product brings risk of underflow. Hence, we introduce **log likelihood**,
$$
\log\frac{P(pos)}{P(neg)}\cdot\prod_{i=1}^{m}\frac{P(w_i|pos)}{P(w_i|neg)}=\log\frac{P(pos)}{P(neg)}+\sum_{i=1}^{m}\log\frac{P(w_i|pos)}{P(w_i|neg)}
$$
which is a log prior $\log\frac{P(pos)}{P(neg)}$ plus a log likelihood $\sum_{i=1}^{m}\log\frac{P(w_i|pos)}{P(w_i|neg)}$.

Let's denote
$$
\lambda(w)=\log\frac{P(w|pos)}{P(w|neg)}
$$
From the example above, we get

![image-20210713133310443](.\imgs\image-20210713133310443.png)

* In testing, we treat unknown words as neutral.

## CH3 Vector Space Model

* **Word by word design**： number of times they occur together within a certain distance k.

![image-20210713140834102](.\imgs\image-20210713140834102.png)

* **Word by document design**: number of times a word occurs within a certain category.

![image-20210713141022175](.\imgs\image-20210713141022175.png)

Build Vector Space:

![image-20210713141205694](.\imgs\image-20210713141205694.png)

### Euclidean Distance

A similarity matrix to identify how far two points or vectors are apart from each other. 

**Euclidean Distance** (n-dimension):
$$
d(A,B)=\sqrt{\sum_{i=1}^n(A_i-B_i)^2}
$$
 i.e. the norm of A - B. 

However, the Euclidean Distance sometimes cannot tell the true similarity.

![image-20210713142413837](.\imgs\image-20210713142413837.png)

In fact, Food corpus should be more similar to Agriculture corpus than History corpus. However, because Food corpus has fewer words, the distance is much more longer. If we compare the cosine of the angle between the two vectors, we will get the true result. That's why we use **cosine similarity** when corpora are different sizes.
$$
\cos<\vec{v},\vec{w}>=\frac{\vec{v}\cdot\vec{w}}{||\vec{v}||\times||\vec{w}||}
$$
Usually, by the method above, the vector would have many dimensions, which is bad for computation. Hence, we need dimension reduction here. The most simple way is to use **PCA**.

### PCA

**Principle Component Analysis** aims to find a projection vector $u$ to retain the **maximum projection variance**, i.e.
$$
\begin{aligned}
        J(u)&=\frac{1}{N-1}\sum_{i=1}^N((\vec{x_i}-\overline{x})^Tu)^2 \\
        &=\frac{1}{N-1}\sum_{i=1}^{N-1}u^T(\vec{x_i}-\overline{x})(\vec{x_i}-\overline{x})^Tu \\
        &=\frac{1}{N-1}u_1^T\sum_{i=1}^{N=1}(\vec{x_i}-\overline{x})(\vec{x_i}-\overline{x})^Tu \\
        &=u_1^TSu
    \end{aligned}
$$
Thus, the problem becomes:
$$
\left\{
        \begin{aligned}
            &\hat{u}=\text{argmax}_{u}u^TSu \\
            \\
            &u^Tu=1
        \end{aligned}
    \right.
$$
which could be solved by **Lagrangian Multiplier Method**:
$$
\begin{aligned}
        &\mathcal{L}(u,\lambda)=u^TSu+\lambda(1-u^Tu) \\
        &\dfrac{\partial\mathcal{L}}{\partial{u}}=2Su-2\lambda{u}=0
\end{aligned}
$$
Obviously, $\lambda$ and $u$ is the corresponding **eigenvalue and eigenvector** of $S$. If we want to reduce to dimension $p$, we only need to choose the first p eigenvectors.

## CH4 Transforming Vectors

The objective we do machine translation is to transform the sentence vector in the source language to the target language, i.e.

![image-20210714135315257](.\imgs\image-20210714135315257.png)

To make the translation as precise as possible, we need to minimize
$$
\text{Loss}=||XR-Y||_F
$$
Firstly, we have an initial estimation $R$, to minimize the loss function, we use gradient descent.
$$
R=R-\alpha\frac{d}{dR}\text{Loss}
$$
其中，$\alpha$为学习率。

If we use Frobenius norm, i.e.
$$
||A||_F\equiv\sqrt{\sum_{i=1}^m\sum_{j=1}^n|a_{ij}|^2}
$$
To compute the gradient more easily, we use the square of the norm as the loss function.
$$
\text{Loss}=||XR-Y||_F^2
$$
The gradient becomes:
$$
g=\frac{d}{dR}\text{Loss}=\frac{2}{m}(X^T(XR-Y))
$$
In machine translation, we could have multiple choices. For example, word "hello" can be translated into "bonjour" or "salut".  Thus, the problem becomes to find k nearest words. Actually, to find k nearest words, we do not need to search all the words. If we divide the words into several parts, we could only search in that part.

As a result, we need to divide the words into different buckets. Here comes **Hash**.

## CH5 Hash

Now, let's find the K-nearest neighbor of some country. As stated above, we only need to search those in the same part. Here, we use hash function to divide them into several parts. For example, we can divide 10, 14, 17, 97, 100 into 10 buckets indexed from 0 to 9 by hash function $\mod{10}$:
$$
bucket(x)=x\mod{10}
$$
![image-20210714161028422](.\imgs\image-20210714161028422.png)

But, how to find a hash function for locality? We need **locality sensitive hashing**!

Basically, **locality sensitive hashing** is to divide the localities by **plane**.

![image-20210714162259813](.\imgs\image-20210714162259813.png)

In mathematical expression, we express the localities by vectors.

![image-20210714162509026](.\imgs\image-20210714162509026.png)

To decide which side of the plane the vector is on, let's take a close look at the **dot product** of the vector. Obviously, $V_1$ is on one side of the plane, $V_3$ is on the opposite site, while $V_2$ is on the plane. **The sign of the dot product decides it.**

For multiple planes, we need to combine the hash values together.

![image-20210714163138582](.\imgs\image-20210714163138582.png)

The algorithm is:

```Python
def hash_multiple_plane(P_l,v):
    hash_value = 0
    for i, P in enumerate(P_l):
        sign = side_of_plane(P,v)
        hash_i = 1 if sign >= 0 else 0
        hash_value += 2**i * hash_i
	return hash_value
```

