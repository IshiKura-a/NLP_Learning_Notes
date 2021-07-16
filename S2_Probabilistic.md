# S2 Probabilistic Model

## CH1 Auto-Correct

Objective: correct the misspelled words while maintaining minimum edit distance.

### How it works

1. Identify the misspelled word
2. Find strings n edit distance away
3. Filter candidates
4. Calculate word probabilities

### Identify the misspelled word

```python
if word not in vocab:
    misspelled = True
```

### Find strings n edit distance away

* Edit: an operation performed on a string to change it.
  * **insert**: add a letter
  * **delete**: remove a letter
  * **switch**: swap 2 adjacent letters
  * **replace**: change 1 letter to another

Here, we find available strings rather than words.

### Filter candidates

Here, we filter the strings that are not words.

![image-20210715202248833](.\imgs\image-20210715202248833.png)

### Calculate word probabilities

![image-20210715202930547](.\imgs\image-20210715202930547.png)

### Minimum Distance

For each operations, the edit cost varies, for example insert cost is only 1, while replace cost is 2. To transform from "play" to "stay", the cost is 4.

![image-20210715205619051](.\imgs\image-20210715205619051.png)

To compute the minimum edit distance in a formal way, we need a minimum edit distance matrix $D$ for the two words.

For example, we edit "play" to "stay". Let's denote the start of the word with '#',  The cell $D[i,j]$ means the minimum edit distance from string $\text{source}[:i]$, to string $\text{target}[:j]$.

![image-20210715211441727](D:\Mydata\Science\NLP\WED_Notes\imgs\image-20210715211441727.png)

Obviously, the cell can be computed by **dynamic programming**.

Let's start from the easiest case. For $D[i,j]$ and $D[i-1,j]$, we only need to delete the last word, that is
$$
D[i,j]=D[i-1,j]+del\_cost
$$
For $D[i,j]$ and $D[i,j-1]$, we only need to insert the last word:
$$
D[i,j]=D[i,j-1]+ins\_cost
$$
For $D[i,j]$ and $D[i-1,j-1]$, we only need to replace the last word:
$$
D[i,j]=D[i-1,j-1]+\left\{
\begin{aligned}
&rep\_cost&,\text{src}[i]\ne\text{tar}[i] \\
&0&,\text{src}[i]=\text{tar}[i] \\
\end{aligned}
\right.
$$
Mind that we need the minimum cost, hence we get the state transition equation:
$$
D[i,j]=\left\{
\begin{aligned}
&D[i-1,j]+del\_cost \\
&D[i,j-1]+ins\_cost \\
&D[i-1,j-1]+\left\{
\begin{aligned}
&rep\_cost&,\text{src}[i]\ne\text{tar}[i] \\
&0&,\text{src}[i]=\text{tar}[i] \\
\end{aligned}
\right.
\end{aligned}
\right.
$$
The result is

![image-20210715213823920](.\imgs\image-20210715213823920.png)

## CH2 Part of Speech Tagging

Part of speech tagging is to mark the words with different tags according to their property.

![image-20210716141632317](.\imgs\image-20210716141632317.png)

### Markov Chain

What is the next word after learn ?

![image-20210716142335843](D:\Mydata\Science\NLP\WED_Notes\imgs\image-20210716142335843.png)

Certainly, it's more likely to have a noun after "learn" than a verb. We can represent the likelihood by the graph below(called **Markov Chain**).

![image-20210716142636985](.\imgs\image-20210716142636985.png)

The most important part of **Markov Chain** is the state $Q=\{q_1,q_2,\cdots,q_n\}$. In the graph above, $Q=\{verb,noun\}$.

Another part of the **Markov Chain** is the **transition probabilities**, i.e. the arrow and the number on it. Transition probabilities implies the probability to transform from one state to another state. We can use a transition matrix to record the transition probabilities:

![image-20210716153146604](.\imgs\image-20210716153146604.png)

### Hidden Markov Model

Hidden Markov Model (HMM) introduce states that are hidden, or cannot directly observed. For example, the graph above has states like NN and VB. These states are hidden states for they cannot be observed by the data.

![image-20210716154540840](.\imgs\image-20210716154540840.png)

Generally, the words "jump", "run" that can be observed by the context is observable, while the tags like "verb" which cannot be observed are hidden states. We add a initial state $\pi$,

![image-20210716154801377](.\imgs\image-20210716154801377.png)

Hence, the transition probabilities can be represented by matrix $A$ which is $N+1$ by $N$. In HMM, we have an additional probability called **Emission Probability**, which is the probability to transform from a hidden state to an observable.

![image-20210716155044229](.\imgs\image-20210716155044229.png)

Accordingly, there's a matrix $B$ which is $N$ by $V$ to represent it because a word like "back" may have multiple properties.

How to calculate the probabilities?

1. Count occurrences of tag pairs

$$
C(t_{i-1},t_i)
$$

2. Calculate probabilities using counts

$$
P(t_i|t_{i-1})=\frac{C(t_{i-1},{t_i})}{\sum_{j=1}^NC(t_{i-1},t_j)}
$$

​	In case all the counts are zero, we add an $\epsilon$ to each count for **smoothing**, that's
$$
P(t_i|t_{i-1})=\frac{C(t_{i-1},{t_i})+\epsilon}{\sum_{j=1}^NC(t_{i-1},t_j)+N\epsilon}
$$

### Viterbi Algorithm

![image-20210716160922144](.\imgs\image-20210716160922144.png)

To pass a sentence "I love to learn", we may have such a sequence $\pi\rightarrow{O}\rightarrow{VB}\rightarrow{O}\rightarrow{VB}$ with probability 0.0003. The Viterbi Algorithm actually computes several such passes at the same time to find the most likely sequence. It can be expressed into 3 steps:

1. Initialization step.
2. Forward pass.
3. Backward pass.

During the steps, we need two auxiliary matrices $C$ and $D$. 

#### Step 1 

In the first column of $C$, we calculate the probability from $\pi$ to word $w_1$, that is
$$
c_{i,1}=\pi_i\times{b_{i,cindex(w_1)}}=a_{1,i}\times{b_{i,cindex(w_1)}}
$$
In the $D$ matrix, we store the labels that represent the different state we are traversing when finding the most likely sequence of POS tags for the given words $w_1$ to $w_k$. In the first column, we simply set $d_{i,1}=0$ as there're no preceding POS tags we have traversed.

 #### Step 2

In the forward pass
$$
\begin{aligned}
c_{i,j}&=\max_{k}c_{k,j-1}\times{a}_{k,i}\times{b}_{i,cindex{(w_j)}} \\
d_{i,j}&=\arg\max_{k}c_{k,j-1}\times{a}_{k,i}\times{b}_{i,cindex{(w_j)}} \\
\end{aligned}
$$

#### Step 3

Using the matrix $C$, we can get
$$
s=\arg\max_{i}c_{i,K}
$$
For example, $s=1$, and for the matrix $D$ below, we can restore the sequence:

![image-20210716163400544](.\imgs\image-20210716163400544.png)

The algorithm stops when we find the start token.

> In case the probabilities are very small, use log probabilities instead.

