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
&D[i,j]=D[i,j-1]+ins\_cost \\
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