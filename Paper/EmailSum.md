# EMAILSUM: Abstractive Email Thread Summarization

**Source**: https://arxiv.org/abs/2107.14691

**Background**: It takes a lot time when user forget the main points of previous discussions or he is
newly included in a discussion thread. Therefore, automatically summarizing email threads can improve work efficiency and provides practical benefits.

**Contribution**: develop an abstractive **Email Thread Summarization** (EMAILSUM) dataset,

**Attribute**:

* contains human-annotated short (<30 words) and long (<100 words) summaries of 2,549 email threads (each containing 3 to 10 emails) over a wide variety of topics.

**Example**:

![image-20210804160646927](.\imgs\image-20210804160646927.png)

**Merit**: creation of dataset.

1. Where do the data come from?

   A: **Resort to existing email collections**: Enron (Klimt and Yang, 2004), W3C (Craswell et al., 2006), and Avocado (Oard et al., 2015). But none of them provides explicit thread structure. Therefore, in this section, we will introduce our **email thread preprocessing and summary collection procedures**.

2. Email Thread Preprocessing

   1. We give every email a “**normalized subject**” by removing the reply or forward tags (e.g., “Re:”, “Fwd:”, etc.) from its original subject.
   2. **Group emails** by the normalized subjects and **sort emails** in the same group (i.e. thread) by time-stamp.
   3. **De-duplicate emails in every thread** by sender’s email plus time-stamp;
   4. We traverse emails in every thread in **temporal order** and **cut off the thread** when none of the senders plus receivers of the current email appears in previous emails;
   5. We **filter out** threads that **only contain single repeated content**. 

3. Clean dataset if violates

   * 3 $\le$ the number of emails $\le$ 10;
   * 5 < the number of words in each email < 200
   * 30 < the total number of words < 1000
   * Does not contain non-English (e.g., German) tokens
   * Does not contain reply or forward tags in the subject of the first email.

4. Protect privacy:

   * Only keep first names
   * **Remove threads that have** “password”, “pwd”, “confidential”, etc.
   * Replace email address, physical address, phone number, URL, IP address, local path, and other sensitive numbers with `USERNAME@DOMAIN.COM`, `ADDRESS`, `PHONENUMBER`, `HTTP://LINK`, `IPADDRESS`, `PATH`, and `NUMBER`, respectively.
   * Conduct **an extensive manual quality scan** to make sure that the extracted threads are truly threads (instead of random emails grouped) and properly anonymized.

5. Thread Summary Collection:
   * We collect summary annotations on Amazon Mechanical Turk.
   * Use several quality control strategies:
     * We select annotators that are located in the US, have **an approval rate greater than 97%**, and have at least 10,000 approved HITs
     * During annotation, we **periodically sample summaries**, **manually check**
       **their quality**, and **reject or block poor-quality annotators**
     * After annotation, we randomly **sample 2 examples per annotator** and manually **categorize annotators into “good”, “fair”, and “bad” groups**, then **filter examples written by bad annotators**.
   * Short summary: concise description of what the thread is mainly talking about.
   * Long summary: a narrative of what happens.
   * Important Instructions
     1. Long summary **MUST** be longer than short summary
     2. Summary length **can be dynamically decided** based on the content of the thread
     3. Short summary should be a concise and abstractive description of what the thread is mainly talking about
     4. Long summary can be a narrative of what happens. But do **NOT simply summarize each email separately**. **The summary should be coherent**
     5. It is NOT necessary to summarize every email in the long summary, i.e., it is **OK to skip unimportant ones** and merge similar ones if needed.
     6. You are encouraged to include important sender and/or receiver names in long summary
     7. You are discouraged to copy a lot from emails for both short and long summaries; You are supposed to write in your own words as much as you can
     8. You may find some content are technical. We do NOT expect any background knowledge. Just **focus on the major concerns, decisions, and consensus**.
     9. In the thread, emails are ordered by time. However, **one email does NOT necessarily reply to the previous one**. It can reply to an earlier email OR forward to new receivers. In other words, **the structure is NOT always continuous**, so please be careful when you read.
6. Final Dataset:
   * 2,549 email threads each with a long and a short summary.
   * We randomly sample **500** examples **from the “good” annotator group** as our **testing set** and split the remaining examples into **training (1,800 threads)** and **development (249 threads) sets**.

