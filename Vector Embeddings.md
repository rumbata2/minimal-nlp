In natural language, the syntax of different words may not at all be related to their semantics.
<details>
    <summary>Examples (optional)</summary>  

- *beautiful* and *attractive* have mostly the same meaning
- *beautiful* and *ugly* have mostly the opposite meaning
- *cat* and *dog* are not the same but they have \*some\* similarity
- *basketball* and *hoop* are not the same but they belong to the same [semantic field](https://en.wikipedia.org/wiki/Semantic_field)
</details>  


In order to capture meaning, we use **vector embeddings** - words will be points in the euclidean space $\mathbb{R}^n$. Their meaning will be inferred from the context in which they occur - the **documents** they belong to, and/or around the words around them.

<details>
    <summary>What is a document?</summary>

A document is a piece of text, one element of a dataset (corpus). For statistical learning purposes, documents are often annotated with additional data, for example the category they belong to. Sample datasets may then look like:  
- $\\{(d_1, c_1), (d_2, c_1), (d_3, c_2), (d_4, c_2), (d_5, c_2)\\}$ 
- $\\{(\text{"this movie sucks"}, \text{negative}), (\text{"Breathtaking from start to end!"}, \text{positive}) \\}$

and so on.
</details>
<p></p>

Consider the following documents:

<details open>
    <summary>doc1.txt (shortened)</summary>

>Cooking acids tend to be mellow, transforming the foods with which they are cooked slowly, over time.
They can be extraordinarily subtle; while their presence may go undetected, their absence is sharply felt.
I learned this painful lesson when at the request of a distant relative, I tried to make beef bourguignon without the Bourgogne in Iran,
where wine isn’t readily available. No matter what I did, I couldn’t get the dish to taste right without that crucial ingredient.
</details>

<details open>
    <summary>doc2.txt (shortened)</summary>

>Tyler gets me a job as a waiter, after that Tyler's pushing a gun in my mouth and saying, the first step to eternal life is you have to die. 
For a long time though, Tyler and I were best friends. People are always asking, did I know about Tyler Durden.
>
>The barrel of the gun pressed against the back of my throat, Tyler says 'We really won't die.'
</details>
<br/>

We define a **term-document matrix** - words are rows and documents are columns (or vice versa). Each entry is a scalar. We choose one of two options to give the words a "weight". 
- $count(t,d)$, which counts how many times the term (word) $t$ appears in the document $d$.
- **tf-idf** weight: $`\underbrace{log_{10}(count(t,d) + 1)}_{tf(t,d)} \hspace{1mm} * \enspace \underbrace{log_{10}\left(\frac{N}{df(t)}\right)}_{idf(t)}`$  
where
  - $tf(t,d)$ is the **term frequency** of the word $t$ with respect to document $d$.  
  We take the logarithm of the count to induce diminishing returns on its importance. Since we can not take the logarithm of $0$, we add $1$ beforehand.
  - $df(t)$ is the **document frequency** of the word $t$ - the number of documents it occurs in. Note that the minimum is $1$ and the maximum is $N$, where $N$ is the total number of documents.
  - $idf(t)$ is the **inverse document frequency** of the word $t$. If a term appears in a small part of all documents, it is considered more representative of the documents it resides in (hence this weight grows) and vice versa. Again we take the base $10$ logarithm for diminishing returns purposes.

Let's assume our corpus is made up of the above 2 documents. We construct the term-document matrix for this corpus:


<details open>
    <summary>4 sample rows of the term-document matrix, no tf-idf (mini version)</summary>

|       | doc1 | doc2 |
|------:|-----:|-----:|
|  dish |    1 |    0 |
| acids |    1 |    0 |
| tyler |    0 |    5 |
|   gun |    0 |    2 |

</details>

<details>
    <summary>The term-document matrix, no tf-idf (full version)</summary>

|                 | doc1 | doc2 |
|-----------------|------|------|
| about           | 0    | 1    |
| absence         | 1    | 0    |
| acids           | 1    | 0    |
| after           | 0    | 1    |
| against         | 0    | 1    |
| always          | 0    | 1    |
| and             | 0    | 2    |
| are             | 1    | 1    |
| as              | 0    | 1    |
| asking          | 0    | 1    |
| at              | 1    | 0    |
| available       | 1    | 0    |
| back            | 0    | 1    |
| barrel          | 0    | 1    |
| be              | 2    | 0    |
| beef            | 1    | 0    |
| best            | 0    | 1    |
| bourgogne       | 1    | 0    |
| bourguignon     | 1    | 0    |
| can             | 1    | 0    |
| cooked          | 1    | 0    |
| cooking         | 1    | 0    |
| couldn          | 1    | 0    |
| crucial         | 1    | 0    |
| did             | 1    | 1    |
| die             | 0    | 2    |
| dish            | 1    | 0    |
| distant         | 1    | 0    |
| durden          | 0    | 1    |
| eternal         | 0    | 1    |
| extraordinarily | 1    | 0    |
| felt            | 1    | 0    |
| first           | 0    | 1    |
| foods           | 1    | 0    |
| for             | 0    | 1    |
| friends         | 0    | 1    |
| get             | 1    | 0    |
| gets            | 0    | 1    |
| go              | 1    | 0    |
| gun             | 0    | 2    |
| have            | 0    | 1    |
| in              | 1    | 1    |
| ingredient      | 1    | 0    |
| iran            | 1    | 0    |
| is              | 1    | 1    |
| isn             | 1    | 0    |
| job             | 0    | 1    |
| know            | 0    | 1    |
| learned         | 1    | 0    |
| lesson          | 1    | 0    |
| life            | 0    | 1    |
| long            | 0    | 1    |
| make            | 1    | 0    |
| matter          | 1    | 0    |
| may             | 1    | 0    |
| me              | 0    | 1    |
| mellow          | 1    | 0    |
| mouth           | 0    | 1    |
| my              | 0    | 2    |
| no              | 1    | 0    |
| of              | 1    | 2    |
| over            | 1    | 0    |
| painful         | 1    | 0    |
| people          | 0    | 1    |
| presence        | 1    | 0    |
| pressed         | 0    | 1    |
| pushing         | 0    | 1    |
| readily         | 1    | 0    |
| really          | 0    | 1    |
| relative        | 1    | 0    |
| request         | 1    | 0    |
| right           | 1    | 0    |
| saying          | 0    | 1    |
| says            | 0    | 1    |
| sharply         | 1    | 0    |
| slowly          | 1    | 0    |
| step            | 0    | 1    |
| subtle          | 1    | 0    |
| taste           | 1    | 0    |
| tend            | 1    | 0    |
| that            | 1    | 1    |
| the             | 4    | 4    |
| their           | 2    | 0    |
| they            | 2    | 0    |
| this            | 1    | 0    |
| though          | 0    | 1    |
| throat          | 0    | 1    |
| time            | 1    | 1    |
| to              | 3    | 2    |
| transforming    | 1    | 0    |
| tried           | 1    | 0    |
| tyler           | 0    | 5    |
| undetected      | 1    | 0    |
| waiter          | 0    | 1    |
| we              | 0    | 1    |
| were            | 0    | 1    |
| what            | 1    | 0    |
| when            | 1    | 0    |
| where           | 1    | 0    |
| which           | 1    | 0    |
| while           | 1    | 0    |
| wine            | 1    | 0    |
| with            | 1    | 0    |
| without         | 2    | 0    |
| won             | 0    | 1    |
| you             | 0    | 1    |

</details>

The vector rows in the matrix are the word (term) embeddings and the vector columns are the document embeddings.  
- The word "acids" is now the vector $[1, 0]^T$, while the word "tyler" is now the vector $[0, 5]^T$.
- In the mini version, doc1 and doc2 are now respectively $[1, 1, 0, 0]^T$ and $[0, 0, 5, 2]^T$.
- In the full version, doc1 and doc2 are the two column vectors of size $106$.

In this case, the **vocabulary** is the set of all $106$ words seen in at least one document, though in a more general context the vocabulary may be chosen by us explicitly or by heuristic rules - if the documents include words not in the vocabulary, we may include an $\langle UNKNOWN \rangle$ term instead to capture all of them.

A general term-document matrix will then be of dimension $|V| \times N_{doc}$, where $V$ is the vocabulary and $N_{doc}$ is the number of documents in the corpus.

<details>
    <summary>Intuition</summary>

As we said, word semantics will be (loosely) defined through the context in which they appear. Since "dish" and "acids" both appear only in the cooking document (in this case the same amount of times), their vector embeddings are very similar (in this case the same).  

In the same vein, the (mini version) vector for the Fight Club document $[0, 0, 5, 2]^T$ contains "tyler" 5 times and "gun" 2 times. This means that from now on, when we want to see if some other document is also a Fight Club excerpt, we will want its vector to look somewhat similar to this one - it should contain words like "tyler" a lot more than cooking related words.

</details>


We now have [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) as a tool to measure similarity between two words or between two vectors. 
We recall the definition $$cos(\textbf{a}, \textbf{b}) = \dfrac{\textbf{a} \cdot \textbf{b}}{|\textbf{a}||\textbf{b}|}$$
