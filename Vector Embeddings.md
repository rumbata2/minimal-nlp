In natural language, the syntax of different words may not at all be related to their semantics:
- *beautiful* and *attractive* have mostly the same meaning
- *beautiful* and *ugly* have mostly the opposite meaning
- *cat* and *dog* are not the same but they have \*some\* similarity
- *basketball* and *hoop* are not the same but they belong to the same [semantic field](https://en.wikipedia.org/wiki/Semantic_field)

In order to capture meaning, we use **vector embeddings** - words will be points in the euclidean space $\mathbb{R}^n$. Their meaning will be inferred from the context in which they occur - the **documents** they belong to, and/or around the words around them.

<details>
    <summary>What is a document?</summary>

A document is a piece of text, one element of a dataset (corpus). For statistical learning purposes, documents are often annotated with additional data, for example the category they belong to. Sample datasets may then look like:  
- $\{(d_1, c_1), (d_2, c_1), (d_3, c_2), (d_4, c_2), (d_5, c_2)\}$ 
- $\{(\text{"this movie sucks"}, \text{negative}), (\text{"Breathtaking from start to end!"}, \text{positive}) \}$.  

and so on.
</details>
<p></p>

Consider the following documents:

<details>
    <summary>doc1.txt (shortened)</summary>

Cooking acids tend to be mellow, transforming the foods with which they are cooked slowly, over time.
They can be extraordinarily subtle; while their presence may go undetected, their absence is sharply felt.
I learned this painful lesson when at the request of a distant relative, I tried to make beef bourguignon without the Bourgogne in Iran,
where wine isn’t readily available. No matter what I did, I couldn’t get the dish to taste right without that crucial ingredient.
</details>

<details>
    <summary>doc2.txt (shortened)</summary>

Tyler gets me a job as a waiter, after that Tyler's pushing a gun in my mouth and saying, the first step to eternal life is you have to die. 
For a long time though, Tyler and I were best friends. People are always asking, did I know about Tyler Durden.

The barrel of the gun pressed against the back of my throat, Tyler says 'We really won't die.'
</details>

We define a **term-document matrix** - words are rows and documents are columns (or vice versa). Each entry is a scalar. We choose one of several options to give the words a "weight". In roughly ascending sophistication:
- $count(t,d)$, which counts how many times the term (word) $t$ appears in the document $d$.
- **tf-idf** weight: $\underbrace{log_{10}(count(t,d) + 1)}_{tf_{t,d}} \enspace * \enspace \underbrace{log_{10}\left(\frac{N}{df_t}\right)}_{idf_t}$  
- **tf-idf** weight: $log_{10}(count(t,d) + 1) * 
\left(\frac{N}{df_t}\right)$  
- **tf-idf** weight: $\underbrace{tf_{t,d}}_{tf(t,d)} \enspace * \enspace idf_t$





