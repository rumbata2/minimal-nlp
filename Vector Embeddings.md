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
    <summary>doc1.txt</summary>

>Cooking acids tend to be mellow, transforming the foods with which they are cooked slowly, over time.
They can be extraordinarily subtle; while their presence may go undetected, their absence is sharply felt.
I learned this painful lesson when at the request of a distant relative, I tried to make beef bourguignon without the Bourgogne in Iran,
where wine isn’t readily available. No matter what I did, I couldn’t get the dish to taste right without that crucial ingredient.
>
>Give acid the time it needs to do its silent work when macerating shallots and onions.
Macerate, from Latin, “to soften,” refers to the process whereby ingredients soak in some form of acid — 
usually vinegar or citrus juice—to soften their harshness. Simply coat the shallots or onions in acid—they don’t need to be completely submerged.
If you plan on using a couple of tablespoons of vinegar for a dressing, just coat the shallots with it first, 
and wait 15 or 20 minutes before adding oil to build the dressing in the same cup or bowl. It will be enough to prevent dragon breath.
>
>There’s no replacement for working acid early into braises and stews; the remarkable alchemy of time and heat will soften any dish’s sharp edges.
Omit the tomatoes and beer from Pork Braised with Chillies and the sweetness of the aromatic base of onions and garlic will dominate.
The sweetness resulting from browning needs the foil of acid, too. 
Deglazing a pan with wine, whether for risotto, pork chops, fish fillets, or a more complex reduction sauce will keep a dish from skewing too sweet.
</details>

<details open>
    <summary>doc2.txt</summary>

>Tyler gets me a job as a waiter, after that Tyler's pushing a gun in my mouth and saying, the first step to eternal life is you have to die. 
For a long time though, Tyler and I were best friends. People are always asking, did I know about Tyler Durden.
>
>The barrel of the gun pressed against the back of my throat, Tyler says 'We really won't die.'
>
>With my tongue I can feel the silencer holes we drilled into the barrel of the gun. Most of the noise a gunshot makes is expanding gases,
and there's the tiny sonic boom a bullet makes because it travels so fast. To make a silencer, you just drill holes in the barrel of the gun, 
a lot of holes. This lets the gas escape and slows the bullet to below the speed of sound.
>
>You drill the holes wrong and the gun will blow off your hand.
>
>'This isn't really death,' Tyler says. 'We'll be legend. We won't grow >old.'
>
>I tongue the barrel into my cheek and say, Tyler, you're thinking of vampires.
>
>The building we're standing on won't be here in ten minutes. You take a 98% concentration of fuming nitric acid and add the acid
to three times that amount of sulfuric acid. Do this in an ice bath. Then add glycerin drop-by-drop with an eye dropper. You have nitroglycerin.
I know this because Tyler knows this.
>
>Mix the nitro with sawdust, and you have a nice plastic explosive. A lot of folks mix their nitro with cotton and add Epsom salts as a sulfate. 
This works too. Some folks, they use paraffin mixed with nitro. Paraffin has never, ever worked for me.
>
>So Tyler and I are on top of the Parker-Morris Building with the gun stuck in my mouth, and we hear glass breaking. Look over the edge. 
It's a cloudy day, even this high up. This is the world's tallest building, and this high up the wind is always cold. It's so quiet this high up,
the feeling you get is that you're one of those space monkeys. You do the little job you're trained to do.
Pull a lever.
Push a button.
You don't understand any of it, and then you just die.

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
    <summary>3 sample rows of the term-document matrix, no tf-idf (mini version)</summary>

|       | doc1 | doc2 |
|------:|-----:|-----:|
|  acid |    5 |    3 |
|  dish |    3 |    0 |
| tyler |    0 |    9 |

</details>

<details>
    <summary>The term-document matrix, no tf-idf (full version)</summary>

|                 | doc1 | doc2 |
|----------------:|-----:|-----:|
| 15              |    1 |    0 |
| 20              |    1 |    0 |
| 98              |    0 |    1 |
| about           |    0 |    1 |
| absence         | 1    | 0    |
| acid            | 5    | 3    |
| acids           | 1    | 0    |
| add             | 0    | 3    |
| adding          | 1    | 0    |
| after           | 0    | 1    |
| against         | 0    | 1    |
| alchemy         | 1    | 0    |
| always          | 0    | 2    |
| amount          | 0    | 1    |
| an              | 0    | 2    |
| and             | 7    | 13   |
| any             | 1    | 1    |
| are             | 1    | 2    |
| aromatic        | 1    | 0    |
| as              | 0    | 2    |
| asking          | 0    | 1    |
| at              | 1    | 0    |
| available       | 1    | 0    |
| back            | 0    | 1    |
| barrel          | 0    | 4    |
| base            | 1    | 0    |
| bath            | 0    | 1    |
| be              | 4    | 2    |
| because         | 0    | 2    |
| beef            | 1    | 0    |
| beer            | 1    | 0    |
| before          | 1    | 0    |
| below           | 0    | 1    |
| best            | 0    | 1    |
| blow            | 0    | 1    |
| boom            | 0    | 1    |
| bourgogne       | 1    | 0    |
| bourguignon     | 1    | 0    |
| bowl            | 1    | 0    |
| braised         | 1    | 0    |
| braises         | 1    | 0    |
| breaking        | 0    | 1    |
| breath          | 1    | 0    |
| browning        | 1    | 0    |
| build           | 1    | 0    |
| building        | 0    | 3    |
| bullet          | 0    | 2    |
| button          | 0    | 1    |
| by              | 0    | 1    |
| can             | 1    | 1    |
| cheek           | 0    | 1    |
| chillies        | 1    | 0    |
| chops           | 1    | 0    |
| citrus          | 1    | 0    |
| cloudy          | 0    | 1    |
| coat            | 2    | 0    |
| cold            | 0    | 1    |
| completely      | 1    | 0    |
| complex         | 1    | 0    |
| concentration   | 0    | 1    |
| cooked          | 1    | 0    |
| cooking         | 1    | 0    |
| cotton          | 0    | 1    |
| couldn          | 1    | 0    |
| couple          | 1    | 0    |
| crucial         | 1    | 0    |
| cup             | 1    | 0    |
| day             | 0    | 1    |
| death           | 0    | 1    |
| deglazing       | 1    | 0    |
| did             | 1    | 1    |
| die             | 0    | 3    |
| dish            | 3    | 0    |
| distant         | 1    | 0    |
| do              | 1    | 3    |
| dominate        | 1    | 0    |
| don             | 1    | 1    |
| dragon          | 1    | 0    |
| dressing        | 2    | 0    |
| drill           | 0    | 2    |
| drilled         | 0    | 1    |
| drop            | 0    | 2    |
| dropper         | 0    | 1    |
| durden          | 0    | 1    |
| early           | 1    | 0    |
| edge            | 0    | 1    |
| edges           | 1    | 0    |
| enough          | 1    | 0    |
| epsom           | 0    | 1    |
| escape          | 0    | 1    |
| eternal         | 0    | 1    |
| even            | 0    | 1    |
| ever            | 0    | 1    |
| expanding       | 0    | 1    |
| explosive       | 0    | 1    |
| extraordinarily | 1    | 0    |
| eye             | 0    | 1    |
| fast            | 0    | 1    |
| feel            | 0    | 1    |
| feeling         | 0    | 1    |
| felt            | 1    | 0    |
| fillets         | 1    | 0    |
| first           | 1    | 1    |
| fish            | 1    | 0    |
| foil            | 1    | 0    |
| folks           | 0    | 2    |
| foods           | 1    | 0    |
| for             | 3    | 2    |
| form            | 1    | 0    |
| friends         | 0    | 1    |
| from            | 4    | 0    |
| fuming          | 0    | 1    |
| garlic          | 1    | 0    |
| gas             | 0    | 1    |
| gases           | 0    | 1    |
| get             | 1    | 1    |
| gets            | 0    | 1    |
| give            | 1    | 0    |
| glass           | 0    | 1    |
| glycerin        | 0    | 1    |
| go              | 1    | 0    |
| grow            | 0    | 1    |
| gun             | 0    | 6    |
| gunshot         | 0    | 1    |
| hand            | 0    | 1    |
| harshness       | 1    | 0    |
| has             | 0    | 1    |
| have            | 0    | 3    |
| hear            | 0    | 1    |
| heat            | 1    | 0    |
| here            | 0    | 1    |
| high            | 0    | 3    |
| holes           | 0    | 4    |
| ice             | 0    | 1    |
| if              | 1    | 0    |
| in              | 4    | 5    |
| ingredient      | 1    | 0    |
| ingredients     | 1    | 0    |
| into            | 1    | 2    |
| iran            | 1    | 0    |
| is              | 1    | 5    |
| isn             | 1    | 1    |
| it              | 3    | 4    |
| its             | 1    | 0    |
| job             | 0    | 2    |
| juice           | 1    | 0    |
| just            | 1    | 2    |
| keep            | 1    | 0    |
| know            | 0    | 2    |
| knows           | 0    | 1    |
| latin           | 1    | 0    |
| learned         | 1    | 0    |
| legend          | 0    | 1    |
| lesson          | 1    | 0    |
| lets            | 0    | 1    |
| lever           | 0    | 1    |
| life            | 0    | 1    |
| little          | 0    | 1    |
| ll              | 0    | 1    |
| long            | 0    | 1    |
| look            | 0    | 1    |
| lot             | 0    | 2    |
| macerate        | 1    | 0    |
| macerating      | 1    | 0    |
| make            | 1    | 1    |
| makes           | 0    | 2    |
| matter          | 1    | 0    |
| may             | 1    | 0    |
| me              | 0    | 2    |
| mellow          | 1    | 0    |
| minutes         | 1    | 1    |
| mix             | 0    | 2    |
| mixed           | 0    | 1    |
| monkeys         | 0    | 1    |
| more            | 1    | 0    |
| morris          | 0    | 1    |
| most            | 0    | 1    |
| mouth           | 0    | 2    |
| my              | 0    | 5    |
| need            | 1    | 0    |
| needs           | 2    | 0    |
| never           | 0    | 1    |
| nice            | 0    | 1    |
| nitric          | 0    | 1    |
| nitro           | 0    | 3    |
| nitroglycerin   | 0    | 1    |
| no              | 2    | 0    |
| noise           | 0    | 1    |
| of              | 8    | 14   |
| off             | 0    | 1    |
| oil             | 1    | 0    |
| old             | 0    | 1    |
| omit            | 1    | 0    |
| on              | 1    | 2    |
| one             | 0    | 1    |
| onions          | 3    | 0    |
| or              | 5    | 0    |
| over            | 1    | 1    |
| painful         | 1    | 0    |
| pan             | 1    | 0    |
| paraffin        | 0    | 2    |
| parker          | 0    | 1    |
| people          | 0    | 1    |
| plan            | 1    | 0    |
| plastic         | 0    | 1    |
| pork            | 2    | 0    |
| presence        | 1    | 0    |
| pressed         | 0    | 1    |
| prevent         | 1    | 0    |
| process         | 1    | 0    |
| pull            | 0    | 1    |
| push            | 0    | 1    |
| pushing         | 0    | 1    |
| quiet           | 0    | 1    |
| re              | 0    | 4    |
| readily         | 1    | 0    |
| really          | 0    | 2    |
| reduction       | 1    | 0    |
| refers          | 1    | 0    |
| relative        | 1    | 0    |
| remarkable      | 1    | 0    |
| replacement     | 1    | 0    |
| request         | 1    | 0    |
| resulting       | 1    | 0    |
| right           | 1    | 0    |
| risotto         | 1    | 0    |
| salts           | 0    | 1    |
| same            | 1    | 0    |
| sauce           | 1    | 0    |
| sawdust         | 0    | 1    |
| say             | 0    | 1    |
| saying          | 0    | 1    |
| says            | 0    | 2    |
| shallots        | 3    | 0    |
| sharp           | 1    | 0    |
| sharply         | 1    | 0    |
| silencer        | 0    | 2    |
| silent          | 1    | 0    |
| simply          | 1    | 0    |
| skewing         | 1    | 0    |
| slowly          | 1    | 0    |
| slows           | 0    | 1    |
| so              | 0    | 3    |
| soak            | 1    | 0    |
| soften          | 3    | 0    |
| some            | 1    | 1    |
| sonic           | 0    | 1    |
| sound           | 0    | 1    |
| space           | 0    | 1    |
| speed           | 0    | 1    |
| standing        | 0    | 1    |
| step            | 0    | 1    |
| stews           | 1    | 0    |
| stuck           | 0    | 1    |
| submerged       | 1    | 0    |
| subtle          | 1    | 0    |
| sulfate         | 0    | 1    |
| sulfuric        | 0    | 1    |
| sweet           | 1    | 0    |
| sweetness       | 2    | 0    |
| tablespoons     | 1    | 0    |
| take            | 0    | 1    |
| tallest         | 0    | 1    |
| taste           | 1    | 0    |
| ten             | 0    | 1    |
| tend            | 1    | 0    |
| that            | 1    | 3    |
| the             | 16   | 27   |
| their           | 3    | 1    |
| then            | 0    | 2    |
| there           | 1    | 1    |
| they            | 3    | 1    |
| thinking        | 0    | 1    |
| this            | 1    | 10   |
| those           | 0    | 1    |
| though          | 0    | 1    |
| three           | 0    | 1    |
| throat          | 0    | 1    |
| time            | 3    | 1    |
| times           | 0    | 1    |
| tiny            | 0    | 1    |
| to              | 10   | 6    |
| tomatoes        | 1    | 0    |
| tongue          | 0    | 2    |
| too             | 2    | 1    |
| top             | 0    | 1    |
| trained         | 0    | 1    |
| transforming    | 1    | 0    |
| travels         | 0    | 1    |
| tried           | 1    | 0    |
| tyler           | 0    | 9    |
| understand      | 0    | 1    |
| undetected      | 1    | 0    |
| up              | 0    | 3    |
| use             | 0    | 1    |
| using           | 1    | 0    |
| usually         | 1    | 0    |
| vampires        | 0    | 1    |
| vinegar         | 2    | 0    |
| wait            | 1    | 0    |
| waiter          | 0    | 1    |
| we              | 0    | 6    |
| were            | 0    | 1    |
| what            | 1    | 0    |
| when            | 2    | 0    |
| where           | 1    | 0    |
| whereby         | 1    | 0    |
| whether         | 1    | 0    |
| which           | 1    | 0    |
| while           | 1    | 0    |
| will            | 4    | 1    |
| wind            | 0    | 1    |
| wine            | 2    | 0    |
| with            | 4    | 6    |
| without         | 2    | 0    |
| won             | 0    | 3    |
| work            | 1    | 0    |
| worked          | 0    | 1    |
| working         | 1    | 0    |
| works           | 0    | 1    |
| world           | 0    | 1    |
| wrong           | 0    | 1    |
| you             | 1    | 13   |
| your            | 0    | 1    |

</details>

The vector rows in the matrix are the word (term) embeddings and the vector columns are the document embeddings.  
- The word "acid" is now the vector $[5, 3]^T$.
- In the mini version, doc1 and doc2 are now respectively $[5, 3, 0]^T$ and $[3, 0, 9]^T$.
- In the full version, doc1 and doc2 are the two column vectors of size $324$.

In this case, the **vocabulary** is the set of all $324$ words seen in at least one document, though in a more general context the vocabulary may be chosen by us explicitly or by heuristic rules - if the documents include words not in the vocabulary, we may include an $\langle UNKNOWN \rangle$ term instead to capture all of them.

A general term-document matrix will then be of dimension $|V| \times N_{doc}$, where $V$ is the vocabulary and $N_{doc}$ is the number of documents in the corpus.


Now that we captured the notion of context with linear algebra, we gain access to [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) as a tool to measure similarity between two words or between two documents. 
We recall the definition $$cos(\textbf{a}, \textbf{b}) = \dfrac{\textbf{a} \cdot \textbf{b}}{|\textbf{a}||\textbf{b}|}$$
and use it to make formal comparisons:
- $cos(\text{"dish", "tyler"}) = cos([3, 0]^T, [0, 9]^T) = 0.$
- $cos("dish", "acid") = cos([3, 0]^T, [5, 3]^T) \approx 0.857$
- $cos("tyler", "acid") = cos([0, 9]^T, [5, 3]^T) \approx 0.514$

The words "dish" and "tyler" have no similarity since they appear in entirely different contexts;

"dish" and "acid" have strong similarity since "acid" appears in both documents (but more in the first one) and "dish" appears only in the first one;

"tyler" and "acid" are somewhat similar because "acid" appears in both documents, however it appears less in the second one.