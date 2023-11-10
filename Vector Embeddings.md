In natural language, the syntax of different words may not at all be related to their semantics:
- *beautiful* and *attractive* have mostly the same meaning
- *beautiful* and *ugly* have mostly the opposite meaning
- *cat* and *dog* are not the same but they have semantic similarities
- *basketball* and *hoop* are not the same but they belong to the same [semantic field](https://en.wikipedia.org/wiki/Semantic_field)

In order to capture meaning, we use **vector embeddings** - words will be points in the euclidean space $\mathbb{R}^n$.

Consider the following texts, often called **documents**:

<details>
    <summary>doc1.txt</summary>

Cooking acids tend to be mellow, transforming the foods with which they are cooked slowly, over time.
They can be extraordinarily subtle; while their presence may go undetected, their absence is sharply felt.
I learned this painful lesson when at the request of a distant relative, I tried to make beef bourguignon without the Bourgogne in Iran,
where wine isn’t readily available. No matter what I did, I couldn’t get the dish to taste right without that crucial ingredient.

Give acid the time it needs to do its silent work when macerating shallots and onions.
Macerate, from Latin, “to soften,” refers to the process whereby ingredients soak in some form of acid — 
usually vinegar or citrus juice—to soften their harshness. Simply coat the shallots or onions in acid—they don’t need to be completely submerged.
If you plan on using a couple of tablespoons of vinegar for a dressing, just coat the shallots with it first, 
and wait 15 or 20 minutes before adding oil to build the dressing in the same cup or bowl. It will be enough to prevent dragon breath.

There’s no replacement for working acid early into braises and stews; the remarkable alchemy of time and heat will soften any dish’s sharp edges.
Omit the tomatoes and beer from Pork Braised with Chillies and the sweetness of the aromatic base of onions and garlic will dominate.
The sweetness resulting from browning needs the foil of acid, too. 
Deglazing a pan with wine, whether for risotto, pork chops, fish fillets, or a more complex reduction sauce will keep a dish from skewing too sweet.

</details>

<details>
    <summary>doc2.txt</summary>

Tyler gets me a job as a waiter, after that Tyler's pushing a gun in my mouth and saying, the first step to eternal life is you have to die. 
For a long time though, Tyler and I were best friends. People are always asking, did I know about Tyler Durden.

The barrel of the gun pressed against the back of my throat, Tyler says 'We really won't die.'

With my tongue I can feel the silencer holes we drilled into the barrel of the gun. Most of the noise a gunshot makes is expanding gases,
and there's the tiny sonic boom a bullet makes because it travels so fast. To make a silencer, you just drill holes in the barrel of the gun, 
a lot of holes. This lets the gas escape and slows the bullet to below the speed of sound.

You drill the holes wrong and the gun will blow off your hand.

'This isn't really death,' Tyler says. 'We'll be legend. We won't grow old.'

I tongue the barrel into my cheek and say, Tyler, you're thinking of vampires.

The building we're standing on won't be here in ten minutes. You take a 98% concentration of fuming nitric acid and add the acid
to three times that amount of sulfuric acid. Do this in an ice bath. Then add glycerin drop-by-drop with an eye dropper. You have nitroglycerin.
I know this because Tyler knows this.

Mix the nitro with sawdust, and you have a nice plastic explosive. A lot of folks mix their nitro with cotton and add Epsom salts as a sulfate. 
This works too. Some folks, they use paraffin mixed with nitro. Paraffin has never, ever worked for me.

So Tyler and I are on top of the Parker-Morris Building with the gun stuck in my mouth, and we hear glass breaking. Look over the edge. 
It's a cloudy day, even this high up. This is the world's tallest building, and this high up the wind is always cold. It's so quiet this high up,
the feeling you get is that you're one of those space monkeys. You do the little job you're trained to do.

Pull a lever.

Push a button.

You don't understand any of it, and then you just die.

</details>


```python

some code
def f(x): 
    return x

```