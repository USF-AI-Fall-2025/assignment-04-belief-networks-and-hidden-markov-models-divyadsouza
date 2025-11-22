# Belief-Networks-Hidden-Markov-Models
Fall 2025 CS 362/562

### Reflections:

**❄️ Give an example of a word which was correctly spelled by the user, but which was incorrectly “corrected” by the algorithm. Why did this happen?**
- "adult" gets changed to "adule". The way this spell checker works is solely dependent on emmission and transition probabilities. The emission probability tells us that if a letter was typed, what is the probability it could be the letter itself or any other letter, the transition probability tells us that based on the previous letter, what is the probability of the current letter being correct. This, along with a relatively small training dataset, gives us a model that makes assumptions that are not always correct. For this case, our model has learned that `a -> d -> u -> l -> e -> END` is more probable than `a -> d -> u -> l -> t -> END`. We can confirm this by typing in similar words like "smelt" which changes to "smele", "dealt" is changed to "deale", "salt" is changed to "sale". 

**❄️ Give an example of a word which was incorrectly spelled by the user, but which was still incorrectly “corrected” by the algorithm. Why did this happen?**
- Several incorrect words are not corrected by this algorithm, including mis-spelled words that were mapped to a correct word in aspell.txt. One example is "aquantance" gets corrected to "cquantance". I think this is an interesting one because this means that the model has correctly learned that 'c'->'q' is more probable than 'a'->'q' (our dataset does not have any correct words with 'a'->'q') but my algorithm does not handle insertions of an additional letter so this algorithm has also evaluated that it is more probable that "quantance" will have a 'c' before the 'q' instead of an 'a'. Similarly, "aqua" gets corrected to "cqar"

**❄️Give an example of a word which was incorrectly spelled by the user, and was correctly corrected by the algorithm. Why was this one correctly corrected, while the previous two were not?**
- "lasdr" is correctly corrected to "laser" this is because "er" and is a common letter pattern in our training dataset. Our model has learned that l -> a -> s -> e -> r is more probable than l -> a -> s -> d -> r. It has also learned that 'd'->'e' has a high emission probability. We can confirm this by trying "lovd" -> "love" but "lasrr" changes to "lasur". Telling us that this correction is quite dependant on the emission probabilities for 'd'.

**❄️ How might the overall algorithm’s performance differ in the “real world” if that training dataset is taken from real typos collected from the internet, versus synthetic typos (programmatically generated)**
- I think that the algorithm will do a lot better with a larger dataset, which includes real world typos, but will still not be perfect.