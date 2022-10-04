# ScriptPreidctor

Uses a trigram language model to calculate perplexities of lines from the Friends TV show to predict who a line was said by.

--

## Model

Each cast member in the show (Joey, Ross, Chandler, Monica, Phoebe, and Rachel) are separate trigram models in this program.
Each model is trained on a corpus that has been cleaned to only include lines that they have said.
The model then calculates the unigram, bigram, and trigram probabilities that it parses in the corpus to generate
a smoothed trigram probability for each set of words.
Using the Markov assumption, we are then able to calculate the "likelihood of a sentence", and we use this to calculate the perplexity.

--

## Predicting

Having trained the different character models on their respective scripts, we are now ready to predict who said a given line.
For each test line, we calculate the perplexity that each model associates with the sentences, then select the model with the lowest perplexity.
The corresponding character is the character that the model believes is most likely to have said that sentence.

--

## Examples
```
Test 1: 'Ooh, here's that macadamia nut!'
Said by: Ross
Perplexity, Joey: 3455.9990668674727
Perplexity, Chandler: 1290.1749908780305
Perplexity, Ross: 20.17207376550621
Perplexity, Rachel: 284.30025878900994
Perplexity, Phoebe: 139.77728883169218
Perplexity, Monica: 109.87130981528395
Predicted: Ross
 
Test 2: 'There's nothing to tell! He's just some guy I work with!'
Said by: Monica
Perplexity, Joey: 4925.547977901087
Perplexity, Chandler: 2573.728406982445
Perplexity, Ross: 1979.733384283092
Perplexity, Rachel: 1209.9895864453715
Perplexity, Phoebe: 377.5464951983128
Perplexity, Monica: 192.96392901887359
Predicted: Monica
 
Test 3: 'I love you.'
Said by: All
Perplexity, Joey: 33.16674709950821
Perplexity, Chandler: 15.792146622730266
Perplexity, Ross: 10.75847095760878
Perplexity, Rachel: 8.279131221881613
Perplexity, Phoebe: 6.108525063134733
Perplexity, Monica: 5.124774052849012
Predicted: Monica
```
