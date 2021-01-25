IFT3395-A-A19 - Fondements de l'apprentissage machine

Groupe: Julia Potokina, Ryan Noel

Comment exécuter:
-Pour le fichier ipynb:
Pour lancer le code, il suffit de lancer la cell du fichier ipynb fourni, cela devrait prendre quelques minutes

-Pour le fichier py:


k = 5
alpha = 0.35
inputs, labels, test = load_data()
model_a = preprocessing()
inputs = model_a.filter_words( inputs,
                 exclusion = True,
                 lemmatize = False,
                 stemmer = False)
test = model_a.filter_words( test,
                 exclusion = True,
                 lemmatize = False,
                 stemmer = False)
x_train, y_train, x_test, y_test= data_split(inputs, k)
i=0
nbc_model_a = NBC(x_train[i],y_train[i],test,y_test[i], alpha[x], TFIDF = False, smoothing=True)
nbc_model_a.compute_predictions()

Organisation du code du classifieur de Bayes naif:

- def load_data(): on charge les données du training sample et du test sample 

- class preprocessing(): Le preprocessing des données. En hyperparamètres dans une de ses fonctions, 
nous pouvons choisir d'exclure des mots, un stemmer,un lemmatizer.

- def get_wordnet_pos(self, treebank_tag): Fonction qui sert à coller un POS tag à chaque mot pour pouvoir utiliser le lemmatizer. Le code
est directement pris de [SOURCE] : https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python . 

- def tokenize(self, inputs): Fonction qui sépare les mots en utilisant le tokenizer de NLTK.

- def filter_words(self, inputs, exclusion = False, lemmatize = False, stemmer = False): Fonction qui prend des hyperparamètes. L'exclusion
permet de filtrer des symboles trop communs (tel que la ponctuation). Il y a aussi option de filtrer les mots avec un stemmer et
un lemmatizer, venant de la librairie NLTK.

- def data_split(train,k = None): on utilise cette fonction afin de faire des tests en local sur le training sample
en séparant de manière aléatoire le training sample avec 80% de training et 20% de test afin d'évaluer en avant-plan 
l'efficacité de notre classifier

- class NBC(): c'est la classe de notre classifier "Naive Bayes Classifier"

- def concatenated_sentence_by_subreddit(self): ici on retourne une liste de "méga-phrases" propres à chaque subreddit, ce qui 
équivaut à 20 méga phrases qui sont la concaténation de tous les textes associés à chaque subreddit. On retourne également, la
liste des mots uniques de chaque méga phrase et leur nombre.

- def compute_probabilities(self): ici on va calculer la probabilité qu'un mot appartienne à un subreddit, on y ajoute un lissage 
de Laplace afin d'éviter les cas où le mot n'apparait pas dans les textes du subreddit. On utilise également une méthode de pondération
TF IDF qui permet d'évaluer le poids d'un poids et sa pertinence dans les calculs. Si le mot est trop présent dans beaucoup de subreddits 
différents, il va perdre en importance. Le lissage et TF-IDF sont des hyperparamètres optionnels.

- def compute_predictions_sentence(self,sentence): ici on va calculer les prédictions sur un texte dont on ne connait pas son appartenance
à un subreddit en multipliant toutes les probabilités des mots de la phrase appartenant à chaque subreddit, on obtient donc une liste de 
20 probabilités et pour établir la prédiction, on retourne le subreddit name associé à la probabilité la plus grande

- def compute_predictions(self): ici on va appliquer compute_predictions_sentence() à tous les textes du test sample

- def validation_error(self): ici on calcule l'erreur de validation

- def conf_matrix(self): ici on retourne la matrice de confusion
