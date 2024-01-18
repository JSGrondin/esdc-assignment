# Examen maison d'apprentissage automatique
#### EDSC, Bureau de la dirigeante principale des données, Division de la Science des Données

### Suggestion de temps à allouer pour le devoir: 
180 minutes réparties entre la section programmation (1) et les questions à développement (1.1 et 2).

## 1. Problème d'apprentissage automatique (en Python)

Vous êtes un.e scientifique des données engagé.e par ArXiv. 
On vous charge de développer un <em>pipeline</em> qui catégorisera automatiquement chaque nouvel article soumis.
Vous avez à votre disposition un ensemble de données étiquetées de 7,500 résumés d'articles scientifiques.

### Structure du dossier
- `assignment_devoir/`
  - `assignment.md`/ `devoir.md`
    - Le document contenant les instructions (dans les deux langues officielles)
- `dataset/`
  -  `abstract_arxiv.csv`
      - Les données sont stockées dans un fichier à trois colonnes: <em>Id</em>, <em>Abstract</em> et <em>Category</em>.
      Si par hasard vous aviez une expérience antérieure avec cet ensemble de données, il s'agit ici d'une version modifiée.
- `scripts/`
  - `dataset.py`
    - Complétez la classe `Dataset` en suivant les TODOs.
  - `model.py`
    - Complétez la classe `Model` en suivant les TODOs.
  - `main_loop.py`
    - Programmez les étapes nécesaires qui résulteraient en 
      un <em>pipeline</em> complet. Vous aurez à utiliser les classes `Dataset` et `Model` qu'on vous a demandé de 
      coder.
      Ensuite, entraînez un <em>Multinomial Naive Bayes</em> (fourni par `scikit-learn`) et utilisez votre modèle pour générer
      des prédictions sur un <em>split</em>. Utiliser ces prédictions avec la classe `Result`
      pour générer la performance (<em>accuracy</em>), le rapport de classification et la <em>heatmap</em>
      de la matrice de confusion.
    - L'objectif ici n'est **pas** de faire plusieurs ajustements pour améliorer la performance de votre modèle. 
      Nous voulons seulement voir comment vous abordez la problématique.
    - Vous pouvez uniquement installer les <em>packages</em> inscrits dans le fichier `requirements.txt` pour faire 
      le devoir. Bien sûr, vous pouvez utiliser ceux de la bibliothèque standard de Python.
  - `utils`
    - `utils.py`
      - Rien à faire ici. C'est simplement du code qui vous est donné et dont vous devrez utiliser une partie. 
        N'hésitez pas à y ajouter vos propres fonctions si cela vous aide.
    - `stopwords.txt`
      - Une liste de mots vides (<em>stopwords</em>) mise à votre disposition.
  
Avant d'implémenter votre programme, vous aurez à installer des <em>packages</em>. Le fichier `requirements.txt` 
vous permettra d' installer les bibliothèques et modules appropriés. Pour ce faire, 
allez dans le terminal, localisez le dossier dans lequel se trouve le programme et exécutez cette ligne:

<code>pip install -r requirements.txt</code>

Si cela ne fonctionne pas pour vous, il est possible d'installer les bibliothèques permises ainsi :

- <code>pip install nltk</code> (et ainsi de suite pour `numpy`, `pandas`, `sklearn`, `matplotlib` et `seaborn`)

## 1.1 Analyse

Décrivez votre solution au problème présenté ci-dessus en répondant aux questions suivantes.

### Données

- Décrivez votre exploration des données.
- Le pré-traitement des données peut être bénéfique. Justifiez chaque étape de 
  pré-traitement implémentée.
- Sans modifier ou relancer votre programme, si vous entraîniez un modèle sans implémenter de pré-traitement, 
  quel en serait l'impact sur le comportement du modèle ainsi que l'impact en termes computationnels? Pourquoi?

### Distribution

Voici un histogramme fictif de données d'entraînement.

![image info](../images/dataset_balanced.png)

Disons que l'ensemble de données d'entraînement avait une distribution différente (voir la seconde figure ci-bas); 
- Quel(s) problème(s) en découlerai(en)t? 
- Que peut-on faire pour réctifier la situation?

![image info](../images/dataset_unbalanced.png)

### Résultats

Dans la partie programmation du devoir, on vous demande de présenter les résultats.
L'une des composantes de la classe `Results` vous permet de montrer une <em>heatmap</em> d'une matrice de confusion.

- Qu'observez-vous dans le coin supérieur gauche de la <em>heatmap</em> concernant les catégories astro-ph* 
  (astro-ph, astro-ph.CO, astro-ph.GA, astro-ph.SR). En examinant la précision et le rappel des catégories 
  astro-ph* dans le rapport de classification, on observe également ce phénomène.

- Pourquoi est-ce que ça arrive? Proposez une solution pour remédier au problème.

### Comparaison avec d'autres modèles de ML

Si on remplaçait le modèle de classification par un <em>K-Nearest Neighbor (KNN)</em> dans la section 1) du devoir,
nous pourrions obtenir une performance (<em>accuracy</em>) d'environ 25 à 30 % (même avec la valeur optimale de K). 

- Pourquoi <em>KNN</em> est moins adapté à ce problème que le <em>Multinomial Naïve Bayes</em>?

## 2. Cas fictif du gouvernement
Au Canada, le programme d'assurance-emploi (AE) est chargé de fournir un soutien de revenu temporaire aux sans-emplois.
Chaque fois qu'un.e employé.e quitte son emploi, on doit lui remplir un formulaire « Relevé d'emploi » (RE).
1 million d'employeurs canadiens produisent plus de 8 millions de formulaires RE pour leurs employé.es chaque année.

Supposons que vous êtes un.e scientifique des données dont la tâche est d'améliorer l'efficacité du programme
d'AE. On vous assigne la tâche de développer un <em>pipeline</em> qui catégorisera les commentaires en différentes
« raisons de départ » (RDD). Ces catégories seront ensuite utilisées par les agent.es de 
l'AE afin de faciliter leur travail. Les catégories n'ont pas été définies, vous devez donc les établir.
- Donnez une description détaillée de la façon dont un tel système serait mis en œuvre et 
  comment vous évalueriez son succès.
  
## 3. Livrables

- Votre implémentation de `dataset.py`, `main_loop.py` et `model.py` complétés.
- Vos réponses correspondant aux questions à développement du devoir
