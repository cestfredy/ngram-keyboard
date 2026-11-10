# Journal

L'histoire du projet dans l'ordre : ce qu'on a décidé, et le chiffre qui l'a justifié.

## Les règles du jeu

- Le jeu de test n'est utilisé qu'**une seule fois**, tout à la fin.
- Le vocabulaire et les comptages sont construits sur **l'entraînement seul**.
- Chaque mesure est notée dès qu'elle est faite.

## Les décisions clés

| Décision | Pourquoi |
|----------|----------|
| Fixer la graine aléatoire | Des découpages reproductibles |
| Stocker les comptages et calculer les probabilités à la demande | Changer de lissage sans tout recompter |
| Découper les élisions (`l'` + `ami`), garder les mots à trait d'union entiers | Un vocabulaire plus petit : 39,75 % des phrases ont une élision, et `est-ce` apparaît à lui seul 13 334 fois |
| Dédupliquer sur les séquences de mots, avant le découpage | 8 597 doublons trouvés, contre 149 sur le texte brut en minuscules : la ponctuation les cachait |
| Construire le vocabulaire sur l'entraînement seul | Le jeu de test ne doit pas influencer le modèle |
| Régler le top-k sur un échantillon de 5 000 phrases de validation | La validation complète coûte plus de 15 minutes par bigramme en Python pur ; la perplexité reste calculée sur toute la validation |
| Évaluer le test une fois, sur des modèles figés après la validation | Vocabulaire de 20 000 mots, ordre 4, d = 0,75, élagage `2,2,2` (standard) et `2,3,4` (léger), rien n'est réglé sur le test |

## 1. Les données

Le corpus est celui de Tatoeba : 730 079 phrases françaises, en UTF-8, sans ligne mal formée.

Ce qu'on y a trouvé en l'explorant :

- **Longueur :** médiane de 6 mots, moyenne de 7,47, 95 % des phrases en ont 15 ou moins, 99 % en ont 26 ou moins, maximum 246. Seules 0,20 % des phrases tiennent en un mot et 0,27 % dépassent 40 mots.
- **Doublons :** aucun exact, mais 149 (0,02 %) une fois passé en minuscules. Il faut les retirer **avant** le découpage, sinon une même phrase peut se retrouver à la fois en entraînement et en test.
- **Apostrophes :** 37,85 % des phrases utilisent l'apostrophe droite `'` et 2,90 % l'apostrophe courbe `’`. Sans normalisation, `l'ami` et `l’ami` seraient deux mots différents.
- **Espaces :** espace insécable (2,52 %) et espace fine insécable (2,11 %), le plus souvent avant `? ! : ;`. On trouve aussi 18 espaces de largeur nulle et 7 traits d'union conditionnels.
- **Traits d'union :** 15,08 % des phrases en contiennent, surtout des inversions (`est-ce` 13 334 fois, `as-tu`, `avez-vous`, `a-t-il`) et des mots composés (`peut-être`, `là-bas`).
- **Élisions :** 39,75 % des phrases en contiennent (`l'`, `n'`, `d'`, `j'`, `qu'`, `c'`, `s'`, `m'`, `t'`).
- **Chiffres et alphabets :** 0,69 % des phrases contiennent des chiffres ; 43 phrases utilisent d'autres alphabets (grec, cyrillique, chinois, hébreu, symboles mathématiques).
- **Un biais de domaine :** le prénom « Tom » apparaît dans 5,99 % des phrases (43 744), « Marie » dans 2,2 %. Ce sont des phrases écrites et traduites, pas des messages. Il faut s'attendre à voir « Tom » dans les suggestions : c'est une limite connue, à comparer plus tard avec un corpus de sous-titres.
- **Lecture à la main** de 100 phrases tirées au hasard : elles sont propres et grammaticales, avec presque aucun argot, et les espaces avant `?` et `!` sont inconsistantes.

### Le nettoyage retenu

1. Normaliser `’` `‘` `ʼ` en `'`, remplacer les espaces insécables par une espace normale, supprimer les espaces de largeur nulle et les traits d'union conditionnels.
2. Passer en minuscules.
3. Écarter les phrases contenant des lettres non latines, de plus de 40 mots, ou de moins de 2 mots.
4. Découper en mots : retirer la ponctuation, couper les élisions après l'apostrophe (`l'ami` devient `l'` + `ami`), garder entiers les mots à trait d'union (`est-ce`, `peut-être`) et `aujourd'hui`, remplacer les suites de chiffres par `<num>`.
5. Retirer les doublons **sur les séquences de mots** (donc `Tom ?` et `Tom !` n'en font qu'un), avant le découpage.

### Ce que ça donne

| Étape | Nombre |
|-------|-------:|
| Phrases brutes | 730 079 |
| Écartées par le nettoyage | 5 212 (1 451 d'un mot, 1 715 d'un seul mot, 2 003 de plus de 40 mots, 43 non latines) |
| Doublons retirés | 8 597 |
| Phrases gardées | 716 270 |
| Entraînement / validation / test | 644 643 / 35 813 / 35 814 |
| Mots d'entraînement | 4 956 646 |
| Mots distincts d'entraînement | 78 166 (dont 29 470 vus une seule fois) |

Deux contrôles : les trois parties n'ont aucune phrase en commun, et lancer la préparation deux fois donne exactement les mêmes fichiers (même empreinte).

## 2. Le vocabulaire

Le vocabulaire est formé des N mots les plus fréquents de l'entraînement ; tout le reste devient `<unk>` (mot inconnu).

| Vocabulaire | Mots inconnus (validation) | Taux | Phrases avec au moins un inconnu |
|------------:|---------------------------:|-----:|---------------------------------:|
| 10 000 | 15 010 | 5,43 % | 31,41 % |
| 20 000 | 8 071 | 2,92 % | 18,37 % |
| 50 000 | 2 841 | 1,03 % | 6,96 % |

C'est le plafond de ce que le modèle pourra jamais proposer : avec 20 000 mots, 2,9 % des mots à prédire sont hors de portée, quel que soit le modèle. Passer de 20 000 à 50 000 mots divise la perte par deux, mais fait grossir le modèle ; le choix se fera plus tard, avec le compromis taille et précision.

Une observation au passage : `tom` est le 22e mot le plus fréquent de l'entraînement (39 275 occurrences) et apparaît dans 2 220 des 35 813 phrases de validation (6,2 %).

## 3. La référence à battre

La référence la plus simple propose toujours les trois mêmes mots, `de je est`, quoi qu'on ait tapé. Sur la validation (276 255 positions, la fin de phrase n'est pas évaluée, un mot cible hors vocabulaire compte comme raté), elle trouve le bon mot en premier dans 3,11 % des cas et parmi les trois dans **7,55 %**. Ces chiffres sont les mêmes pour 10 000, 20 000 ou 50 000 mots ; seule la précision restreinte aux mots connus bouge (top-3 : 7,98 % à 10 000, 7,77 % à 20 000, 7,63 % à 50 000).

Tout modèle n-gramme doit battre ces **7,55 %** de top-3 pour se justifier.

Cette référence a aussi une perplexité, celle du modèle de fréquence des mots qui se cache derrière : **487,8** sur toute la validation, avec 20 000 mots, puisqu'il donne une probabilité à chaque mot (la liste fixe `de je est`, elle, n'en donne pas). Ce chiffre montre ce qu'apporte le contexte : 488 sans lui, 42 avec trois mots de contexte.

## 4. Les modèles Kneser-Ney

Le top-1 et le top-3 sont mesurés sur un échantillon aléatoire fixe de 5 000 phrases de validation (environ 38 000 positions, précision d'environ ± 0,3 point) ; la perplexité, sur toute la validation. La latence est celle de Python pur sur un seul cœur, avec deux calculs qui se partageaient la machine : c'est donc un majorant.

### L'ordre du modèle (d = 0,75, 20 000 mots)

| Ordre | Perplexité | Top-1 | Top-3 | Latence |
|------:|-----------:|------:|------:|--------:|
| 1 (référence) | 487,8 | 3,11 % | 7,55 % | |
| 2 | 81,0 | 14,85 % | 28,45 % | 3,4 ms |
| 3 | 48,1 | 22,24 % | 37,78 % | 5,1 ms |
| 4 | 42,2 | 25,71 % | 40,88 % | 7,0 ms |
| 5 | 42,5 | 26,46 % | 41,48 % | 8,9 ms |

Les gains rétrécissent vite : de 4 à 5, on ne gagne que 0,6 point de top-3 pour 63 % de n-grammes en plus et 1,9 ms de plus, et la perplexité n'améliore plus (42,2 contre 42,5). L'ordre 4 est retenu.

### La décote d

À l'ordre 4, le top-3 est plat pour d entre 0,6 et 0,9 (40,88 / 40,88 / 40,73) et baisse à 0,95 (40,35). La perplexité est la meilleure autour de 0,75 à 0,9 (42,2 / 42,7). On garde **d = 0,75**.

### La taille du vocabulaire (ordre 4, d = 0,75, même échantillon)

| Vocabulaire | Top-1 | Top-3 | Types de n-grammes | Latence | Perplexité |
|------------:|------:|------:|-------------------:|--------:|-----------:|
| 10 000 | 25,08 % | 40,34 % | 4,40 M | 3,6 ms | 35,5 |
| 20 000 | 25,71 % | 40,88 % | 4,90 M | 7,0 ms | 42,2 |
| 50 000 | 25,92 % | 41,13 % | 5,34 M | 8,5 ms | 49,5 |

Les perplexités **ne sont pas comparables** d'un vocabulaire à l'autre : un petit vocabulaire transforme plus de mots en `<unk>`, faciles à prédire. On ne compare donc que le top-k. De 10 000 à 50 000 mots, le gain est de 0,8 point de top-3 pour 21 % de n-grammes en plus. La latence dépend aussi du nombre de candidats, qui grandit avec le vocabulaire. On garde 20 000 mots jusqu'à ce que l'élagage et une mesure de latence propre tranchent.

### Le modèle complet sur toute la validation

Sur les 276 255 positions, avec 20 000 mots, d = 0,75 et une machine par ailleurs libre : top-1 **25,52 %**, top-3 **40,80 %**, perplexité 42,2, entraînement en 16,4 s, **5,0 ms** par prédiction (Python pur, 4,9 millions de n-grammes). L'estimation sur l'échantillon (40,88 %) était à 0,08 point près.

## 5. L'élagage

Il reste à rendre le modèle assez petit. L'idée : supprimer les n-grammes dont le comptage est sous un seuil, un seuil par ordre (2, 3 et 4). Les comptages sont ceux que le lissage utilise : des comptages de continuation sous l'ordre maximal, bruts à l'ordre maximal. Le top-k est mesuré sur le même échantillon de 5 000 phrases, la perplexité sur toute la validation. La taille brute estimée compte 4 octets par identifiant de mot et 4 par comptage, sans compression : c'est une règle graduée, pas la taille finale du fichier.

| Seuils (ordres 2, 3, 4) | N-grammes | Taille brute estimée | Top-1 | Top-3 | Perplexité | Latence |
|------------------------:|----------:|---------------------:|------:|------:|-----------:|--------:|
| 1, 1, 1 (aucun) | 4 900 470 | 86,5 Mo | 25,71 % | 40,88 % | 42,2 | 6,3 ms |
| 1, 1, 2 | 2 782 181 | 44,1 Mo | 24,69 % | 39,95 % | 47,9 | 5,9 ms |
| 1, 2, 2 | 1 413 574 | 22,2 Mo | 24,00 % | 38,86 % | 57,3 | 5,5 ms |
| 2, 2, 2 | 1 054 301 | 17,9 Mo | 23,98 % | 38,79 % | 62,9 | 3,6 ms |
| 1, 2, 3 | 1 140 086 | 16,8 Mo | 23,41 % | 38,14 % | 63,1 | 5,4 ms |
| 2, 3, 3 | 606 195 | 9,7 Mo | 22,91 % | 37,41 % | 78,8 | 3,3 ms |
| 3, 3, 3 | 523 753 | 8,7 Mo | 22,86 % | 37,25 % | 88,2 | 2,5 ms |
| 2, 3, 4 | 519 615 | 7,9 Mo | 22,41 % | 36,80 % | 85,0 | 3,2 ms |

Ce qu'on en retient :

- Supprimer 78 % des n-grammes (2, 2, 2) coûte 2,1 points de top-3 (40,88 à 38,79). En supprimer 89 % (2, 3, 4) en coûte 4,1.
- Élaguer les bigrammes vus une seule fois est presque gratuit pour le top-3 : 1, 2, 2 et 2, 2, 2 sont égaux au bruit près (38,86 contre 38,79), alors que 2, 2, 2 est 25 % plus petit.
- La perplexité se dégrade beaucoup plus vite que le top-3 (42 à 63 pour 2, 2, 2) : les n-grammes rares portent de la masse de probabilité que le modèle doit répartir ailleurs, et la perplexité y est sensible, alors que le classement des trois premiers mots bouge à peine. Pour un clavier, c'est le top-k qui compte, mais c'est une faiblesse connue des simples seuils de comptage ; un critère plus fin (élagage par entropie) ferait peut-être mieux à taille égale.
- La latence s'améliore jusqu'à un facteur 2 environ, car moins de candidats sont notés (les temps sont bruités : deux calculs partageaient la machine).

## 6. Le format compact

Le modèle élagué est écrit sous forme de tableaux triés (identifiants de mots sur 16 bits, comptages sur 32 bits). Un lecteur dédié retrouve les n-grammes par recherche dichotomique sur des clés de 64 bits et note tous les candidats d'un coup. Mêmes paramètres : ordre 4, 20 000 mots, d = 0,75, même échantillon de validation.

| Seuils | N-grammes | Fichier | Fichier compressé | Chargement | Top-1 | Top-3 | Latence |
|-------:|----------:|--------:|------------------:|-----------:|------:|------:|--------:|
| 2, 2, 2 | 1 054 301 | **11,2 Mo** | 4,1 Mo | 0,05 s | 23,98 % | 38,79 % | **0,42 ms** |
| 2, 3, 4 | 519 615 | **5,2 Mo** | 2,0 Mo | 0,03 s | 22,41 % | 36,80 % | **0,40 ms** |

- La précision est identique à celle du modèle de départ (mêmes top-1 et top-3 que dans le tableau d'élagage), et les suggestions sont identiques sur 7 547 positions contrôlées ; les tests comparent aussi les deux modèles sur des corpus aléatoires.
- La latence est divisée par 10 environ (3,6 ms à 0,4 ms), parce que les candidats sont notés en une seule étape vectorisée au lieu d'une boucle Python. Le modèle complet (4,9 millions d'entrées, 1,2 Go sous forme de dictionnaires Python) tient maintenant dans 11 Mo après élagage : l'objectif de taille est atteint avec une grande marge.
- La « taille brute estimée » du tableau d'élagage n'était qu'une règle graduée ; les vrais fichiers s'en écartent d'environ 40 % (11,2 Mo contre 17,9 Mo pour 2, 2, 2).

## 7. Les touches économisées

Un clavier ne se juge pas qu'à son top-3 avec zéro lettre tapée : l'utilisateur tape des lettres, et les suggestions se resserrent. On a donc simulé la frappe.

**La définition.** Un utilisateur simulé tape chaque mot lettre par lettre. Avant chaque lettre, le clavier affiche les 3 mots les plus probables qui commencent par ce qui est déjà tapé. Dès que le mot apparaît, il le touche (1 appui, qui insère le mot et l'espace). Sans prédiction, un mot coûte `longueur + 1` appuis. Les mots hors vocabulaire sont tapés en entier avec leur vraie longueur, et les `<num>` sont ignorés (leur longueur a été perdue au nettoyage). L'économie vaut 1 − touches avec prédiction / touches sans. Elle est mesurée sur toute la validation : 275 943 mots et 1 446 648 appuis sans prédiction.

| Modèle | Économie | Mot proposé avant toute lettre | Mot proposé avant sa dernière lettre | Lettres tapées avant le succès | Latence |
|--------|---------:|-------------------:|--------------------:|------:|--------:|
| Fréquence des mots, sans contexte (20 000) | 38,32 % | 7,56 % | 86,93 % | 2,00 | 0,11 ms/mot |
| 4-grammes élagués `2,2,2` (11,2 Mo) | **56,18 %** | 38,70 % | 93,04 % | 1,03 | 0,51 ms/mot |
| 4-grammes élagués `2,3,4` (5,2 Mo) | **55,12 %** | 36,70 % | 92,85 % | 1,08 | 0,50 ms/mot |

- Une barre de prédiction qui ne connaît aucun contexte économise déjà 38 % sur ce corpus, parce que les mots les plus fréquents qui commencent par une ou deux lettres sont de courts mots-outils. Le contexte amène le modèle à 56 %, soit 18 points de plus, et divise par deux les lettres tapées avant le succès (2,0 à 1,0).
- Le modèle léger (`2,3,4`, deux fois plus petit) ne perd que 1,1 point d'économie, bien moins que les 4 points perdus en top-3 sans lettre tapée : dès qu'une ou deux lettres sont tapées, l'ensemble de candidats est petit et le contexte compte moins.
- Une réserve : Tatoeba, ce sont des phrases courtes, simples et traduites (médiane de 6 mots), donc ces chiffres sont optimistes par rapport à de vrais messages, et ils supposent que l'utilisateur voit et touche toujours une bonne suggestion sans effort. Il faudra les remesurer sur un autre corpus (sous-titres) avant de les présenter comme un résultat général.

## 8. La démo web

Le modèle élagué est exporté dans un format binaire plat (un en-tête JSON suivi de tableaux d'entiers triés) que l'application lit directement en JavaScript, sans dépendance, avec la même formule et le même départage que le modèle Python. Le modèle est téléchargé une seule fois et gardé par le navigateur, et un service worker garde l'application : après la première visite, tout marche sans réseau.

- **Parité :** sur 3 000 positions réelles de validation par modèle (avec et sans début de mot tapé), le lecteur JavaScript donne exactement les mêmes 3 suggestions que le modèle Python (0 écart, modèles standard et léger). Dix tests unitaires comparent aussi les deux sur un tout petit modèle (450 cas de suggestions, 151 probabilités, 18 cas de découpage en mots).
- **Vitesse (Node) :** 0,2 ms en moyenne par suggestion sans début de mot, 0,05 à 0,07 ms avec un début de mot ; la lecture du modèle léger de 5,2 Mo prend environ 45 ms. Dans Chromium sans interface, une frappe complète (suggestions et mise à jour de l'écran) prend environ 0,6 ms en médiane, 1,3 ms au plus.
- **Hors ligne :** testé dans Chromium avec le réseau coupé et la page rechargée : l'application s'ouvre, lit le modèle dans le cache et prédit.
- **Léger ou standard :** le léger fait 5,2 Mo (seuils 2, 3, 4), le standard 11,3 Mo (2, 2, 2) ; la page laisse choisir.
- Seuls les deux modèles de la démo sont versionnés (16,5 Mo à eux deux), pour que la page puisse les servir ; tous les autres modèles restent hors du dépôt.

## 9. Le verdict : le jeu de test

Une fois tous les choix figés, le jeu de test (35 814 phrases, 275 535 mots) a été lu **une seule fois**. Le programme d'évaluation avait d'abord été essayé sur la validation, puis verrouillé : il refuse de relire le test une seconde fois. Le top-1 et le top-3 portent sur **toutes** les positions du test (pas d'échantillon), la fin de phrase n'est pas évaluée, un mot cible hors vocabulaire compte comme raté ; la perplexité inclut la fin de phrase ; l'économie de touches suit la même simulation que sur la validation (3 suggestions).

| Modèle | N-grammes | Perplexité | Top-1 | Top-3 | Touches économisées |
|--------|----------:|-----------:|------:|------:|--------------------:|
| Référence : mots les plus fréquents (avec filtre sur les premières lettres dans la simulation) | | n/a | 3,08 % | 7,55 % | 38,45 % |
| 4-grammes, sans élagage | 4 900 470 | 41,5 | 25,75 % | 41,11 % | |
| 4-grammes, standard (`2,2,2`, 11,3 Mo) | 1 054 301 | 62,1 | 24,28 % | 39,05 % | 56,41 % |
| 4-grammes, léger (`2,3,4`, 5,2 Mo) | 519 615 | 83,4 | 22,77 % | 36,99 % | 55,36 % |

- La précision et l'économie de touches sont de 0,2 à 0,4 point au-dessus de la validation (top-3 en validation : 40,80 / 38,79 / 36,80 ; économie : 56,18 / 55,12), et la perplexité est de 1 à 2 % plus basse (41,5 / 62,1 / 83,4 contre 42,2 / 63,0 / 85,0). C'est la taille du bruit d'échantillonnage mesuré en validation (environ 0,3 point), et l'écart va dans le même sens pour tous les modèles : rien n'indique un surajustement à la validation.
- Le classement des modèles et le coût de l'élagage sont inchangés : le modèle léger perd 4,1 points de top-3 par rapport au modèle sans élagage, mais seulement 1,05 point d'économie de touches par rapport au standard.
- La perplexité de la référence n'a pas été mesurée sur le test (487,8 en validation) : on a gardé le test pour une seule évaluation des modèles finaux.

## Ce qu'on retient, et la suite

- Le contexte fait toute la différence : la perplexité passe de 488 à 42, et le top-3 de 7,55 % à plus de 40 %.
- Un modèle de 5 Mo, qui prédit en une fraction de milliseconde, garde l'essentiel de la qualité du modèle complet de 1,2 Go : 37 % de top-3 contre 41 %, et 55 % d'économie de touches contre 56 %.
- Le jeu de test est désormais consommé : aucun réglage ne peut plus s'appuyer dessus. Toute nouvelle variante (autre corpus, autre lissage) aura besoin d'un nouveau jeu de réserve. La suite la plus utile est un corpus plus proche de vrais messages, avec des sous-titres.
