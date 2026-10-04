# Plan de surveillance du biais en production

## Pourquoi surveiller

Le modèle corrigé referme l'écart d'égalité des chances sur les données historiques. Rien ne garantit que
cela dure :
- **les proxys de la région restent présents** : distance, heures travaillées, revenu (`audit_rapport.ipynb`,
  section 2). Le biais peut revenir par eux si la population change ;
- **le « mérite réel » n'est pas observé en production** : il faut des indicateurs calculables sans
  étiquette, complétés par un contrôle humain périodique.

## Indicateurs (à chaque cohorte d'attribution)

Calculés par `indicateurs()` dans `modele_corrige.ipynb` (section 6). Les valeurs de référence sont celles
du jeu de test au déploiement.

| Indicateur | Définition | Référence (test) | Seuil d'alerte |
|---|---|---|---|
| Taux d'octroi | part des candidats qui reçoivent la bourse | 0.425 | hors de 36 – 44 % |
| Écart de parité | taux d'octroi Centre − Éloignée | 0.013 | \|écart\| > 0.05 |
| Écart d'égalité des chances | TPR Centre − TPR Éloignée, les qualifiés étant les 40 % meilleures cotes R | -0.029 | \|écart\| > 0.05 |
| Dérive des données | statistique de Kolmogorov-Smirnov maximale entre la cohorte et l'historique (cote R, revenu, heures, distance) | 0.027 | > 0.10 |

L'égalité des chances est la métrique retenue par l'audit (section 1.B). Mesurée contre la cote R, elle ne
demande aucune étiquette : la cote R est dans chaque dossier.

**Première cohorte (les 4 000 candidats)** : taux d'octroi 0.432, écart de parité 0.017, écart d'égalité des
chances -0.034, dérive 0.017. Tout est sous les seuils.

## Quand une alerte se déclenche

1. **Analyser la cause** : dérive des données, changement de la part des régions éloignées, ou problème du
   modèle. Refaire l'analyse par tranches (région, cote R, revenu) de `audit_rapport.ipynb`.
2. **Si l'alerte persiste deux cohortes de suite**, ou si le contrôle annuel est hors seuil : suspendre les
   décisions automatiques, revenir à la revue humaine et réentraîner le modèle.
3. **Avant tout redéploiement** : refaire le front de Pareto (`modele_corrige.ipynb`, section 5) et l'audit,
   et versionner le modèle, le paramètre `retrait` et les seuils.
