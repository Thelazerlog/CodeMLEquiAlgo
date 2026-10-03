- [x] Mettre le projet sur GitHub
- [ ] Utiliser un kernel sur Narval
- [x] Redéfinir le target (cible contrefactuelle, `cible.py` + `model_corrige.ipynb` section 1)
- [x] Ajouter les fonctions d'analyse de base (AUC, feature importance)
- [x] Analyse de données (`analyse_donnees.ipynb`)
- [x] Audit_rapport (`audit_rapport.ipynb`)
- [x] Faites varier la contrainte d'equite et tracez le nuage (equite, exactitude) : c'est votre front de Pareto, et le jury va vous demander de le defendre (`model_corrige.ipynb` section 2)
- [x] Ajouter l'analyse en slice (`audit_rapport.ipynb` §2.3)
- [ ] `fairlearn.postprocessing.ThresholdOptimizer` et `fairlearn.reductions.ExponentiatedGradient`
  sont les deux outils les plus rapides a mettre en place. Le premier ajuste les seuils apres
  coup, le second reentraine sous contrainte:
    - [x] ThresholdOptimizer (comparateur, section 2)
    - [ ] ExponentiatedGradient
- [ ] Feature enginering(cote r vs region median...) 
- [ ] weighting
- [ ] Tester avec AutoML, XGBoost, Optuna

Worst case scenario:
- [ ] Parité au serving