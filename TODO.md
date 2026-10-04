- [x] Mettre le projet sur GitHub
- [ ] Utiliser un kernel sur Narval
- [x] Redéfinir le target (cible contrefactuelle, `cible.py` + `model_corrige.ipynb` section 1)
- [x] Ajouter les fonctions d'analyse de base (AUC, feature importance)
- [x] Analyse de données (`analyse_donnees.ipynb`)
- [x] Audit_rapport (`audit_rapport.ipynb`)
- [x] Faites varier la contrainte d'équité et tracez le nuage (équité, exactitude) : c'est votre front de Pareto, et le jury va vous demander de le défendre (`model_corrige.ipynb` section 2)
- [x] Ajouter l'analyse en slice (`audit_rapport.ipynb` §2.3)
- [x] `fairlearn.postprocessing.ThresholdOptimizer` et `fairlearn.reductions.ExponentiatedGradient`
  sont les deux outils les plus rapides à mettre en place. Le premier ajuste les seuils après
  coup, le second réentraîne sous contrainte:
    - [x] ThresholdOptimizer (comparateur, section 2)
    - [x] ExponentiatedGradient (comparateur, balayage de la borne, section 2.2)
- [x] Feature engineering (cote R vs région médiane...)
- [x] Faire audit: (WIP)
- [x] front de Pareto pour plusieurs réglages de la contrainte
- [x] Plan de surveillance (`plan_surveillance.md` + `model_corrige.ipynb` section 6)
- [ ] Améliorer métriques encore si possible
- [ ] Préparer présentation