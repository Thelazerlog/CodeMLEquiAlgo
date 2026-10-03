"""Fonctions d'analyse de base : AUC et importance des variables.

Rappel : toute mesure calculee contre `decision_octroi` mesure la fidelite au
comite historique (biaise), pas le merite reel.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CATEGORIELLES = ['programme_etudes', 'region_administrative', 'code_postal_3']


def colonne_d_origine(colonne_encodee):
    """Retrouve la colonne brute d'une colonne one-hot (ex. 'code_postal_3_G0L' -> 'code_postal_3')."""
    for c in CATEGORIELLES:
        if colonne_encodee.startswith(c + '_'):
            return c
    return colonne_encodee


def evaluer_auc(modele, X, y, groupes=None):
    """AUC globale, et par groupe si `groupes` est fourni.

    Retourne une Series indexee par 'global' puis par groupe.
    """
    scores = modele.predict_proba(X)[:, 1]
    resultats = {'global': roc_auc_score(y, scores)}
    if groupes is not None:
        groupes = np.asarray(groupes)
        y = np.asarray(y)
        for g in np.unique(groupes):
            masque = groupes == g
            resultats[g] = roc_auc_score(y[masque], scores[masque])
    return pd.Series(resultats, name='auc')


def importance_impurete(modele, X):
    """Importance par impurete (MDI) de la foret, regroupee par colonne d'origine.

    Rapide mais biaisee vers les variables continues et a forte cardinalite.
    """
    imp = pd.Series(modele.feature_importances_, index=X.columns)
    return imp.groupby(colonne_d_origine).sum().sort_values(ascending=False)


def importance_permutation(modele, X, y, n_repetitions=5, random_state=42):
    """Importance par permutation : baisse d'AUC quand on melange une colonne d'origine.

    Les colonnes one-hot d'une meme variable sont permutees ensemble, pour que
    `region_administrative` compte comme une seule variable.
    Retourne un DataFrame (moyenne, ecart-type) trie par importance.
    """
    rng = np.random.default_rng(random_state)
    auc_ref = roc_auc_score(y, modele.predict_proba(X)[:, 1])

    blocs = pd.Series(X.columns, index=X.columns).groupby(colonne_d_origine).apply(list)
    lignes = {}
    for variable, colonnes in blocs.items():
        baisses = []
        for _ in range(n_repetitions):
            Xp = X.copy()
            Xp[colonnes] = X[colonnes].to_numpy()[rng.permutation(len(X))]
            baisses.append(auc_ref - roc_auc_score(y, modele.predict_proba(Xp)[:, 1]))
        lignes[variable] = {'moyenne': np.mean(baisses), 'ecart_type': np.std(baisses)}
    return pd.DataFrame(lignes).T.sort_values('moyenne', ascending=False)


def tracer_importances(importances, titre, ax=None):
    """Barres horizontales ; accepte une Series ou le DataFrame de importance_permutation."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    if isinstance(importances, pd.DataFrame):
        d = importances.iloc[::-1]
        ax.barh(d.index, d['moyenne'], xerr=d['ecart_type'], color='#4C72B0')
    else:
        d = importances.iloc[::-1]
        ax.barh(d.index, d.values, color='#4C72B0')
    ax.set_title(titre)
    ax.grid(axis='x', alpha=0.3)
    return ax


def ecart_egalite_chances(y_reference, decisions, groupes, favorise='Centre', defavorise='Eloignee'):
    """Ecart de TPR (favorise - defavorise) par rapport a une reference de merite.

    Signe conserve : positif = le groupe `favorise` est avantage. Le scoreur mesure
    cet ecart contre l'etalon cache ; ici, `y_reference` en est une approximation.
    """
    y, d, g = np.asarray(y_reference), np.asarray(decisions), np.asarray(groupes)
    tpr = lambda groupe: d[(g == groupe) & (y == 1)].mean()
    return tpr(favorise) - tpr(defavorise)


def metriques_equite(y_reference, decisions, groupes, favorise='Centre', defavorise='Eloignee'):
    """Metriques d'equite usuelles, en ecart (favorise - defavorise), contre une reference.

    - parite demographique : ecart de taux de selection (n'utilise pas la reference) ;
    - egalite des chances : ecart de TPR ;
    - ecart de FPR : avec l'ecart de TPR, definit l'egalite des chances egalisee ;
    - parite predictive : ecart de precision (part des octrois meritants).
    """
    y, d, g = np.asarray(y_reference), np.asarray(decisions), np.asarray(groupes)

    def par_groupe(fonction):
        return fonction(g == favorise) - fonction(g == defavorise)

    return pd.Series({
        'taux_selection_' + favorise: d[g == favorise].mean(),
        'taux_selection_' + defavorise: d[g == defavorise].mean(),
        'ecart_parite': par_groupe(lambda m: d[m].mean()),
        'ecart_egalite_chances': ecart_egalite_chances(y, d, g, favorise, defavorise),
        'ecart_fpr': par_groupe(lambda m: d[m & (y == 0)].mean()),
        'ecart_precision': par_groupe(lambda m: y[m & (d == 1)].mean()),
        'exactitude': (y == d).mean(),
    })


def intervalle_bootstrap(fonction, *tableaux, strates=None, n=1000, niveau=0.95, random_state=42):
    """Intervalle de confiance bootstrap (percentiles) d'une statistique.

    `fonction(*tableaux)` retourne un scalaire ou une Series. Les lignes sont
    reechantillonnees avec remise, a l'interieur de chaque strate si `strates` est
    fourni (ex. les groupes, pour garder leurs effectifs). Retourne un DataFrame
    (estimation, bas, haut).
    """
    rng = np.random.default_rng(random_state)
    tableaux = [np.asarray(t) for t in tableaux]
    strates = np.zeros(len(tableaux[0]), dtype=int) if strates is None else np.asarray(strates)
    indices_strates = [np.flatnonzero(strates == s) for s in np.unique(strates)]

    tirages = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(i, size=len(i), replace=True) for i in indices_strates])
        tirages.append(pd.Series(fonction(*(t[idx] for t in tableaux))))
    tirages = pd.DataFrame(tirages)

    alpha = (1 - niveau) / 2
    return pd.DataFrame({
        'estimation': pd.Series(fonction(*tableaux)),
        'bas': tirages.quantile(alpha),
        'haut': tirages.quantile(1 - alpha),
    })


def tableau_tranches(y_reference, decisions, tranches, complet=False):
    """Effectif, nombre de meritants, taux de selection et TPR (contre la reference) par tranche.

    `tranches` : Series, ou DataFrame pour croiser plusieurs variables. `complet=True`
    ajoute la FPR, la precision et l'exactitude. Une metrique sans denominateur dans
    une tranche (ex. TPR sans meritant) est mise a NaN.
    """
    from fairlearn.metrics import (MetricFrame, count, false_positive_rate, selection_rate,
                                   true_positive_rate)

    metriques = {'effectif': count, 'meritants': lambda y, d: int(np.sum(y)),
                 'taux_selection': selection_rate, 'tpr': true_positive_rate}
    if complet:
        metriques.update({
            'fpr': false_positive_rate,
            'precision': lambda y, d: y[d == 1].mean() if (d == 1).any() else np.nan,
            'exactitude': lambda y, d: (y == d).mean(),
        })
    cadre = MetricFrame(
        metrics=metriques,
        y_true=np.asarray(y_reference),
        y_pred=np.asarray(decisions),
        sensitive_features=tranches,
    )
    tableau = cadre.by_group
    tableau.loc[tableau['meritants'] == 0, 'tpr'] = np.nan
    if complet:
        tableau.loc[tableau['meritants'] == tableau['effectif'], 'fpr'] = np.nan
    return tableau
