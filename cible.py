"""Redefinition de la cible : score contrefactuel debiaise de `decision_octroi`.

Hypothese : le comite est juste pour le groupe Centre. Le merite estime d'un
candidat est la probabilite d'octroi qu'il aurait eue s'il avait ete traite
comme un candidat du Centre, toutes choses egales par ailleurs.

On modelise la decision avec un indicateur de region explicite (regression
logistique), puis on predit avec cet indicateur force a 0. `code_postal_3` est
exclu : il est determine par la region et absorberait l'effet a retirer.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ELOIGNEES = ['Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine']
NUMERIQUES = [
    'cote_r_equivalent',
    'revenu_familial_estime',
    'heures_travail_semaine',
    'distance_domicile_campus_km',
    'premiere_generation_universitaire',
    'eloignee',
]
CATEGORIELLES = ['programme_etudes']


def ajouter_eloignee(df):
    """Copie de `df` avec l'indicateur binaire `eloignee` (1 = region eloignee)."""
    df = df.copy()
    df['eloignee'] = df['region_administrative'].isin(ELOIGNEES).astype(int)
    return df


def ajuster_modele_decision(df):
    """Regression logistique de `decision_octroi` avec la region explicite."""
    df = ajouter_eloignee(df)
    pretraitement = ColumnTransformer([
        ('num', StandardScaler(), NUMERIQUES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORIELLES),
    ])
    modele = make_pipeline(pretraitement, LogisticRegression(max_iter=2000))
    modele.fit(df[NUMERIQUES + CATEGORIELLES], df['decision_octroi'])
    return modele


def coefficients(modele):
    """Coefficients standardises du modele de decision, tries par valeur absolue."""
    noms = modele[0].get_feature_names_out()
    coef = pd.Series(modele[-1].coef_[0], index=noms)
    return coef.reindex(coef.abs().sort_values(ascending=False).index)


def score_contrefactuel(modele, df, neutraliser=None, reference=None, retrait=1.0, ponderations=None):
    """Probabilite d'octroi avec la penalite regionale retiree.

    `retrait` : fraction de la penalite retiree (0 = comite, 1 = `eloignee` force a 0,
    > 1 = surcorrection). Le modele etant lineaire en logit, `eloignee = 1 - retrait`
    retire exactement cette fraction.
    `neutraliser` : colonnes supplementaires fixees a leur mediane dans `reference`
    (par defaut `df`), pour retirer aussi leur effet (ex. ['revenu_familial_estime']).
    `ponderations` : {colonne: multiplicateur} applique au poids du comite pour cette
    colonne, autour de sa moyenne dans `reference` (1 = inchange, 0 = neutralise,
    -1 = effet inverse). Exact, car le modele est lineaire en logit.
    """
    df = ajouter_eloignee(df)
    reference = df if reference is None else reference
    df['eloignee'] = df['eloignee'] * (1 - retrait)
    for colonne in neutraliser or []:
        df[colonne] = reference[colonne].median()
    for colonne, multiplicateur in (ponderations or {}).items():
        moyenne = reference[colonne].mean()
        df[colonne] = moyenne + multiplicateur * (df[colonne] - moyenne)
    return modele.predict_proba(df[NUMERIQUES + CATEGORIELLES])[:, 1]


def octroi_top_k(scores, taux=0.40):
    """Accorde la bourse exactement aux `taux` meilleurs scores."""
    scores = np.asarray(scores)
    k = int(round(taux * len(scores)))
    decisions = np.zeros(len(scores), dtype=int)
    decisions[np.argsort(-scores, kind='stable')[:k]] = 1
    return decisions


def octroi_par_tranche_r(df, decisions, bornes=(0, 26, 27, 28, 29, 30, 32, 45)):
    """Taux d'octroi par tranche de cote R et par groupe (Centre / Eloignee)."""
    df = ajouter_eloignee(df)
    tranche = pd.cut(df['cote_r_equivalent'], list(bornes))
    groupe = np.where(df['eloignee'] == 1, 'Eloignee', 'Centre')
    return (
        pd.Series(np.asarray(decisions), index=df.index)
        .groupby([tranche, groupe], observed=True).mean()
        .unstack()
    )


# --- Cible retenue ----------------------------------------------------------------

# Criteres du comite retires du merite, en plus de la penalite regionale. Calibres contre
# l'etalon cache avec les sondes de `sondes_cible.ipynb` : le revenu n'en fait pas partie
# (optimum du multiplicateur a -0.01), et la distance est un proxy pur de la region.
PONDERATIONS_MERITE = {'revenu_familial_estime': 0, 'distance_domicile_campus_km': 0}


def score_merite(modele, df, reference=None):
    """Score de merite retenu : penalite regionale retiree, revenu et distance neutralises.

    `reference` : jeu dont les moyennes servent a neutraliser (par defaut `df`).
    """
    return score_contrefactuel(modele, df, reference=reference, retrait=1.0,
                               ponderations=PONDERATIONS_MERITE)


# --- Cible d'entrainement pour un modele supervise -------------------------------

# Variables du modele entraine : ni la region ni le code postal, ni le revenu ni la
# distance, dont la cible a retire l'effet (le modele leur apprendrait un poids nul).
VARIABLES_MODELE = [c for c in NUMERIQUES
                    if c not in ['eloignee', *PONDERATIONS_MERITE]] + CATEGORIELLES


def cible_individuelle(modele, df, reference=None):
    """Probabilite d'octroi contrefactuelle de chaque dossier, sachant la decision reelle.

    Contrairement a `score_merite`, qui ne depend que des variables, cette cible garde
    l'information individuelle du comite (abduction, au sens de Pearl). On note p la
    probabilite d'octroi du comite et q le score de merite :
    - comite a accorde : min(1, q / p). Un candidat que le revenu avait aide peut descendre ;
    - comite a refuse : max(0, (q - p) / (1 - p)). Un candidat penalise peut monter.
    En moyenne sur la decision du comite, la cible vaut exactement q.
    `reference` : jeu dont les moyennes servent a neutraliser (par defaut `df`).
    """
    p = score_contrefactuel(modele, df, retrait=0)
    q = score_merite(modele, df, reference)
    y = df['decision_octroi'].to_numpy()
    return np.where(y == 1, np.minimum(1.0, q / p), np.maximum(0.0, (q - p) / (1 - p)))


def dupliquer_cible_souple(X, cible):
    """Transforme une cible souple en classification ponderee.

    Chaque dossier apparait deux fois : etiquette 1 avec poids `cible`, etiquette 0
    avec poids `1 - cible`. Retourne (X, y, poids).
    """
    cible = np.asarray(cible)
    n = len(cible)
    return (pd.concat([X, X], ignore_index=True),
            np.r_[np.ones(n, dtype=int), np.zeros(n, dtype=int)],
            np.r_[cible, 1 - cible])


def entrainer_modele_cible(df, cible, estimateur=None, new_features=None):
    """Entraine un modele de production sur une cible souple (variables : `VARIABLES_MODELE`).

    `estimateur` : classifieur scikit-learn (regression logistique par defaut).
    """
    if estimateur is None:
        estimateur = LogisticRegression(max_iter=3000)
    features = VARIABLES_MODELE + (new_features if new_features is not None else [])
    numeriques = [c for c in features if c not in CATEGORIELLES]
    pretraitement = ColumnTransformer([
        ('num', StandardScaler(), numeriques),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORIELLES),
    ])
    modele = make_pipeline(pretraitement, estimateur)
    X, y, poids = dupliquer_cible_souple(df[features], cible)
    modele.fit(X, y, **{f'{modele.steps[-1][0]}__sample_weight': poids})
    return modele
