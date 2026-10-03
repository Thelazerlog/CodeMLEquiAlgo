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


def score_contrefactuel(modele, df, neutraliser=None, reference=None, retrait=1.0):
    """Probabilite d'octroi avec la penalite regionale retiree.

    `retrait` : fraction de la penalite retiree (0 = comite, 1 = `eloignee` force a 0,
    > 1 = surcorrection). Le modele etant lineaire en logit, `eloignee = 1 - retrait`
    retire exactement cette fraction.
    `neutraliser` : colonnes supplementaires fixees a leur mediane dans `reference`
    (par defaut `df`), pour retirer aussi leur effet (ex. ['revenu_familial_estime']).
    """
    df = ajouter_eloignee(df)
    reference = df if reference is None else reference
    df['eloignee'] = df['eloignee'] * (1 - retrait)
    for colonne in neutraliser or []:
        df[colonne] = reference[colonne].median()
    return modele.predict_proba(df[NUMERIQUES + CATEGORIELLES])[:, 1]


def octroi_top_k(scores, taux=0.40):
    """Accorde la bourse exactement aux `taux` meilleurs scores (budget garanti)."""
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


# --- Cible d'entrainement pour un modele supervise -------------------------------

# Variables du modele entraine : ni la region, ni ses proxys directs.
VARIABLES_MODELE = [c for c in NUMERIQUES if c != 'eloignee'] + CATEGORIELLES


def cible_individuelle(modele, df):
    """Probabilite d'octroi contrefactuelle de chaque dossier, sachant la decision reelle.

    Contrairement a `score_contrefactuel`, qui ne depend que des variables, cette cible
    garde l'information individuelle du comite (abduction, au sens de Pearl). Si le
    comite a accorde, le candidat aurait aussi obtenu la bourse sans la penalite : 1.
    S'il a refuse, la probabilite qu'il soit passe sans la penalite est
    (p_cf - p) / (1 - p). Pour le Centre, p_cf = p : la cible vaut `decision_octroi`.
    """
    p = score_contrefactuel(modele, df, retrait=0)
    p_cf = score_contrefactuel(modele, df, retrait=1)
    y = df['decision_octroi'].to_numpy()
    return np.where(y == 1, 1.0, (p_cf - p) / (1 - p))


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
