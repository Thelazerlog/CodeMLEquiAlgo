"""Redefinition de la cible : merite contrefactuel debiaise de `decision_octroi`.

Etapes, dans l'ordre du module :
1. modele de decision du comite : regression logistique de `decision_octroi` avec la region ;
2. score de merite : probabilite d'octroi du comite, penalite regionale retiree ;
3. cible d'entrainement et modele corrige, sans la region en variable ;
4. decisions : seuil choisi dans le budget d'octroi.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ELOIGNEES = ['Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine']
CATEGORIELLES = ['programme_etudes']
# Variables du modele de decision du comite, dont l'indicateur de region `eloignee`.
NUMERIQUES = [
    'cote_r_equivalent',
    'revenu_familial_estime',
    'heures_travail_semaine',
    'distance_domicile_campus_km',
    'premiere_generation_universitaire',
    'eloignee',
]


def _pretraitement(numeriques):
    """Standardise les variables numeriques et encode le programme en one-hot."""
    return ColumnTransformer([
        ('num', StandardScaler(), numeriques),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORIELLES),
    ])


# --- 1. Modele de decision du comite ---------------------------------------------

def ajouter_eloignee(df):
    """Copie de `df` avec l'indicateur binaire `eloignee` (1 = region eloignee)."""
    df = df.copy()
    df['eloignee'] = df['region_administrative'].isin(ELOIGNEES).astype(int)
    return df


def ajuster_modele_decision(df):
    """Regression logistique de `decision_octroi`, avec la region explicite (`eloignee`)."""
    df = ajouter_eloignee(df)
    modele = make_pipeline(_pretraitement(NUMERIQUES), LogisticRegression(max_iter=2000))
    modele.fit(df[NUMERIQUES + CATEGORIELLES], df['decision_octroi'])
    return modele


def coefficients(modele):
    """Coefficients standardises d'un pipeline logistique, tries par valeur absolue."""
    noms = modele[0].get_feature_names_out()
    coef = pd.Series(modele[-1].coef_[0], index=noms)
    return coef.reindex(coef.abs().sort_values(ascending=False).index)


# --- 2. Score de merite -----------------------------------------------------------

# Multiplicateur du poids du comite pour le revenu et la distance dans le merite (1 = poids
# du comite, 0 = neutralise). Les deux sont gardes au poids du comite : seule la penalite
# regionale est retiree. Les sondes HxBuddy (`sondes/`) favorisaient pourtant la neutralisation
# du revenu (94.43 % contre 92.38 %).
PONDERATIONS_MERITE = {'revenu_familial_estime': 1, 'distance_domicile_campus_km': 1}


def score_contrefactuel(modele, df, reference=None, retrait=1.0, ponderations=None):
    """Probabilite d'octroi du comite, avec la penalite regionale retiree.

    `retrait` : fraction de la penalite retiree (0 = comite, 1 = `eloignee` force a 0,
    > 1 = surcorrection). Le modele etant lineaire en logit, `eloignee = 1 - retrait`
    retire exactement cette fraction.
    `ponderations` : {colonne: multiplicateur} applique au poids du comite pour cette
    colonne, autour de sa moyenne dans `reference` (par defaut `df`) : 1 = inchange,
    0 = neutralise, -1 = effet inverse.
    """
    df = ajouter_eloignee(df)
    reference = df if reference is None else reference
    df['eloignee'] = df['eloignee'] * (1 - retrait)
    for colonne, multiplicateur in (ponderations or {}).items():
        moyenne = reference[colonne].mean()
        df[colonne] = moyenne + multiplicateur * (df[colonne] - moyenne)
    return modele.predict_proba(df[NUMERIQUES + CATEGORIELLES])[:, 1]


def score_merite(modele, df, reference=None, retrait=1.0):
    """Score de merite retenu : penalite regionale retiree, revenu et distance ponderes selon
    `PONDERATIONS_MERITE`.

    `retrait` : 1 pour la cible retenue ; les autres valeurs servent au front de Pareto.
    """
    return score_contrefactuel(modele, df, reference=reference, retrait=retrait,
                               ponderations=PONDERATIONS_MERITE)


# --- 3. Cible d'entrainement et modele corrige -------------------------------------

# Variables du modele corrige : ni la region ni le code postal, ni le revenu ni la distance
# en variables brutes. Leur effet reste dans la cible (poids du comite), et le revenu et la
# region entrent aussi par NOUVELLES_VARIABLES (cote R relative).
VARIABLES_MODELE = [c for c in NUMERIQUES
                    if c not in ['eloignee', *PONDERATIONS_MERITE]] + CATEGORIELLES
NOUVELLES_VARIABLES = ['cote_r_rel_a_region', 'cote_r_rel_au_revenu']


def cible_individuelle(modele, df, reference=None, retrait=1.0):
    """Probabilite d'octroi contrefactuelle de chaque dossier, sachant la decision reelle.

    Contrairement a `score_merite`, qui ne depend que des variables, cette cible garde
    l'information individuelle du comite (abduction, au sens de Pearl). On note p la
    probabilite d'octroi du comite et q le score de merite :
    - comite a accorde : min(1, q / p). Un candidat avantage par un critere retire du merite
      peut descendre (aucun avec PONDERATIONS_MERITE = 1 : la cible reste 1) ;
    - comite a refuse : max(0, (q - p) / (1 - p)). Un candidat penalise peut monter.
    En moyenne sur la decision du comite, la cible vaut exactement q.
    """
    p = score_contrefactuel(modele, df, retrait=0)
    q = score_merite(modele, df, reference, retrait)
    y = df['decision_octroi'].to_numpy()
    return np.where(y == 1, np.minimum(1.0, q / p), np.maximum(0.0, (q - p) / (1 - p)))


def ajouter_attributs(df, reference):
    """Ajoute en place la cote R relative a la region et au revenu.

    - cote_r_rel_a_region : cote R / mediane de la cote R dans la region du dossier ;
    - cote_r_rel_au_revenu : cote R / mediane de la cote R dans la tranche de revenu du dossier
      (15 tranches de meme effectif).
    Les medianes et les bornes viennent de `reference` : les candidats sont mesures avec la meme
    definition que l'historique.
    """
    _, bornes = pd.qcut(reference['revenu_familial_estime'], q=15, retbins=True)
    bornes[0], bornes[-1] = -np.inf, np.inf

    def tranche_revenu(d):
        return pd.cut(d['revenu_familial_estime'], bornes, labels=False)

    mediane_region = reference.groupby('region_administrative')['cote_r_equivalent'].median()
    mediane_revenu = reference['cote_r_equivalent'].groupby(tranche_revenu(reference)).median()
    df.insert(0, 'cote_r_rel_au_revenu', df['cote_r_equivalent'] / tranche_revenu(df).map(mediane_revenu))
    df.insert(0, 'cote_r_rel_a_region',
              df['cote_r_equivalent'] / df['region_administrative'].map(mediane_region))


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


def entrainer_modele_cible(df, cible, estimateur=None, variables_supplementaires=()):
    """Entraine le modele corrige sur une cible souple.

    Variables : `VARIABLES_MODELE` plus `variables_supplementaires` (ex. `NOUVELLES_VARIABLES`).
    `estimateur` : classifieur scikit-learn (regression logistique par defaut).
    """
    estimateur = LogisticRegression(max_iter=3000) if estimateur is None else estimateur
    variables = VARIABLES_MODELE + list(variables_supplementaires)
    numeriques = [c for c in variables if c not in CATEGORIELLES]
    modele = make_pipeline(_pretraitement(numeriques), estimateur)
    X, y, poids = dupliquer_cible_souple(df[variables], cible)
    modele.fit(X, y, **{f'{modele.steps[-1][0]}__sample_weight': poids})
    return modele


# --- 4. Decisions dans le budget ----------------------------------------------------

TAUX_MIN, TAUX_MAX = 0.36, 0.44   # budget : taux d'octroi exige sur les candidats
MARGE = 0.01                       # le seuil est cherche a l'interieur du budget : le taux varie d'un jeu a l'autre


def octroi_top_k(scores, taux=0.40):
    """Accorde la bourse exactement aux `taux` meilleurs scores."""
    scores = np.asarray(scores)
    k = int(round(taux * len(scores)))
    decisions = np.zeros(len(scores), dtype=int)
    decisions[np.argsort(-scores, kind='stable')[:k]] = 1
    return decisions


def appliquer_seuil(proba, seuil):
    """Octroi si la probabilite atteint le seuil."""
    return (np.asarray(proba) >= seuil).astype(int)


def exactitude_attendue(cible, decisions):
    """Exactitude moyenne contre une cible souple : un octroi vaut cible, un refus vaut 1 - cible."""
    cible, decisions = np.asarray(cible), np.asarray(decisions)
    return np.mean(decisions * cible + (1 - decisions) * (1 - cible))


def ecart_parite(decisions, groupes):
    """Ecart de parite demographique : plus grand moins plus petit taux d'octroi entre les groupes."""
    taux = pd.Series(np.asarray(decisions)).groupby(np.asarray(groupes)).mean()
    return taux.max() - taux.min()


def choisir_seuil(proba, cible, taux_min=TAUX_MIN + MARGE, taux_max=TAUX_MAX - MARGE,
                  critere=exactitude_attendue, groupes=None, tolerance_parite=0.005):
    """Seuil de probabilite parmi ceux dont le taux d'octroi est dans [taux_min, taux_max].

    Sans `groupes` : le seuil qui maximise `critere`.
    Avec `groupes` : priorite a la parite. On garde les seuils dont l'ecart de parite est a moins
    de `tolerance_parite` du plus petit ecart atteignable, puis celui qui maximise `critere`.
    Les seuils candidats sont les quantiles de `proba` correspondant a ces taux.
    Retourne (seuil, taux d'octroi obtenu).
    """
    proba = np.asarray(proba)
    seuils = np.unique(np.quantile(proba, np.linspace(1 - taux_max, 1 - taux_min, 161)))
    decisions = [appliquer_seuil(proba, s) for s in seuils]
    scores = np.array([critere(cible, d) for d in decisions])
    if groupes is not None:
        ecarts = np.array([ecart_parite(d, groupes) for d in decisions])
        scores[ecarts > ecarts.min() + tolerance_parite] = -np.inf
    meilleur = seuils[int(np.argmax(scores))]
    return meilleur, np.mean(proba >= meilleur)
