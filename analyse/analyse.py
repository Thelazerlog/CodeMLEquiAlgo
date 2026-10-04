"""Fonctions d'analyse : donnees, AUC, importance des variables, equite, tranches, proxys.

Rappel : toute mesure calculee contre `decision_octroi` mesure la fidelite au
comite historique (biaise), pas le merite reel.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CATEGORIELLES = ['programme_etudes', 'region_administrative', 'code_postal_3']
ELOIGNEES = ['Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine']
GROUPES = ['Centre', 'Eloignee']

# Palette categorielle de reference : un role par couleur, partout dans les carnets.
COULEURS_GROUPE = {'Centre': '#2a78d6', 'Eloignee': '#eb6834'}
COULEURS_JEU = {'historique': '#2a78d6', 'candidats': '#1baf7a'}
ENCRE, ENCRE_SECONDAIRE, GRILLE = '#0b0b0b', '#52514e', '#e4e3df'


def groupe_region(df):
    """Regroupe les cinq regions en deux blocs : grands centres / regions eloignees."""
    return np.where(df['region_administrative'].isin(ELOIGNEES), 'Eloignee', 'Centre')


# --- Style des graphiques ---------------------------------------------------------

def styliser(ax, titre=None):
    """Axes discrets : grille legere, sans cadre haut/droite."""
    if titre:
        ax.set_title(titre, color=ENCRE, fontsize=11, loc='left')
    ax.grid(color=GRILLE, lw=0.8)
    ax.set_axisbelow(True)
    for cote in ['top', 'right']:
        ax.spines[cote].set_visible(False)
    ax.tick_params(colors=ENCRE_SECONDAIRE)
    return ax


def formater_milliers(ax, col):
    """Affiche le revenu en milliers de dollars (ex. 50k) pour eviter les chevauchements."""
    if col == 'revenu_familial_estime':
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v / 1000:.0f}k'))


# --- Analyse descriptive des donnees ----------------------------------------------

def resume_colonnes(df):
    """Type, valeurs manquantes, nombre de valeurs distinctes, min et max de chaque colonne."""
    return pd.DataFrame({
        'type': df.dtypes.astype(str),
        'manquantes': df.isna().sum(),
        'distinctes': df.nunique(),
        'min': df.min(numeric_only=True),
        'max': df.max(numeric_only=True),
    }).loc[df.columns]


def stats_par_groupe(df, colonnes, groupe='groupe'):
    """Moyenne, ecart-type, quartiles et asymetrie de chaque colonne, par groupe."""
    lignes = {}
    for col in colonnes:
        for g, s in df.groupby(groupe)[col]:
            lignes[(col, g)] = {
                'moyenne': s.mean(), 'ecart_type': s.std(), 'min': s.min(),
                'q25': s.quantile(0.25), 'mediane': s.median(), 'q75': s.quantile(0.75),
                'max': s.max(), 'asymetrie': s.skew(),
            }
    return pd.DataFrame(lignes).T


def tracer_distributions(df, colonnes, groupe='groupe', couleurs=COULEURS_GROUPE, bins=40):
    """Histogrammes normalises (densite) de chaque colonne, un par groupe, sur des bacs communs."""
    fig, axes = plt.subplots(1, len(colonnes), figsize=(4.2 * len(colonnes), 3.6))
    for ax, col in zip(np.atleast_1d(axes), colonnes):
        if pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() <= 50:
            # Entier a peu de valeurs : un bac par valeur, sinon des bacs vides creent des trous.
            bacs = np.arange(df[col].min() - 0.5, df[col].max() + 1.5)
        else:
            bacs = np.histogram_bin_edges(df[col], bins=bins)
        for g, couleur in couleurs.items():
            ax.hist(df.loc[df[groupe] == g, col], bins=bacs, density=True, histtype='step',
                    lw=2, color=couleur, label=g)
        styliser(ax, col)
        formater_milliers(ax, col)
        ax.set_yticks([])
    np.atleast_1d(axes)[0].legend(frameon=False)
    plt.tight_layout()
    plt.show()


def chevauchement(df, col, groupe='groupe'):
    """Part de chaque groupe comprise dans l'etendue [min, max] de l'autre groupe (support commun)."""
    a, b = (df.loc[df[groupe] == g, col] for g in GROUPES)
    return pd.Series({
        f'{GROUPES[0]} dans etendue {GROUPES[1]}': a.between(b.min(), b.max()).mean(),
        f'{GROUPES[1]} dans etendue {GROUPES[0]}': b.between(a.min(), a.max()).mean(),
    }, name=col)


def repartition(df, col, groupe='groupe'):
    """Effectifs et proportions (en colonne) d'une variable categorielle, par groupe."""
    effectifs = pd.crosstab(df[col], df[groupe], margins=True, margins_name='total')
    proportions = pd.crosstab(df[col], df[groupe], normalize='columns')
    return pd.concat({'effectif': effectifs, 'proportion': proportions}, axis=1)


def taux_octroi_par(df, col, cible='decision_octroi'):
    """Effectif et taux d'octroi par modalite de `col`."""
    return df.groupby(col)[cible].agg(effectif='size', taux_octroi='mean')


def tracer_taux_par_quantile(df, colonnes, groupe='groupe', cible='decision_octroi', q=10):
    """Taux d'octroi par decile de chaque variable, une courbe par groupe.

    Les deciles sont calcules dans chaque groupe : chaque point regroupe environ 10 % du
    groupe, meme quand les distributions se chevauchent peu (distance).
    """
    fig, axes = plt.subplots(1, len(colonnes), figsize=(4.2 * len(colonnes), 3.6), sharey=True)
    for ax, col in zip(np.atleast_1d(axes), colonnes):
        for g, couleur in COULEURS_GROUPE.items():
            s = df[df[groupe] == g]
            bacs = pd.qcut(s[col], q=q, duplicates='drop')
            points = s.groupby(bacs, observed=True).agg(x=(col, 'median'), taux=(cible, 'mean'))
            ax.plot(points['x'], points['taux'], '-o', color=couleur, lw=2, ms=6, mec='white', mew=1.5, label=g)
        styliser(ax, col)
        formater_milliers(ax, col)
    np.atleast_1d(axes)[0].set_ylabel("taux d'octroi", color=ENCRE_SECONDAIRE)
    np.atleast_1d(axes)[0].legend(frameon=False)
    plt.tight_layout()
    plt.show()


def matrice_correlation(df, colonnes, methode='spearman'):
    """Matrice de correlation de `colonnes`."""
    return df[colonnes].corr(method=methode)


def tracer_matrice(matrice, titre):
    """Carte de chaleur divergente (bleu negatif, orange positif, blanc a 0), valeurs affichees."""
    from matplotlib.colors import LinearSegmentedColormap
    divergente = LinearSegmentedColormap.from_list('divergente', ['#2a78d6', '#f4f3f0', '#eb6834'])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    image = ax.imshow(matrice, cmap=divergente, vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrice)), matrice.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(matrice)), matrice.index)
    for i in range(len(matrice)):
        for j in range(len(matrice)):
            ax.text(j, i, f'{matrice.iat[i, j]:.2f}', ha='center', va='center', fontsize=8, color=ENCRE)
    ax.set_title(titre, color=ENCRE, fontsize=11, loc='left')
    ax.tick_params(colors=ENCRE_SECONDAIRE, length=0)
    for cote in ax.spines.values():
        cote.set_visible(False)
    fig.colorbar(image, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()


def comparer_jeux(a, b, colonnes):
    """Moyennes des deux jeux et test de Kolmogorov-Smirnov, par colonne numerique."""
    from scipy.stats import ks_2samp

    lignes = {}
    for col in colonnes:
        test = ks_2samp(a[col], b[col])
        lignes[col] = {'moyenne_historique': a[col].mean(), 'moyenne_candidats': b[col].mean(),
                       'ks_statistique': test.statistic, 'ks_p_valeur': test.pvalue}
    return pd.DataFrame(lignes).T


# --- Modeles ----------------------------------------------------------------------

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


# --- Modele de production (baseline) et variables proxys --------------------------

def encoder_rf(df_entrainement, df_cible, exclues=()):
    """Encodage de la baseline (one-hot, colonnes de `df_cible` alignees sur l'entrainement).

    Les colonnes `exclues` sont retirees avant l'encodage (equite par ignorance).
    """
    retirer = [c for c in ['id_candidat', 'decision_octroi', 'eloignee', 'groupe', *exclues]
               if c in df_entrainement.columns]
    categorielles = [c for c in CATEGORIELLES if c not in exclues]
    X = pd.get_dummies(df_entrainement.drop(columns=retirer), columns=categorielles)
    Xc = pd.get_dummies(df_cible.drop(columns=retirer), columns=categorielles
                        ).reindex(columns=X.columns, fill_value=0)
    return X, Xc


def entrainer_rf(train, test, exclues=()):
    """Foret aleatoire de la baseline, ajustee sur `train` contre `decision_octroi`.

    Retourne (modele, X_test encode, decisions sur `test`).
    """
    from sklearn.ensemble import RandomForestClassifier

    X_tr, X_te = encoder_rf(train, test, exclues)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42)
    rf.fit(X_tr, train['decision_octroi'])
    return rf, X_te, rf.predict(X_te)


def auc_prediction_region(df, colonnes, cv=5):
    """AUC en validation croisee d'un modele qui predit le groupe eloigne a partir de `colonnes`.

    0.5 : les variables ne portent aucune information sur la region ; 1.0 : elles la determinent.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    categorielles = [c for c in colonnes if c in CATEGORIELLES]
    pretraitement = ColumnTransformer(
        [('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorielles)],
        remainder='passthrough')
    modele = make_pipeline(pretraitement, HistGradientBoostingClassifier(max_iter=100, random_state=42))
    y = (groupe_region(df) == 'Eloignee').astype(int)
    return cross_val_score(modele, df[colonnes], y, cv=cv, scoring='roc_auc').mean()
