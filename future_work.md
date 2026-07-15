# Pistes pour la suite

Notes issues d'une exploration du vrai pipeline NILM (`home-control/trainer/`
et `home-control/nilmtk-poc/`, un autre dépôt local) pour voir quels outils
au-delà de numpy/pandas/matplotlib/scikit-learn seraient pertinents à
présenter aux étudiants. Le pipeline `trainer/` est la version de production
de ce que `python_ds` enseigne en simplifié — c'est une bonne source pour
prioriser les prochaines parties du cours sur des besoins réels plutôt que
sur des outils choisis arbitrairement.

## Fait

- **Autocorrélation** (Partie 2, section 2.3) — alternative à la FFT pour
  trouver le rythme d'un signal, sans domaine fréquentiel. Ajoutée après
  avoir vu que `trainer/sessions.py` l'utilise pour le tag automatique du
  frigo par périodicité. Les deux méthodes convergent sur le même résultat
  (~17 min pour le frigo fatigué), ce qui renforce le message pédagogique.

## Recommandé en priorité : Partie 4 — CUSUM et fusion de sessions

Motivation directe et concrète : dans la Partie 3 (section 3.1), le
détecteur de fronts `windowing.slice_on_edges` génère **16 fenêtres** pour
un seul évènement de cuit-vapeur (une par bascule ON/OFF du thermostat),
contre 2 pour le micro-ondes. C'est exactement le problème que
`trainer/sessions.py` résout en production :

> « Un appareil cyclique (sèche-linge, cuit-vapeur) déclenche plusieurs
> fronts montants/descendants rapprochés à cause du thermostat qui
> coupe/relance la résistance. Si on traite chaque front comme un événement
> isolé, un seul créneau de sèche-linge ressemble à s'y méprendre à un four. »

Deux techniques à introduire :

- **CUSUM** (cumulative sum) sur la puissance lissée (médiane 3 points) —
  détection de fronts plus robuste que le simple `abs(delta) > seuil` de
  `windowing.detect_edges`.
- **Fusion de sessions** — regrouper les pulses consécutifs de même
  amplitude séparés de moins de `max_gap` en une seule session d'usage.

Une Partie 4 qui reprend le bug déjà découvert par les étudiants en Partie 3
et le corrige avec l'outil réel serait un excellent fil narratif.

À intégrer dans cette même partie : le **lave-linge de la Partie 3 est trop
régulier**. Son profil actuel (`profil_lave_linge` — chauffe/lavage/essorage,
3 phases fixes) est le même à chaque instance, à la durée et au bruit près.
Un vrai lave-linge a des **programmes différents** (éco 30°, coton 60°,
essorage seul, express...) qui changent le nombre de phases, leur durée
relative et leur amplitude — bien plus irrégulier qu'un cuit-vapeur ou un
frigo, qui ont un seul comportement caractéristique. À modéliser : plusieurs
« programmes » possibles tirés aléatoirement par instance, pas juste un
bruit/jitter autour d'un seul profil. Bon test pour voir si RF/DTW
tiennent le coup face à une classe interne hétérogène (contrairement aux
autres appareils qui sont chacun un profil quasi unique).

## Autres outils identifiés (non priorisés)

| Outil | Source | Ce qu'il apporte |
|---|---|---|
| `scipy.signal.find_peaks`, `scipy.stats.skew/kurtosis` | `trainer/features.py` | déjà installé (dépendance de scikit-learn) ; évite de réinventer certains calculs à la main |
| Run-length encoding | `trainer/features.py` (`_rle`) | durées ON/OFF sans boucle manuelle `np.diff`/`np.where` |
| Coefficient de variation, skewness, kurtosis | `trainer/features.py` | features statistiques au-delà de moyenne/écart-type |
| CNN1D (TensorFlow/Keras) | `trainer/train.py --no-cnn` (optionnel) | classification sur signal brut, alternative au DTW pour la Partie 3 |
| HDBSCAN / UMAP | deps optionnelles de `trainer/` | clustering non supervisé, découverte de signatures sans labels |
| NILMTK (Hart85, CO, FHMM) | `home-control/nilmtk-poc/` | algorithmes historiques de la recherche NILM académique, comparaison avec le RF « maison » |
| [nilmtk-contrib](https://github.com/nilmtk/nilmtk-contrib) | lien fourni par l'utilisateur | modèles deep learning pour NILM (seq2point, seq2seq, RNN, DAE...) — direction recherche actuelle |
| pytest | `trainer/tests/` | tester du code data science sur signaux synthétiques, formalisation de ce que font déjà les cellules `check()` |
| Parquet | tout `trainer/` | format colonnaire adapté aux séries temporelles volumineuses, vs CSV |
| Streamlit | `trainer/app.py` | dashboard interactif pour piloter sync/entraînement/déploiement sans ligne de commande |

## Idée d'enchaînement

1. **Partie 4** : CUSUM + fusion de sessions (corrige le bug de la Partie 3)
2. **Partie 5** (éventuelle) : CNN1D vs Random Forest vs DTW — trois façons de
   classifier, ou clustering non supervisé (HDBSCAN/UMAP) pour découvrir des
   appareils sans labels
3. **Partie 6** (éventuelle) : NILMTK et les algorithmes académiques (Hart85,
   CO, FHMM) en comparaison avec le pipeline « maison »
