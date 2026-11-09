# Clavier prédictif (démo web hors-ligne)

HTML, CSS et JavaScript purs, sans dépendance. Le modèle n-gramme (Kneser-Ney) est lu directement
par `js/model.js` ; il est téléchargé une seule fois, gardé dans la Cache API du navigateur, et un
service worker garde l'application : après la première visite, tout marche sans réseau.

## Lancer en local

```bash
# 1. Construire les modèles (depuis ngram/, après prepare_data.py) -> keyboard/models/*.ngm
python src/make_web_model.py

# 2. Servir le dossier (le service worker exige localhost ou https)
cd keyboard && python -m http.server 8000     # puis ouvrir http://localhost:8000
```

Les deux modèles de la démo (`ngram-light.ngm`, 5,2 Mo, et `ngram-standard.ngm`, 11,3 Mo) sont versionnés pour que la page puisse les servir ; les fichiers de vérification `parity_*.json` ne le sont pas. Le modèle vient de Tatoeba (CC-BY 2.0 FR, voir le README du dépôt pour l'attribution).

## Tests

```bash
node --test keyboard/tests/model.test.mjs     # lecteur JS = modèle Python (mini modèle versionné)
node keyboard/tests/parity.mjs keyboard/models/ngram-light.ngm keyboard/models/parity_light.json
python keyboard/tests/e2e.py                  # vrai Chromium : saisie, hors-ligne, latence
```

## Fichiers

| Fichier | Rôle |
|---------|------|
| `js/model.js` | lecteur du format `.ngm`, probabilités et suggestions (avec préfixe) |
| `js/context.js` | texte tapé -> mots de la phrase en cours + début du mot (mêmes règles que `text.py`) |
| `js/store.js` | téléchargement unique + Cache API |
| `js/app.js` | clavier, barre de suggestions, compteur de touches |
| `sw.js` | cache de l'application pour le mode hors-ligne |
| `tests/fixtures/` | mini modèle et réponses de référence calculées par le modèle Python |

Le format `.ngm` est décrit dans `ngram/src/ngramlm/compact.py` (`export_web`).
