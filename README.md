# churnlab : prédiction du churn Telco

## Question métier

Une entreprise de télécommunications veut identifier les clients susceptibles de quitter son service. Ce projet construit progressivement un système de prédiction du churn : la **colonne cible** est `Churn` (`Yes` ou `No`). Ce dépôt couvre pour l'instant la base reproductible : un environnement Python déclaré, une source de données identifiable et un script d'inspection. Aucun modèle n'est entraîné à ce stade.

## Installation

Prérequis : Python 3.11, [uv](https://docs.astral.sh/uv/) et Git.

```sh
git clone https://github.com/smaaartt/ML_TD1.git
cd ML_TD1
uv sync --locked
uv run python -c "import churnlab; print(churnlab.__version__)"
```

Le résultat attendu est `0.1.0`. L'option `--locked` impose que `uv.lock` reste cohérent avec `pyproject.toml` : si ce n'est pas le cas, la commande échoue au lieu de résoudre silencieusement un nouvel environnement.

## Données

- **Jeu de données** : IBM Telco Customer Churn (7 043 lignes, 21 colonnes).
- **Source** : [CSV épinglé sur un commit du dépôt IBM](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv)
- **SHA-256 attendu** :

```text
16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
```

Le CSV n'est pas versionné dans Git (voir `.gitignore`). Pour le récupérer et vérifier son empreinte, puis inspecter le jeu de données :

```sh
uv run python scripts/download_data.py
uv run python scripts/inspect_telco.py
```

Le second script vérifie le hash, puis écrit un rapport dans `reports/tp01_data_overview.json`. Valeurs attendues : 5 174 `No`, 1 869 `Yes` (environ 26,54 % de churn), 0 identifiant client dupliqué, 11 valeurs vides dans `TotalCharges`.

Un nom de fichier n'est pas une version de données : deux fichiers `telco.csv` peuvent avoir des octets différents. Le hash identifie les octets utilisés, mais ne prouve ni la qualité des étiquettes ni l'absence de fuite de données.

## Structure du projet

| Dossier | Rôle |
|---|---|
| `src/churnlab/` | Code Python importable du projet |
| `scripts/` | Commandes exécutables : téléchargement, inspection, enregistrement de l'environnement |
| `data/raw/` | Snapshot d'origine (non modifié) et métadonnées de source `telco.source.json` |
| `tests/` | Petites vérifications de comportement (`uv run pytest`) |
| `evidence/tp01/` | Preuves du TP01 : environnements avant/après, rapport, réponses |

## Nature des données

Ce jeu de données est un **instantané de clients** : une ligne par client, à un moment donné. Ce n'est ni un journal d'événements daté, ni une expérience de rétention randomisée. On ne peut donc pas en tirer d'historique temporel ni d'effet causal d'une action de rétention, et il ne faut pas en inventer.

## Limites de la reproductibilité

`uv.lock` fige les versions des paquets, et le hash identifie les données. Cela ne suffit pas à garantir des résultats d'entraînement identiques : le matériel, le système d'exploitation, les graines aléatoires et la configuration d'entraînement ne sont pas encore capturés.