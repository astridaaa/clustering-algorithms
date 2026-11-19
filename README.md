# clustering-algorithms

## Aplinka

Reikia **Python 3.12**: `numpy==1.26.4` neturi paketų Python 3.13 ir naujesnėms versijoms.

```
py -3.12 -m venv .venv            # Mac/Linux: python3.12 -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Duomenų paruošimas

`python prepare.py` sukuria `data/paruosta/`. Jei įdiegtos bibliotekų versijos
nesutampa su `requirements.txt`, skriptas sustoja – su kitomis versijomis gaunami
kiti PCA ir t-SNE rezultatai.

`data/paruosta/` yra viena bendra versija, kuria remiasi visi algoritmai:

- duomenis pergeneruoja ir commit'ina vienas žmogus, kiti tik `git pull`;
- su tomis pačiomis versijomis visi failai sutampa, išskyrus t-SNE
  (`tsne_visi.csv`, `stabilumas/tsne_visi.csv`): t-SNE jautrus mažiausiems
  skaitiniams skirtumams, todėl kitame kompiuteryje ar OS jo koordinatės
  gali šiek tiek skirtis;
- jei `prepare.py` leidote tik pasitikrinti, grąžinkite duomenis:
  `git restore data/paruosta`.
