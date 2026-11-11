# Checklist de livraison M3

- [ ] Le dashboard affiche `F02_R002` et `F00_R017` avec les graphes et les preuves correspondantes.
- [ ] Les quatre signaux BMS du CSV s'affichent : `pack_voltage_v`, `pack_current_a`, `temperature_c`, `soc_pct`.
- [ ] L'onglet Rapport déclenche **LangGraph + RAG** entièrement en local.
- [ ] Le rapport affiche les sources sous forme `chemin:Ldébut-Lfin`, et les extraits s'ouvrent.
- [ ] La règle PACK_UNDERVOLTAGE retrouve une preuve de tension inférieure au seuil de 300 V du simulateur.
- [ ] Le rapport indique `NOT_EVALUATED`, une cause non confirmée et un test non effectué.
- [ ] Les exports Markdown + JSON sont téléchargeables.
- [ ] `python -m pytest -q member3_agent_app/tests` est vert sur ton Windows.
- [ ] `git status` propre après commits et `git push origin feature/member3-agentic-ai`.
- [ ] Une Pull Request de ton fork vers `ARWA044/ECU-BMS-Project:main` est préparée après accord de l'équipe.

## Bloquants liés au projet global

La validation fonctionnelle ECU/EOL, les plans OEM, les mesures indépendantes et l'évaluation sur des données automobiles réelles dépendent de futurs livrables M1/M2 et sont **hors périmètre** du prototype actuel.
