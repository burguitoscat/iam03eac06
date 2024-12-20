# Projecte Portcanto

## Descripció
El Port del Cantó és un port de muntanya que uneix les comarques de l’Alt Urgell (Adrall) i el Pallars
Sobirà (Sort). Són 18Km de pujada i 18Km de baixada, que típicament es puja entre 54 i 77min; i es
baixa entre 24 i 36min. Es generaran dades sintètiques que simularan una cursa ciclista entre Adrall i
Sort. Es treballarà sobre aquestes dades per tal d'aconseguir una classificació dels diferents ciclistes.

## Instal·lació
Per instal·lar el projecte, segueix aquests passos:

1. Clona el repositori:
    ```bash
    https://github.com/burguitoscat/iam03eac06
    ```
2. Creem un entorn virutal de Python per poder treballar:
    ```bash
    python -m venv venv_m3
    source ./venv_m3/bin/activate
    ```
3. Instal·la les dependències:
    ```bash
    pip install -r requirements.txt
    ```

## Ús
Per generar el dataset amb dades d'exemple, executa:
```bash
python3 generardataset.py
```
Un cop generades les dades d'exemple, per entrenar el model i mostrar dades i onformes, executa:
```bash
python3 clusterciclistes.py
```

## Contribució
Si vols contribuir al projecte, segueix aquests passos:

1. Fes un fork del repositori.
2. Crea una branca nova (`git checkout -b feature/nova-funcionalitat`).
3. Fes els teus canvis i commiteja'ls (`git commit -am 'Afegeix nova funcionalitat'`).
4. Puja els canvis a la teva branca (`git push origin feature/nova-funcionalitat`).
5. Obre una pull request.

## Llicència
Aquest projecte està sota la llicència MIT. Consulta el fitxer `LICENSE` per a més informació.

## Contacte
Per a qualsevol dubte o suggeriment, pots contactar amb nosaltres a [email@example.com](mailto:jordiburgos77@gmail.com).