# Validazione della decorazione 2D

Queste prove misurano la sovrapposizione della decorazione con la superficie annotata. La qualità visiva, la scala e la prospettiva richiedono controlli separati.

Per la collocazione guidata sono stati scelti a vista gli angoli di due immagini ADE20K. Sul caso `ADE_val_00000506` il quadro è rimasto interamente sulla parete annotata. Su `ADE_val_00001517` il tappeto è rimasto interamente sul pavimento annotato. I riferimenti sono disponibili in [assets/planes](../../assets/planes).

La prova automatica sperimentale usa le foto di validazione del ZIP ufficiale ADE20K SceneParse150. La selezione alterna camere da letto, soggiorni e sale da pranzo con almeno l'8 % dell'immagine annotato come parete e almeno l'8 % come pavimento. Il programma sceglie la posizione a partire dalla foto. La valutazione confronta il risultato con le annotazioni umane.

| Campione | Quadri collocati | Quadro sulla parete | Tappeti collocati | Tappeto sul pavimento |
| --- | ---: | ---: | ---: | ---: |
| Versione iniziale, 9 foto | 9/9 | 74,4 % | 9/9 | 98,4 % |
| Versione rivista, stesse 9 foto | 6/9 | 100 % | 7/9 | 100,0 % |
| Versione rivista, 18 foto riservate | 12/18 | 99,6 % | 11/18 | 99,9 % |

Le percentuali sono medie per elemento misurabile, calcolate sui pixel con annotazione valida. La versione iniziale misurava l'intero poligono. Quella rivista considera i pixel in cui la decorazione raggiunge almeno il 50 % di opacità. Nel campione riservato, il quadro con il risultato peggiore ha ottenuto il 96,3 % e il tappeto con il risultato peggiore il 98,5 %. Il conteggio delle omissioni indica quante foto il metodo riesce ad arredare.

La verifica visiva comprende il quadro che nella prima versione invadeva una finestra (`ADE_val_00000117`), un soggiorno con entrambe le decorazioni (`ADE_val_00000506`) e soggiorni e sale da pranzo del campione riservato. In questa prova storica, il tappeto segue un piano approssimato e il quadro è trattato come frontale.

Per ripetere le prove, salva il ZIP ufficiale nel percorso indicato e lancia i comandi dalla radice del repository. [Istruzioni per ADE20K](DATI.md#foto-e-segmentazione)

```powershell
$env:PYTHONPATH = 'src'
python -m decorador_ar photo-study data/raw/ADEChallengeData2016.zip results/repeticion-ejemplos --count 9
python -m decorador_ar photo-study data/raw/ADEChallengeData2016.zip results/repeticion-reserva --count 18 --offset 27
```
