# Decoratore AR per interni

Miniprogetto di Visione Artificiale per arredare fotografie reali di interni con quadri, tappeti, mobili 3D e oggetti da appoggiare su ripiani e piani di lavoro.

| Documento | Contenuto |
| --- | --- |
| [Piano tecnico e sperimentale](PIANO.md) | Tecniche, prove riproducibili, copertura dei blocchi BT1–BT7 e verifiche da completare. |
| [Dataset e modelli](DATI.md) | Download, percorsi, annotazioni e protocollo di valutazione. |
| [Validazione della decorazione 2D](VALIDAZIONE_2D.md) | Risultati su ADE20K e comandi per ripetere le prove. |
| [Risorse grafiche e attribuzioni](RISORSE.md) | Provenienza di quadri e tappeti usati nel catalogo. |
| [Proposta iniziale in italiano](../proposta_iniziale_IT.pdf) | PDF della proposta originale. Le istruzioni per la versione attuale sono in questa guida. |

## Avvio in PyCharm

1. Apri la cartella `va-decorador-ar` in PyCharm.
2. Dal terminale, prepara l'ambiente se è la prima installazione.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements-pycharm.txt
   ```

3. In **Settings → Project → Python Interpreter**, scegli `.venv\Scripts\python.exe`.
4. Apri il [notebook principale](../../notebooks/Decorador_AR.ipynb), seleziona il kernel di `.venv` ed esegui tutte le celle. La prima esecuzione prepara le viste di ZInD e carica i modelli necessari. L'archivio ZIP di ADE20K serve per le prove con annotazioni e metriche. [Istruzioni per i dati](DATI.md)
5. Vai alla sezione **Componer una habitación**. Se PyCharm non visualizza l'editor integrato, fai clic su **Abrir editor en el navegador** oppure copia nel browser l'indirizzo locale stampato dalla cella. Lascia attivo il kernel durante la composizione.

### Usare l'editor

Seleziona una stanza, una superficie e un elemento. L'anteprima semitrasparente segue il cursore. Fai clic sulla foto per confermare la posizione. Puoi anche muovere l'anteprima con le frecce e confermare con **Invio** quando la foto è selezionata. Il catalogo offre 20 viste reali della stessa abitazione e 27 elementi.

| Etichetta nell'editor | Significato |
| --- | --- |
| **Habitación** | Stanza o inquadratura. |
| **Superficie** | **Pared** per i quadri, **Suelo** per tappeti e mobili, **En altura** per gli oggetti da appoggiare. |
| **Pieza** | Elemento del catalogo. |
| **Apoyo** | Piano d'appoggio disponibile, per esempio un ripiano o una porzione di piano cucina. |
| **Detectar apoyos** | Individua possibili piani d'appoggio nella foto e ne indica il contorno in ciano. |
| **Ajustes** | Scala, rotazione, orientamento del tappeto, luce e ombra, marcatura manuale di un piano. |
| **Piezas colocadas / Quitar pieza** | Elenco degli elementi confermati e rimozione dell'elemento selezionato. |
| **Guardar PNG** | Esporta gli elementi confermati della vista attuale. Il percorso appare sotto la foto. |

Per provare un piano d'appoggio reale, scegli **cocina 01 → En altura → Detectar apoyos → Plato**. Il contorno ciano delimita la zona utile del piano cucina. Puoi anche collocare un oggetto sul ripiano di un mobile virtuale. Per indicare una superficie reale a mano, usa **Ajustes → Marcar tablero**, scegli l'altezza relativa alla camera e fai clic sui quattro angoli in ordine. La stima automatica richiede una parte visibile del pavimento e può tralasciare superfici strette, coperte o segmentate male.

Un elemento può sporgere dall'inquadratura se la parte visibile coincide con la superficie scelta. L'anteprima facilita il controllo della posizione e **Luz y sombra** permette di confrontare l'effetto del colore locale e delle ombre di contatto sui modelli 3D.

### Orientarsi nel notebook

| Sezione in spagnolo | Cosa mostra |
| --- | --- |
| **Foto real, pared y suelo** | La foto di partenza e il ruolo di SegFormer e delle annotazioni di ZInD. |
| **Rectificar y diseñar la habitación** | Parete e pavimento rettificati e sei esempi di quadri e tappeti in prospettiva. |
| **Evaluar la segmentación** | IoU delle maschere di ADE20K, morfologia, soglia di confidenza e confronto con una CNN. |
| **Colocar muebles 3D en fotos** | Proiezione dei modelli e verifica della loro base sul pavimento. |
| **Registrar un cuadro entre dos vistas** | Trasferimento del contorno con ORB e confronto tra RANSAC e la stima senza scarto delle corrispondenze. |
| **Apoyos reales y luz local** | Stima del piano d'appoggio in cucina e confronto con e senza correzione della luce. |
| **Evaluar profundidad cuando haya mapas medidos** | Calcolo dell'errore tra due mappe di profondità registrate in metri. |
| **Componer una habitación** | Editor interattivo ed esportazione PNG. |
| **Cobertura del enunciado** | Evidenze per i blocchi BT1–BT7 e verifiche da completare. |

Per eseguire le prove su ADE20K, metti `ADEChallengeData2016.zip` in `data/raw/`. Imposta `COMPARE_CNN = False` nella sezione di segmentazione se vuoi evitare il download della CNN.

### Configurazioni di PyCharm

| Configurazione | Operazione |
| --- | --- |
| **01 - Ejecutar notebook principal** | Esegue gli esperimenti del notebook e salva `decorador-ejecutado.ipynb` in `results/`. Per usare l'editor, esegui le celle dal notebook aperto. |
| **02 - Probar foto guiada de ADE20K** | Legge una foto dall'archivio ZIP, segmenta le superfici e colloca il quadro grazie ai riferimenti manuali. |
| **03 - Estudiar 9 fotos de ADE20K** | Esegue la prova automatica e salva il CSV dei risultati. |
| **04 - Demo AR sintética** | Genera la sequenza sintetica, i video aumentati e i CSV. |
| **05 - Generar marcador ArUco** | Salva un marcatore PNG da stampare. |

Le configurazioni 02 e 03 richiedono l'archivio ZIP di ADE20K. Ogni esecuzione stampa il percorso della propria cartella in `results/`.

## Elaborare foto da terminale

Il comando `photo` segmenta parete e pavimento. Con `--planes` proietta le decorazioni usando un JSON con quattro angoli. Senza il JSON esporta le maschere. L'opzione `--experimental` prova la collocazione automatica approssimata.

```powershell
python -m pip install -r requirements-photo.txt
$env:PYTHONPATH = 'src'
python -m decorador_ar photo habitacion.jpg results/mi-habitacion --planes ubicaciones.json
python -m decorador_ar photo-study data/raw/ADEChallengeData2016.zip results/ade-estudio --count 8
```

Nel JSON, usa le chiavi `wall` e `floor`. Inserisci gli angoli nell'ordine alto a sinistra, alto a destra, basso a destra, basso a sinistra. Ci sono due esempi in [assets/planes](../../assets/planes). Per una singola foto, puoi usare `--annotation` per confrontare un PNG indicizzato di ADE20K, oppure `--art` e `--rug` per passare texture personalizzate.

`photo-study` scrive un CSV con l'IoU per superficie, le collocazioni eseguite e la quota di pixel collocati sulla classe corretta. Usa `--offset` per cambiare il punto di partenza e `--metrics-only` per omettere i montaggi. Il file `estudio.json` registra la selezione e la revisione del modello.

## Esperimenti separati sul video

La demo con ArUco e le prove di riproiezione con le pose di TUM sono esperimenti a parte. Il notebook principale lavora su fotografie.

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
python -m decorador_ar demo results/demo-inicial
```

Usa una cartella di destinazione vuota. Il comando genera il video sintetico, la calibrazione, due versioni con elementi virtuali e i rispettivi CSV. Gli MP4 hanno frequenza costante e sono privi di traccia audio.

Per un video reale, fotografa una scacchiera stampata da almeno dieci angolazioni e distanze. Mantieni camera, modalità, risoluzione, orientamento, ottica e zoom del video. Misura il lato di un quadrato. Stampa il marcatore con un margine bianco, fissalo al pavimento e misura il lato del quadrato nero. Nel filmato, includi una pausa a camera ferma, un lento movimento laterale e un tratto in cui il marcatore esce dall'inquadratura e poi ricompare.

```powershell
python -m decorador_ar calibrate data/raw/calibracion results/camera.npz --square-m 0.025
python -m decorador_ar marker results/aruco.png
python -m decorador_ar video data/raw/salon.mp4 results/camera.npz results/salon-ar.mp4 --side 0.20 --size 0.65 0.42 0.44 --xy 0.34 0.12 --yaw 10
```

Le dimensioni si esprimono in metri. `--xy` posiziona il centro del tavolo rispetto al centro del marcatore. Usa `--alpha 0.55` per attivare il filtro temporale. Se il marcatore esce dall'inquadratura o la stima supera la soglia di errore, l'oggetto ricompare alla prima rilevazione valida.

`depth-score` confronta due mappe `.npy` registrate ed espresse in metri. Salva MAE, RMSE, errore relativo assoluto e copertura in JSON. [Protocollo dei dati](DATI.md)

## Eseguire i test

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Struttura

```text
assets/       Risorse visive, attribuzioni ed esempi geometrici
data/         Istruzioni per preparare i dati
docs/         Proposta e documentazione in spagnolo e italiano
notebooks/    Notebook principale ed esperimento diagnostico
results/      Risultati generati in locale
src/          Codice sorgente
tests/        Test automatici
```

Materiale di riferimento del corso di Visione Artificiale 2026/2027 e lavoro precedente di calibrazione e proiezione. [PepeGdlT/Entrega-Via](https://github.com/PepeGdlT/Entrega-Via)
