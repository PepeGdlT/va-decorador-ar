# Dataset e modelli

| Fonte | Obiettivo | Selezione |
| --- | --- | --- |
| [ADE20K SceneParse150](https://ade20k.csail.mit.edu/) | Confrontare parete e pavimento con le annotazioni umane. | Foto reali di validazione. Il primo campione richiede almeno l'8 % di area annotata per ciascuna classe. |
| Video propri con ArUco | Misurare stabilità, rilevazioni valide, latenza e recupero. | Registrare filmati distinti per calibrazione, regolazione e valutazione. |
| [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) | Misurare l'errore di profondità in ambienti interni. | Archivio annotato di circa 2,8 GB con 1.449 coppie RGB e profondità allineate. Iniziare con poche scene e fissare poi un elenco per scena. |
| [TUM RGB-D](https://webshare.cvg.cit.tum.de/g/rgbd/dataset/) | Verificare la riproiezione in video con pose di riferimento e profondità registrata. | Due tratti di `fr1/room`, registrati in un ufficio. |
| [Zillow Indoor Dataset (ZInD)](https://github.com/zillow/zind) | Provare la collocazione su fotografie reali. | Venti viste rettilinee di una stessa abitazione non arredata, ricavate dal tour pubblico di esempio. ZInD è destinato a uso accademico non commerciale e richiede attribuzione. |

## Foto e segmentazione

Scarica il ZIP di SceneParse150 e salvalo come `data/raw/ADEChallengeData2016.zip`. Il programma legge direttamente l'archivio. Nelle annotazioni, 0 indica i pixel senza etichetta, 1 la parete e 4 il pavimento. Il codice registra la revisione usata di SegFormer. [ZIP ufficiale del MIT](https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) · [SegFormer B0 addestrato su ADE20K](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)

Il notebook prepara 20 fotografie di una sola abitazione. Usa sei viste per le composizioni 2D, sei per i mobili, due per gli appoggi in cucina e sei aggiuntive tra sala da pranzo, bagno, lavanderia, garage e soggiorno. Gli originali vengono salvati in `data/raw/zind-sample/` e le viste di 960 × 720 pixel in `data/processed/zind-rectified/`. [Tour ufficiale di esempio](https://github.com/zillow/zind/tree/main/sample_tour/000)

Il confronto facoltativo di segmentazione scarica nella cache di Hugging Face i pesi del modello CNN, circa 241 MB. Per saltarlo, imposta `COMPARE_CNN = False` nel notebook. [UperNet con backbone ConvNeXt-Tiny](https://huggingface.co/openmmlab/upernet-convnext-tiny)

## Profondità e appoggi

La dimostrazione in cucina e il pulsante **Detectar apoyos** usano Depth Anything V2 Metric Indoor Small. Il modello viene scaricato nella cache di Hugging Face al primo utilizzo. La scala della previsione viene regolata usando il pavimento annotato e le altezze sono espresse in rapporto all'altezza della camera. I due esempi di cucina servono a provare l'interazione nella stessa abitazione. I pesi per interni sono addestrati su Hypersim e restituiscono una stima in metri. Misurare errore e velocità prima di usarli su video. [Scheda del modello](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf) · [Documentazione degli autori](https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/metric_depth/README.md)

Per NYU, esporta `rawDepths` in matrici 2D espresse in metri e allineate con le immagini RGB. Conserva i dati mancanti. Registra nome della scena, indice originale e partizione nell'elenco dei dati. Verifica visivamente l'orientamento durante la lettura del file MATLAB. Separa le stanze usate per la regolazione da quelle di prova. Valuta il modello metrico senza riallinearlo alla mappa di riferimento della singola foto. Un eventuale confronto di forma con un modello relativo riallineato va riportato a parte.

Prima di fissare il modello, documenta i dati usati per addestrare i pesi e le possibili sovrapposizioni con il campione di prova. Verifica le condizioni d'uso e le attribuzioni. Conserva i dati pesanti e le riprese in uno spazio locale o condiviso con il gruppo con i permessi appropriati.

## Video RGB-D

Salva l'archivio TUM come `data/raw/tum-fr1-room.tgz` (782 MB). Il codice legge i tratti selezionati direttamente dal file compresso. Gli MP4 hanno frequenza costante e il CSV conserva le marcature temporali originali. Il benchmark di Sturm et al. è pubblicato con licenza CC BY 3.0. [Archivio ufficiale](https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz) · [Articolo e specifiche](https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros.pdf) · [Licenza](https://creativecommons.org/licenses/by/3.0/)
