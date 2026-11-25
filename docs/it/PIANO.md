# Piano tecnico e sperimentale

Il decoratore lavora su fotografie reali. Riconosce parete e pavimento per collocare quadri e tappeti. Nella stessa foto permette di aggiungere mobili 3D e oggetti su piani d'appoggio. Le prove video costituiscono un'estensione separata.

## Sviluppo ed evidenze

| Fase | Risultato verificabile |
| --- | --- |
| Foto e decorazione 2D | Il [notebook principale](../../notebooks/Decorador_AR.ipynb) mostra maschere, rettifica di parete e pavimento e collocazione in sei viste di ZInD. ADE20K permette di misurare l'IoU per classe. Restano da verificare la geometria con riferimenti indipendenti e i casi di collocazione non riuscita. |
| Mobili 3D su fotografie | L'editor offre l'anteprima e l'inserimento di più elementi. Proietta i mobili a partire dal pavimento, stima una zona d'appoggio su un piano cucina reale e applica correzioni locali della luce e ombre di contatto. |
| Estensione video | TUM RGB-D e ArUco sono esperimenti indipendenti. |
| Consegna | Relazione PDF, codice e guida per riprodurre le prove. Il breve video dimostrativo è facoltativo nella valutazione continua. La valutazione globale richiede un video di presentazione. |

## Protocollo degli esperimenti

- **Parete e pavimento.** Misurare l'IoU rispetto alle annotazioni umane e contare quante fotografie consentono la collocazione di ciascun elemento. Il campione di sviluppo alterna camere da letto, soggiorni e sale da pranzo. Per ogni categoria si selezionano per nome le prime foto di validazione con almeno l'8 % dell'immagine annotato per ognuna delle due classi. Per una valutazione più rappresentativa, riservare foto senza selezionarle in base al risultato e includere stanze con occlusioni o superfici assenti.
- **Decorazione 2D.** Gli spigoli annotati di ZInD forniscono le due omografie. Forma e rotazione sono definite sul piano di ciascun elemento. SegFormer misura la parte che ricade sulla classe prevista. Le sei viste di sviluppo appartengono a una sola abitazione. Per valutare la geometria, annotare in altre foto linee o angoli indipendenti e misurare deviazione angolare ed errore in pixel. Applicare un'omografia e la sua inversa permette di verificare la coerenza numerica.
- **Calibrazione.** Salvare camera, risoluzione ed errore sulle viste di calibrazione. Usare foto riservate e misure fisiche per verificare la precisione fuori dal campione di calibrazione.
- **Ancoraggio.** Misurare percentuale di fotogrammi ancorati, durata delle perdite, tempo di recupero e latenza. Misurare le oscillazioni con la camera ferma e l'errore rispetto a riferimenti fisici indipendenti durante il movimento. L'errore sui quattro angoli usati per stimare la posa misura la coerenza dell'adattamento.
- **Filtro temporale.** Elaborare due volte lo stesso video per confrontare la dispersione a camera ferma e il ritardo o l'errore in movimento. Fissare i parametri sui video di sviluppo e riutilizzarli nelle prove.
- **Profondità e occlusione.** Misurare MAE, RMSE, errore relativo e percentuale di pixel validi. Confrontare le maschere visibili usando la profondità misurata come riferimento e contare i valori mancanti. Esprimere la profondità reale e quella virtuale nello stesso sistema di coordinate della camera.

## Copertura del programma

I codici e i pesi si riferiscono alla sezione 6 di *Enunciados de Prácticas* 2026/2027. La tabella riporta i risultati riproducibili e le verifiche ancora da fare.

| Blocco | Argomenti candidati | Evidenze disponibili | Verifica da completare |
| --- | --- | --- | --- |
| BT1 | BT1a, BT1b, BT1c · 15 % | Versioni delle librerie, visualizzazioni ed editor interattivo. | Definire un elenco di immagini di prova di altre abitazioni. |
| BT2 | BT2a, BT2b, BT2e, BT2f · 20 % | Omografie, rettifica, trasferimento tra viste con rotazione nota e proiezione 3D. | Validare calibrazione e posa con foto proprie. Ripetere la registrazione con traslazione della camera. |
| BT3 | BT3a, BT3d, BT3g · 20 % | Ombre sfumate, regioni d'appoggio e confronto della morfologia in NumPy con OpenCV. | Misurare il contributo della morfologia alla rilevazione degli appoggi. |
| BT4 | BT4a, BT4b, BT4d · 15 % | Punti ORB, corrispondenze e trasferimento del contorno del quadro. | Ripetere il confronto in altre stanze. |
| BT5 | BT5d, BT5f, BT5h · 15 % | IoU, confronto con una CNN, mappe di probabilità e soglie su ADE20K. | Ampliare il campione e misurare il tempo di inferenza di ogni modello. |
| BT6 | BT6a, BT6f · 10 % | SegFormer e stima di profondità monoculare applicata a piani d'appoggio reali. | Misurare con RGB-D gli errori di profondità e di altezza. |
| BT7 | BT7a, BT7c · 10 % | Confronto tra RANSAC e la stima con tutte le corrispondenze. | Studiare la sensibilità alla soglia e includere più coppie di viste. |

Gli argomenti candidati con evidenze nel notebook totalizzano **105 %**. Il programma richiede **100 % per la valutazione continua** e **120 % per quella globale**. Il riconoscimento della copertura spetta al docente. Per arrivare al 120 %, tre estensioni candidate sono la calibrazione (BT2c), la validazione della posa (BT2d) e l'analisi della soglia di RANSAC (BT7d).

## Priorità per la consegna

1. Riservare foto di altre abitazioni e misurare appoggio e prospettiva con riferimenti indipendenti. Confrontare la maschera originale con quella ripulita tramite morfologia e annotare IoU, falsi rifiuti e casi difficili.
2. Registrare un piano tra due foto della stessa stanza prese da posizioni diverse. Usare ORB e confrontare stima robusta e ordinaria contando le corrispondenze accettate e misurando l'errore su punti di controllo.
3. Misurare la profondità su foto RGB-D e verificare se aiuta a nascondere le parti del mobile che si trovano dietro oggetti reali. Per gli appoggi, annotare aree utilizzabili e altezze e contare rilevazioni corrette, omissioni ed errori d'altezza. Per l'illuminazione, confrontare le ombre di contatto con una foto di riferimento prima di aggiungere ombre direzionali.

## Struttura del codice

- `photo.py` e `photo_study.py` gestiscono segmentazione e valutazione su ADE20K.
- `panorama.py` prepara le viste ZInD e converte le annotazioni nel sistema della camera.
- `rectification.py` definisce i piani e la collocazione in prospettiva.
- `placement.py` compone le immagini e misura la sovrapposizione con la superficie. `composition.py` conserva gli elementi e `editor_web.py` collega il notebook all'editor.
- `course_experiments.py` contiene il confronto delle maschere, la registrazione e la dimostrazione dei piani d'appoggio.
- `geometry.py` gestisce camera, proiezioni, errori e stima della posa. `calibration.py` e `marker.py` gestiscono la scacchiera e ArUco.
- `render.py` definisce i modelli parametrici e proietta le facce. `video.py` gestisce lettura, ancoraggio, filtro ed esportazione.
- `demo.py` genera una sequenza riproducibile. `depth.py` stima la profondità e calcola le metriche.

Il lavoro precedente offre riferimenti per calibrazione, flusso ottico e proiezione. Se riusi codice, indicane la provenienza e documenta le aggiunte. [Entrega-Via](https://github.com/PepeGdlT/Entrega-Via)
