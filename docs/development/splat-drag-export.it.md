# Esportazione degli splat tramite trascinamento

## Ambito e isolamento

Sviluppo esclusivamente locale. Branch `feat/splat-drag-export`, worktree `D:/repos/LichtFeld-Studio/.worktrees/splat-drag-export`, base master ufficiale `d161094a9eac9cd1f22f8c280549d2b5dc3c63e8` del 17 ottobre 2026. Nessun push o PR per questo lavoro. Non modifica Blender e non richiede l’add-on per il trasferimento LichtFeld → Blender. La correzione SPZ Blender pubblicata in precedenza è separata.

Il requisito Blender viene interpretato come 5.3, coerentemente con il lavoro precedente; la conferma del refuso “3.5” è stata richiesta. Blender 3.5 non ha l’importatore nativo di splat usato da questo progetto e non viene dichiarato supportato.

## Decisione sul formato al rilascio

Il drag interoperabile comunica un file con nome ed estensione già determinati. Windows Explorer copia un file offerto con CF_HDROP; le applicazioni ricevono il suo percorso. Linux usa URI di file tramite XDND. Non c’è un percorso di destinazione portabile ritornato alla sorgente con cui sostituire dopo il rilascio un file in un altro formato.

Windows supporta anche file virtuali e Linux X11 ha X Direct Save, ma non costituiscono un unico contratto condiviso con Blender e Wayland. Una richiesta dati OLE può avvenire già durante DragEnter: non basta ricevere GetData per dedurre un rilascio. Occorre distinguere metadati e contenuto, verificare il destinatario e gestire reentrancy e annullamento; non simulare un secondo gesto.

Pertanto:

- Modalità predefinita: formato PLY, qualità massima, conversione assi per Blender. Si trascina direttamente la riga dello splat, senza maniglie.
- La modalità “chiedi” con secondo trascinamento è stata rimossa: non è un flusso accettabile. Il drag usa il formato configurato. La scelta al rilascio resta oggetto di ricerca, non una funzione disponibile.
- Se la preparazione dura più del primo gesto, il rilascio non produce un falso trasferimento: una notifica invita a trascinare nuovamente il file pronto.
- Destinazione cartella o desktop: è il file manager a decidere dove copiare il file e a gestire un eventuale nome già presente.
- Destinazione Blender 5.3: PLY viene importato dal supporto nativo. Non occorrono add-on o associazioni del sistema operativo.
- Tasto destro → Esporta…: apre il pannello di esportazione normale sullo splat cliccato o sugli splat della selezione corrente. Qui si sceglie normalmente il percorso e restano disponibili tutti i formati dell’esportazione manuale.

## Preferenze

Interfaccia → Albero della scena:

- formato di drag predefinito: PLY, SPZ v4 o SOG;
- massima qualità oppure impostazioni personalizzate per formato;
- grado SH per PLY, SPZ e SOG: tutti i coefficienti disponibili oppure 0–3;
- iterazioni di clustering SOG personalizzate: 1–100;
- per PLY, conversione negli assi Blender oppure mantenimento delle coordinate sorgenti.

La qualità massima mantiene tutti gli splat non eliminati e il massimo grado SH realmente memorizzato, anche quando la visualizzazione usa un grado inferiore. PLY è binario float32. SPZ e SOG restano intrinsecamente compressi con perdita di precisione: la dicitura non promette un risultato senza perdita. Il preset massimo SOG usa 100 iterazioni, un limite operativo del preset e non una dimostrazione di ottimalità matematica. SPZ usa il contenitore v4 e il livello Zstd già definito dal writer dell’applicazione: cambiare il livello di compressione non recupera la precisione persa nella quantizzazione.

I formati multifile, streaming, viewer e pubblicazione non vengono ancora offerti dal drag: rimangono nell’esportazione manuale. Questo evita di offrire un singolo file che dipende da risorse non trasferite.

## Contratto dei dati e sicurezza del gesto

Lo snapshot viene acquisito dalla sorgente sotto il mutex di lettura del modello, copiando i tensori su CPU. La preparazione e l’esportazione avvengono su un worker; il file viene reso disponibile solo dopo il completamento e la rinomina atomica. Il writer filtra le righe eliminate. Per un albero LOD si esportano le foglie, senza duplicare i nodi interni. LOD con dati solo parzialmente residenti non è dichiarato esportabile a qualità completa.

La trasformazione nel mondo viene applicata allo snapshot. I dati interni usano la base RDF. Per PLY destinato a Blender la conversione è `(x,y,z) → (x,z,-y)` e viene applicata anche a covarianza e SH. Per SPZ/SOG si mantengono le convenzioni dei rispettivi writer. L’opzione PLY per Blender va disabilitata quando il destinatario richiede le coordinate sorgenti.

Il trasferimento è una copia: non elimina nodi o file. Escape annulla; un pulsante già rilasciato non avvia un drag nativo. UUID e generazione della scena impediscono di riutilizzare uno snapshot di un modello sostituito o modificato. Il nome del nodo viene sanitizzato e prefissato, senza interpretarlo come percorso. I file temporanei vengono conservati per consentire letture differite del destinatario; una politica di pulizia con scadenza resta da definire.

## Piattaforme

- Windows: sorgente OLE CF_HDROP, riusa il backend già provato fisicamente nel prototipo precedente. La nuova build richiede una propria verifica del gesto verso Explorer/desktop e Blender.
- Linux X11/XWayland: sorgente GTK3 `text/uri-list`, senza consumare la coda eventi X11 di SDL. Il supporto GTK3 è già una dipendenza Linux del progetto. Usare `SDL_VIDEODRIVER=x11 GDK_BACKEND=x11` quando la sessione usa Wayland con XWayland.
- Wayland nativo: non implementato. `wl_data_device.start_drag` richiede il seriale dell’implicit grab del client SDL; non è corretto inventarlo su una connessione GTK separata. Occorre un’integrazione che riceva il seriale originale dalla connessione SDL. Non viene dichiarato supportato in base ai soli test X11.
- macOS: non incluso in questa richiesta.

## Limiti attuali da non nascondere

Il drag esporta uno o più nodi splat selezionati, con un file per nodo. Le selezioni miste con gruppi o altri tipi di nodo non vengono esportate parzialmente. Il riordinamento interno resta separato; un nodo bloccato può essere esportato ma non deve essere riordinato o raggruppato tramite il drag. Durante training attivo si richiede una pausa per evitare snapshot incoerenti. Trasformazioni SH con shear, riflessione o scala non uniforme dell’oggetto sono rifiutate anziché perdere silenziosamente la correttezza direzionale; per SH0 il percorso di trasformazione delle covarianze resta disponibile.

Il primo snapshot CPU può richiedere tempo e memoria proporzionali al modello. La modalità “chiedi” è stata rimossa, inclusa la preferenza persistita: vecchi valori `ask` vengono ignorati. Un drop accettato dal sistema non dimostra da solo che il destinatario abbia importato i dati.

## Verifiche

Verifiche completate il 17 ottobre 2026:

- build completa Windows sul nuovo master, incluso il controllo delle firme Python generate;
- CTest `splat_transfer_contract`: superato. Verifica snapshot indipendente, SH0–SH3 anche con visualizzazione SH0, righe eliminate, trasformazioni e covarianze, rotazione SH1, foglie LOD, PLY personalizzato, SPZ, SOG, nomi file e cancellazione;
- Blender originale 5.3.0 Beta `346d005c9994`: importazione nativa PLY SH0–SH3 senza add-on, due splat per fixture, posizione massima errata di circa 1.91e-7, attributi e conteggi verificati;
- 89 test Python delle suite `test_export_panel.py` e `test_preferences_panel.py`: superati, inclusa la selezione di uno o più splat richiesta dal tree;
- backend Linux GTK/X11 compilato e collegato con GCC su Ubuntu 24.04 WSL. Contratti di payload vuoto/mancante, driver assente, Wayland non supportato e pulsante già rilasciato: superati. Il test usa stub SDL per questi controlli e non certifica un drag fisico;
- runtime MCP della build di prova: lettura/scrittura delle preferenze, limiti dei valori, formato non valido e ripristino dei default verificati; profilo isolato, fixture di 64 splat SH0;
- parsing Python, JSON delle traduzioni e controllo whitespace del diff.

Le firme Python sono rigenerate con il target ufficiale `refresh_python_stubs`. Nei test del pannello è stata normalizzata un’aspettativa preesistente sui separatori del percorso COLMAP, così la stessa verifica gira su Windows e Linux senza cambiare il codice COLMAP.

Non verificato: aspetto visivo, apertura del menu destro tramite mouse, dialogo formato tramite gesto reale, drag fisico della nuova build verso cartella/desktop e Blender, drag fisico Linux. La sessione grafica non è disponibile e l’utente è fuori casa; nessuna prova visiva viene dichiarata completata. La sessione di test è stata chiusa, senza toccare altre applicazioni.

Artefatti di dettaglio locali (non versionati): `.local/build-final.log`, `.local/build/Testing/Temporary/LastTest.log`, `.local/python-tests.log`, `.local/blender-native-results.json`, `.local/native-linux-contract.log`, `.local/lab/preferences-test.json`.

Per la futura prova interattiva: eseguibile `.local/build/LichtFeld-Studio.exe`, fixture `.local/lab/source-splats.ply`. Prima provare con impostazioni predefinite: riga del tree → cartella vuota e riga → viewport Blender 5.3 senza add-on. Verificare che la modalità “chiedi” non sia più esposta; poi verificare selezione multipla e tasto destro → Esporta… senza includere altri nodi. SPZ SH0 v4 nel Blender originale può ancora richiedere la correzione upstream separata; SOG non è proposto come formato nativo Blender. Per il trasferimento senza add-on usare PLY.

## Fonti tecniche

- https://learn.microsoft.com/en-us/windows/win32/shell/datascenarios
- https://docs.gtk.org/gtk3/method.Widget.drag_begin_with_coordinates.html
- https://wiki.freedesktop.org/www/Specifications/XDND/
- https://wayland.freedesktop.org/docs/html/apa.html#protocol-spec-wl_data_device


## Indagine: un unico gesto con dati differiti (17 ottobre 2026)

La possibilità non è esclusa. Distinguiamo tre problemi: avviare il drag senza esportare, conoscere il rilascio, scegliere nome ed estensione senza invalidare dati già letti dal destinatario. Nessuno richiede necessariamente un secondo gesto. La precedente conclusione di impossibilità era troppo forte.

### Windows: evidenze e candidato

La documentazione Microsoft di [DoDragDrop](https://learn.microsoft.com/en-us/windows/win32/api/ole2/nf-ole2-dodragdrop) specifica che `QueryContinueDrag` riceve i cambiamenti del pulsante; restituendo `DRAGDROP_S_DROP` si passa al `Drop` del destinatario. Questo è un segnale distinto da `IDataObject::GetData`, che da solo non prova che sia avvenuto un rilascio.

Nel sorgente Blender locale, `intern/ghost/intern/GHOST_DropTargetWin32.cc`, `DragEnter` chiama `getGhostType`, che controlla `CF_HDROP` con `QueryGetData`; `Drop` chiama `getGhostData` e quindi `GetData` per ottenere i percorsi. Non legge `FILECONTENTS`/`FILEDESCRIPTOR`. È quindi plausibile offrire la capacità di fornire percorsi durante il gesto e materializzare il file prima di restituire i dati al drop. Questa è un'inferenza dal sorgente locale, non una prova con l'interfaccia.

Prototipo candidato: oggetto dati proprietario con stato `Dragging -> Released -> Choosing -> Exporting -> Ready` oppure `Cancelled/Failed`. Nessun export o popup in `QueryGetData`/enumerazione. Rilevare il rilascio in `QueryContinueDrag`; valutare se la scelta può essere completata prima di restituire `DRAGDROP_S_DROP`, oppure durante una richiesta dati esplicitamente successiva al rilascio. Consegnare soltanto percorsi reali e completi dopo export e rename atomico. Mai un file vuoto, un'estensione finta o un secondo drag sintetico. La scelta deve essere unica anche con richieste dati ripetute e deve supportare annullamento e protezione dalla reentrancy.

Rischi da misurare: popup RmlUi nel loop OLE richiede rendering e input propri, il loop OLE ordinario non esegue automaticamente il frame applicativo; destinatario in attesa COM, reentrancy, dialogo dietro la finestra target, timeout durante export grande, rilascio su destinazione non valida. Non basta chiamare l'attuale `enqueueModal` da `GetData` e aspettarne il risultato sullo stesso thread.

Per Explorer esistono i [file virtuali](https://learn.microsoft.com/en-us/windows/win32/shell/clipboard#cfstr_filecontents), ad esempio trasferimenti di messaggi email rappresentati come file. Separano descrittore (nome e attributi) e contenuto stream. Confermano che il contenuto può non essere un file già scritto, ma non dimostrano che si possa cambiare estensione dopo che Explorer ha letto il descrittore. Occorre registrare l'ordine effettivo delle richieste per Explorer/cartella e Desktop: `EnumFormatEtc`, `QueryGetData`, `GetData` dei descrittori, rilascio, lettura contenuto. Se il nome è richiesto prima del rilascio, il contenuto differito da solo non risolve la scelta del formato. Non restituire dati incoerenti per lo stesso drag.

### Linux

Il backend X11 Blender locale accetta `text/uri-list`, senza X Direct Save nella lista dei tipi; `XdndActionAsk` è esplicitamente disabilitato. Non possiamo quindi delegare a Blender un menu di scelta dell'azione. GTK documenta che [drag-data-get](https://docs.gtk.org/gtk3/signal.Widget.drag-data-get.html) viene emesso quando il target richiede dati; le richieste possono servire anche a verificare il target durante il movimento, come mostra la documentazione [drag-data-received](https://docs.gtk.org/gtk3/signal.Widget.drag-data-received.html). Anche qui serve distinguere il rilascio dalla richiesta dati.

Candidato X11: offerta MIME iniziale, rilascio osservabile nel backend sorgente, scelta locale, export e risposta URI dopo completamento. Da verificare timeout XDND, popup durante il grab e riceventi che richiedono gli URI prima del drop. XDS è un protocollo aggiuntivo per file manager compatibili, non una soluzione automaticamente condivisa con Blender. Wayland rimane distinto: manca nel backend attuale l'integrazione col seriale del grab della connessione SDL. Nessun supporto nativo Wayland dichiarato.

### Esito e verifiche ancora necessarie

Rimossa la modalità a due gesti; resta il drag nel formato configurato. Prima di esporre nuovamente “chiedi”, servono tracce reali e un prototipo con popup, annullamento ed export differito verso Explorer, Desktop e Blender, poi file manager X11. Una prova automatizzata con un target controllato non sostituisce questi destinatari.

Resta anche un limite separato del backend attuale: esporta prima dell'avvio del drag nativo. Se si rilascia durante la preparazione, il drop non avviene e la cache permette un tentativo successivo. Questo non è la modalità “chiedi” rimossa e non soddisfa ancora il requisito di un gesto per modelli grandi: deve essere risolto dallo stesso trasferimento differito, non presentato come UX finale. Nessuna prova grafica eseguita in questa sessione, come richiesto dall'utente fuori casa.

Verifica dopo la rimozione: 89 test Python passati; `splat_transfer_contract` passato (0,53 s); JSON delle traduzioni validi; nessun riferimento residuo al dialogo o al binding `splat_drag_ask` nel codice e nei test.
