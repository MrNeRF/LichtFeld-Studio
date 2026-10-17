# Esportazione degli splat tramite trascinamento

## Branch e ambito

Solo locale, branch `feat/splat-drag-export`, worktree `D:/repos/LichtFeld-Studio/.worktrees/splat-drag-export`. Base master ufficiale `c18db2a79ced0850151df5b874083125804246d1` del 17 ottobre 2026, riallineata senza conflitti. Nessun push, PR o modifica a Blender per questo lavoro. L’importazione nativa di riferimento è Blender 5.3; il precedente “3.5” è stato interpretato come refuso, non come una versione verificata.

## Comportamento

Si trascina direttamente la riga di uno splat o di un gruppo, anche bloccato, oppure una selezione composta da splat e gruppi. Un gruppo viene espanso nei suoi splat discendenti, inclusi sottogruppi, righe chiuse, nodi nascosti e bloccati; ciascuno produce un file distinto. Gli antenati e figli selezionati insieme non duplicano i file. La gerarchia dei gruppi non viene serializzata, mentre le trasformazioni mondo sono applicate a ogni splat. Il movimento interno all’applicazione resta quello del tree. Uscendo dall’applicazione si passa al drag nativo con effetto copia. Non ci sono maniglie o pannelli ausiliari.

Il drag uscente non viene accettato dalla stessa applicazione: su Windows il DropTarget restituisce NONE prima di GetData; su Linux si disabilita temporaneamente la ricezione file/testo tramite SDL, ripristinando entrambi gli stati precedenti alla fine del gesto anche in caso di errore. I drop già in coda vengono scartati. Restano accettati gli import esterni dopo la fine del gesto.

Su Windows il feedback usa l’icona LichtFeld incorporata nell’eseguibile a 64 pixel logici, adattata al DPI della finestra, con badge copia/divieto separati. Non viene disegnata una freccia a mano e non si dipende dall’IDropTargetHelper del destinatario, che Blender non usa. I due cursori vengono allocati una volta per gesto, ripristinati e distrutti a fine operazione. Se manca la risorsa si usa l’icona file della Shell. Su X11 GTK riceve la stessa icona LichtFeld PNG a 64 pixel. Su Wayland la superficie icona contiene il logo a 64 pixel e un badge copia/divieto, per un canvas di 72 pixel; due buffer ARGB premoltiplicati immutabili sono aggiornati dai messaggi target/action. I buffer usano memfd anonimi, senza file temporanei. Superficie e buffer vengono distrutti a fine gesto. Il PNG usa il percorso risorse ufficiale del progetto e, se manca, un file generico. Non serve più wayland-cursor.

Si abilita SDL_HINT_MOUSE_FOCUS_CLICKTHROUGH prima dell’inizializzazione video: il clic che attiva LichtFeld viene consegnato alla UI e può essere l’inizio dello stesso drag. La scelta vale per la finestra intera. Non si fabbricano clic o seriali; su Wayland restano necessari il press e il seriale effettivamente consegnati dal compositor. Il comportamento fisico con le politiche di focus dei desktop Linux richiede la prova reale.

Il quick export offre solo PLY e SPZ, con etichette semplici in tutte le dieci lingue. Un precedente valore SOG nelle preferenze viene normalizzato a PLY; SOG resta disponibile nell’esportazione manuale. La raccolta dei discendenti usa la scena viva e una visita iterativa, senza dipendere dalle righe visibili e senza ricorsione C++. Camere/mesh interni a un gruppo non vengono esportati; selezionarli esplicitamente insieme ai gruppi/splat rifiuta il gesto intero. Splat senza modello caricato, collegamenti mancanti o cicli rifiutano il gruppo intero. Un gruppo vuoto non avvia un export. L’errore UI compare solo se si tenta di uscire dall’applicazione, non per un normale riordino interno. Le regole interne dei nodi bloccati restano protette. Il pannello export manuale mantiene il comportamento esistente di unione in un singolo file.

Il drag usa il formato e le opzioni delle preferenze. La modalità “chiedi” e il doppio trascinamento sono stati rimossi, incluso il vecchio valore persistito `ask`, che viene ignorato. Nessun percorso viene chiesto durante il drag. La cartella/desktop ricevente gestisce destinazione e conflitti di nome. Tasto destro → Esporta… apre il pannello normale sugli splat richiesti, con tutte le sue opzioni e il percorso.

Il file viene prodotto **su richiesta dei dati** (su Windows anche alla conferma di un drop accettato, se il lettore rinvia la richiesta): enumerare i formati o entrare su un destinatario non avvia automaticamente l’esportazione. Il destinatario può tuttavia richiedere i dati anche prima del rilascio; il protocollo lo consente. I percorsi vengono restituiti solo quando tutti i file sono stati scritti e rinominati atomicamente. Non si consegnano placeholder, percorsi inesistenti o file vuoti. Il payload viene risolto una sola volta, anche con richieste ripetute. Su Windows un IDataObject trattenuto dal ricevente può leggere successivamente solo dati già preparati: non avvia frame dall’interno di una futura lettura OLE.

Questo elimina il precedente prerequisito di attendere la preparazione prima di avviare il drag nativo e la richiesta di ripetere il gesto per modelli grandi. Se il destinatario legge i dati, la chiamata attende l’export; un destinatario che imponga propri timeout deve essere verificato sul desktop reale. Non viene inventato un gesto se il pulsante era già stato rilasciato prima dell’avvio del drag nativo.

## Avanzamento e reattività durante il trasferimento

Il drag viene accodato durante il callback del tree e avviato soltanto dopo il completamento del frame. La coda è posseduta dal WindowManager, non dall'elemento Rml: chiudere il pannello sorgente non lascia un puntatore pendente. Una seconda azione nativa viene rifiutata mentre la prima è attiva.

Il provider acquisisce uno snapshot CPU indipendente sotto lock, poi rilascia il lock e avvia trasformazione, compressione e scrittura sul worker. Durante questa attesa il protocollo nativo resta modale sul thread principale, ma può avanzare normali frame completi dell'applicazione. Non si richiama render da un render o da un callback Rml attivo. MainLoop verifica e rifiuta il rientro in un frame attivo; la coda evita un secondo dispatch nativo dai frame intermedi. La coda Wayland privata non viene dispatchata ricorsivamente dall'interno dei suoi listener. Durante l’azione nativa i normali aggiornamenti del cursore UI/resize non competono con il cursore del protocollo; la proprietà del cursore viene rilasciata anche in caso di errore.

Si riusa il job export di AsyncTaskManager, lo stato export già esposto e lo stesso pannello `lfs.export`, con barra e Annulla. Il dialogo Shell/GTK separato del prototipo è stato rimosso. Il pannello viene aperto dopo 250 ms se il trasferimento è ancora in corso e il pulsante è stato rilasciato. Su Wayland il rilascio viene confermato dall’evento dnd_drop_performed. Una lettura anticipata del destinatario può comunque avviare la preparazione prima del rilascio; in quel caso il pannello non deve comparire sotto al puntatore. Un drag attivo impedisce l'avvio di un secondo export, ma non applica il blocco del viewport riservato all'export manuale che può prendere dati in prestito. Le opzioni e il percorso dell'export manuale non vengono richiesti durante il drag.

I callback del worker aggiornano atomiche; soltanto il thread principale pubblica gli aggiornamenti e la conclusione nel job export. La transizione di lavoro del JobRegistry viene eseguita dal worker, rispettando il suo contratto anche in build Debug. Il progresso della selezione è monotono e assegna una porzione uguale a ogni oggetto, senza promettere una stima temporale esatta. Lo stage mostra l’ordinale, il totale e il nome originale del nodo (per esempio Esportazione 2/5: Foresta), usando traduzioni in tutte le dieci lingue. L’indice corrente è atomico, mentre i nomi e la formattazione restano sul thread UI. Anche un cambio di nodo a percentuale invariata aggiorna il testo. La chiusura dell'applicazione richiede l'annullamento e lascia terminare il rollback prima dello shutdown. Le attese eventi durante un drag sono limitate anche se la finestra è minimizzata o il renderer è in recupero.

Annulla usa lo stesso comando dell'export normale e segnala il token del worker. L'annullamento volontario resta distinto dagli errori del writer. Il protocollo del destinatario può richiedere dati prima o dopo il rilascio: non viene inventato un nuovo gesto per cambiarne il momento. I file vengono consegnati soltanto dopo il commit dell'intero batch e l’ultimo controllo di annullamento. Il bookkeeping dei percorsi viene allocato prima di scrivere, così un errore di allocazione della lista non lascia file committati fuori dal rollback.

La copia iniziale dello snapshot CPU rimane sincrona sul thread principale e può produrre una pausa, soprattutto per modelli grandi. Il limite di ammissione è 2 GiB stimati. Compressione e scrittura lunghe non mantengono quel lock e possono aggiornare l'applicazione durante l'attesa; non si promette latenza zero durante lo snapshot né rispetto di qualunque timeout deciso da un destinatario esterno.

## Nomi e file già esistenti

Il nome deriva dal nodo, con prefisso `splat-`, sostituzione dei caratteri di controllo e dei caratteri riservati, rimozione di punti/spazi finali ed estensione scelta. Il prefisso evita i nomi dispositivo Windows come CON e AUX. Nomi oltre 120 byte UTF-8 usano un identificatore di ripiego. Per selezioni multiple la UI aggiunge l’UUID completo del nodo, evitando collisioni dei prefissi corti in gruppi molto numerosi. Anche il ripiego dei nomi lunghi usa l’intero UUID della cartella riservata.

Ogni export riserva atomicamente una nuova cartella UUID. Se esistesse già, fallisce senza toccarla; il rollback rimuove soltanto ciò che questa operazione ha creato. Due export con lo stesso nome hanno cartelle sorgenti diverse e non si sovrascrivono nello staging.

Il protocollo consegna file con azione copia. Explorer o il file manager ricevente sceglie la cartella finale e gestisce file omonimi, chiedendo se sostituire, ignorare o mantenere entrambi secondo le proprie regole. LichtFeld non conosce il percorso finale e non può imporre una politica universale di rinomina o divieto di sovrascrittura al destinatario. Non effettua una propria copia sulla cartella finale e non considera l'accettazione del drop prova che il file sia stato importato. L'export manuale resta nel percorso esistente con le sue conferme.

## Preferenze e dati

Interfaccia → Albero della scena: formato PLY o SPZ e un unico limite SH da 0 a 3, predefinito 3. Il grado esportato è `min(limite, grado memorizzato)`: non vengono inventati coefficienti, anche se il viewport mostra un grado minore. Il limite riguarda i colori dipendenti dalla vista, non elimina punti né riduce la precisione geometrica del PLY. SPZ conserva la propria compressione con perdita. Il vecchio checkbox di qualità e i gradi separati per formato sono stati rimossi. Tutte le spiegazioni del drag sono tooltip tradotti nelle dieci lingue, senza riferimento a SOG nelle preferenze rapide. Le vecchie impostazioni vengono migrate: qualità massima o valore -1 diventano limite 3, altrimenti si conserva il grado del formato attualmente selezionato. Un limite nuovo esplicito prevale sulla configurazione precedente.

L’opzione Assi Blender compare solo per PLY. Converte RDF in `(x, z, -y)`, con Z verticale, e trasforma insieme geometria, covarianza e SH; disattivandola si mantengono gli assi originali della scena. SPZ mantiene le convenzioni del proprio writer e non usa questo checkbox. Blender 5.3 importa nativamente PLY e SPZ; SPZ SH0 sul Blender originale usato in laboratorio può richiedere la correzione upstream separata. SOG resta un formato dell’export manuale e dei test IO, senza opzione nel quick export.

Le etichette sono tradotte in tutte le dieci lingue del progetto. Errori tecnici dettagliati vanno nel log; la UI mostra un messaggio localizzato. Non viene esposto il testo non tradotto delle eccezioni.

## Componenti condivisi e gestione risorse

I writer sono quelli esistenti `io::save_ply`, `io::save_spz`, `io::save_sog`; le trasformazioni usano `core::transform`, l’estrazione LOD usa `extract_lod_leaves`, i percorsi attraversano `core/native_fs.hpp`. Non sono stati riscritti gli encoder. L’esportazione normale continua a usare il proprio pannello e `AsyncTaskManager`.

Il trasferimento nativo usa un servizio limitato separato: l’esportazione normale può fondere nodi e aggiornare il pannello globale, mentre il drag deve consegnare file distinti al ricevente durante un loop nativo. Il servizio ha un solo worker di processo e al massimo un job in coda. La selezione multipla è un unico job seriale, senza N thread o aggregatori `std::async`. I future usano promise/packaged_task: distruggerli non attende il worker sul thread UI. Il worker viene fermato e unito alla chiusura del servizio.

Il provider conserva identità UUID e generazione, non puntatori al tree o ai modelli. Alla richiesta dati ricontrolla la scena, acquisisce il lock del modello e produce uno snapshot CPU posseduto; poi rilascia il lock prima dell’IO. Lo snapshot richiede ancora tempo e memoria sul thread UI, ma non avviene per un semplice drag interno o annullato senza richiesta dati. Il limite conservativo degli snapshot di un batch è 2 GiB; non è una promessa sul picco totale degli allocator. Per modelli superiori va usata l’esportazione normale. L’addestramento attivo deve essere messo in pausa.

Windows: ownership COM con RAII, OLE bilanciato, ogni lettura riceve il proprio STGMEDIUM, niente eccezioni attraverso COM. GTK: widget, URI, target list, guardia di rientranza e timer hanno rilascio esplicito/RAII; nessuna eccezione attraversa i callback C. Il contesto nativo è posseduto da WindowManager, condiviso fra i pannelli e rilasciato esplicitamente prima di distruggere la finestra e chiamare SDL_Quit. Non si ribindano i global Wayland a ogni ricreazione del tree. Wayland: la connessione SDL è presa in prestito e non disconnessa; sono distrutti solo proxy/coda creati dal backend. I callback dei file descriptor chiudono sempre la pipe e gestiscono il lettore chiuso senza SIGPIPE fatale.

Le scritture sono staged in una cartella UUID e committate mediante rename sullo stesso filesystem. Errore o annullamento rimuovono partial e marker; se un elemento del batch fallisce, vengono eliminati anche i fratelli già completati ma mai consegnati al ricevente. Errori di snapshot e scrittura conservano l’indice del nodo tramite SplatTransferError, valido anche per un solo elemento. Stato del job e toast mostrano il nome originale, la posizione nel batch e che non è stato trasferito alcun file, in tutte le dieci lingue. I dettagli del writer rimangono nel log. Anche eccezioni non standard del writer vengono convertite in errori con indice, seguite dal rollback; un errore inatteso della UI conclude comunque il job. Anche il rifiuto preliminare di una gerarchia conserva l’identità del nodo non caricabile o non esportabile; quando il nodo esiste, il toast ne mostra il nome. Per gruppi vuoti il messaggio identifica il gruppo. Tutte queste frasi sono tradotte nelle dieci lingue. Gli errori globali non vengono attribuiti arbitrariamente a un nodo. Il toast viene emesso al termine del loop nativo, anche se l’export è fallito prima dell’apertura del pannello; un annullamento volontario non produce un toast di errore.

I file consegnabili rimangono disponibili per lettori differiti. Alla prima esportazione del processo si eliminano soltanto cartelle UUID non simboliche, con il nostro marker, più vecchie di sette giorni sotto la radice dei trasferimenti. File recenti, cartelle non marcate e vecchi artefatti del prototipo senza marker sono preservati. Non esiste cancellazione immediata dopo DoDragDrop, che interromperebbe lettori differiti.

## Linux e Wayland

X11/XWayland: GTK3 su connessione X11 separata, payload `text/uri-list`; non viene consumata la coda eventi SDL. Il timeout del loop di drag è due minuti. Se il display predefinito GTK è Wayland o appartiene a un altro server X, si apre una connessione GTK X11 corrispondente a quella SDL e un widget sul suo screen, poi si chiude solo quella connessione. Non si cambia il display GTK globale. Un GDK_BACKEND imposto dall’utente che vieti X11 viene rispettato e segnalato come incompatibile.

Wayland: collegamento nativo su **quella stessa connessione wl_display di SDL**, tramite proprietà pubbliche SDL3. Il backend crea registry, seat, pointer, keyboard e data-device propri, su una coda privata, senza rimpiazzare listener SDL. Osserva il seriale reale del press e la superficie originaria prima del gesto e li passa a `wl_data_device.start_drag`. Non inventa seriali, non usa connessioni GTK separate e non necessita coordinate globali. L’uscita dalla superficie è rilevata dalle coordinate locali del pointer durante il grab.

Offre solo URI e azione copia. Gestisce più seat, rimozione del seat, Escape, cancellazione, lettore interrotto, fine del trasferimento e timeout. Gli eventi SDL letti dalla connessione restano accodati ai proxy SDL; solo la coda privata viene dispatchata nel loop nativo. Non vengono cambiate clipboard o selezione. La versione minima del data-device manager usata è 3 e del seat 5; se non disponibili il trasferimento fallisce in modo controllato.

## Test e riproducibilità

Windows: configurare `BUILD_SPLAT_TRANSFER_TESTS=ON`, compilare e usare:

```text
ctest --test-dir .local/build --output-on-failure -R "^(splat_transfer_contract|native_drag_contract|file_transfer_progress_contract|native_modal_frame_contract)$"
```

Il contratto export usa i writer reali: SH0-3, snapshot indipendente, maschera eliminati, assi e trasformazioni mondo, covarianza, opacità/DC, rotazione numerica SH1, LOD foglie, riduzione SH, tutte le 16 combinazioni tra grado disponibile 0-3 e limite 0-3 su PLY e SPZ (metadati e conteggio punti), nomi portabili, SPZ/SOG, input invalidi, annullamento durante scrittura, batch, rollback, errore IO, coda satura, future non bloccante e temporanei scaduti. SPZ/SOG sono verifiche di produzione del file, non round-trip numerici completi di tutti i coefficienti compressi.

Il contratto dei frame modali verifica dispatch dopo il frame, avanzamento di tre frame durante un export sintetico pendente, rifiuto del rendering ricorsivo, coda indipendente dal pannello, errori e chiusura. Il contratto di avanzamento verifica aggregazione, valori non finiti, assenza di aggiornamenti per future già pronti, annullamento e attesa della chiusura del worker in caso di errore UI. La raccolta copre gruppi annidati e sovrapposti, radici ripetute, selezioni miste, gruppi vuoti, modelli non caricati, figli mancanti, cicli e una catena di 20.000 gruppi. Il contratto IO verifica inoltre tre file distinti con trasformazioni diverse, avanzamento monotono fino al 100% e annullamento a metà batch con rimozione dei fratelli già scritti. Non apre finestre durante i test headless. Il test Python del pannello verifica che un job avviato dal drag venga riconosciuto senza mostrare il modulo manuale, che Annulla usi il comando esistente e che il pannello si chiuda alla conclusione. Il contratto COM esegue 6.000 cicli: enumerazione senza generazione, generazione una sola volta, letture ripetute con storage indipendente, annullamento senza dati ed errore del provider. Dopo riscaldamento misura handle, oggetti USER/GDI e memoria privata. Verifica anche la dimensione minima di 64 pixel del cursore creato dalla risorsa icona reale. Questo non certifica ogni percorso di ogni applicazione ricevente.

Linux, senza CUDA o desktop:

```sh
cmake -S tools/splat_transfer_test -B build-native-drag \
  -DLFS_SDL_INCLUDE_DIR=/percorso/include-con-SDL3 \
  -DLFS_DRAG_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-native-drag -j2
ASAN_OPTIONS=detect_leaks=1 ctest --test-dir build-native-drag --output-on-failure
```

Dipendenze: compilatore C++23, GTK3, X11, wayland-client/server, SDL3 headers, pthread. Il contratto Wayland usa libwayland client/server reali con compositor sintetico: seriale/superficie, URI codificati, trasferimento differito, annullamento senza generare, lettore chiuso, eccezione provider, riuso del contesto e rilascio risorse. Non è un test fisico GNOME/KDE. Il contratto X11 copre le guardie e i backend incompatibili, non il percorso completo di un file manager X11.

I contratti sono integrati nei workflow Windows/Ubuntu con `BUILD_FORMAT_TESTS`; il caricamento CPU Linux riusa lo stub driver CUDA già usato dal progetto. La configurazione standalone permette di verificare i backend nativi indipendentemente dal resto dell’applicazione.

Controlli progetto: `tools/check_locale_completeness.py`, `tools/check_ui_hardcoded.py --fail-on-candidates`, `tests/python/test_localization_contracts.py`, `tools/error_debt_census.py --baseline tools/error_debt_baseline.json`, relativi unit test e preflight Windows/percorsi. Formattazione con clang-format 18 di Ubuntu 24.04 e `.clang-format` del progetto, senza regole alternative né aumento della baseline del debito.

## Limiti e verifica fisica

Non si promette trasformazione SH esatta per scale non uniformi, shear o riflessioni: un grado SH esportato maggiore di zero viene rifiutato in questi casi per evitare colori corrotti. SH0 supporta geometria affine non singolare, anche quando proviene da una sorgente di grado superiore ridotta mediante il limite SH. Trasformazioni non finite, singolari o prospettiche vengono rifiutate. Le LOD caricate parzialmente non vengono spacciate per l’intero modello.

L’utente ha confermato che il trascinamento su Windows funziona. Si tratta di una conferma manuale generale, senza una matrice registrata dei singoli destinatari e delle condizioni limite. L’automazione non ha completato un drag fisico della nuova versione. Restano da verificare fisicamente i destinatari Linux: file manager/Blender in X11 e Wayland GNOME/KDE. Non sono certificati individualmente Escape, cambio finestra, oggetti grandi, nomi Unicode e conflitti di nome su ogni destinatario. I sanitizer e i test di protocollo non sostituiscono questi destinatari reali. Una CI completa Linux Release/Debug non è stata eseguita su questa macchina Windows; sono stati compilati ed eseguiti i backend Linux isolati con GCC 14 e sanitizer.

## Risultati locali del 17 ottobre 2026

Build Windows completata; contratti CTest Windows 4/4 e Linux standalone 3/3 superati. Su Linux GCC 14 compila i backend con warning come errori e ASan/UBSan; il controllo delle perdite è abilitato. Il test COM copre 6.000 cicli; non sostituisce una prova fisica con Explorer. I test Python mirati a export e preferenze sono 98 superati, incluso il contratto PLY/SPZ e le etichette di tutte le lingue. Il contratto Windows verifica il rifiuto del self-drop prima di leggere i dati e il conteggio delle risorse cursore/GDI su 6.000 cicli. Il contratto Wayland verifica anche superfici icona, buffer SHM e ripristino della ricezione drop.

Blender originale 5.3.0 Beta, build `346d005c9994`, importa senza addon i PLY SH0-3 generati dal contratto, inclusi nomi Unicode. Due punti per fixture; errore massimo sulle posizioni circa 1,91e-7. Questo verifica l’importatore nativo, non un gesto fisico tra finestre.

La suite CPU Python estesa su Windows ha prodotto 3.762 pass, 66 segnalazioni di fallimento, 40 skip e 410 deselezioni. I 61 identificatori unici falliti sono stati rieseguiti sui sorgenti della base `d161094a9eac9cd1f22f8c280549d2b5dc3c63e8`: tutti e 61 falliscono anche lì, usando lo stesso ambiente Windows e lo stesso runtime compilato locale. Non è una build nativa indipendente del master e non prova la correttezza di ogni percorso runtime. Questi problemi non sono stati corretti in questo branch, come richiesto dall’utente.

È stata inoltre osservata una race intermittente nel log del compilatore fittizio del test preesistente `test_shader_header_target.py`. Il test resta invariato: un’eventuale correzione appartiene a un branch separato. Nessuna baseline del debito o regola di controllo è stata indebolita.

Compatibilità nativa: [note di rilascio Blender 5.3](https://developer.blender.org/docs/release_notes/5.3/rendering/).
