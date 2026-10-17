# Esportazione degli splat tramite trascinamento

## Branch e ambito

Solo locale, branch `feat/splat-drag-export`, worktree `D:/repos/LichtFeld-Studio/.worktrees/splat-drag-export`. Base master ufficiale `d161094a9eac9cd1f22f8c280549d2b5dc3c63e8` del 17 ottobre 2026. Nessun push, PR o modifica a Blender per questo lavoro. L’importazione nativa di riferimento è Blender 5.3; il precedente “3.5” è stato interpretato come refuso, non come una versione verificata.

## Comportamento

Si trascina direttamente la riga di uno splat, anche bloccato, oppure una selezione composta da splat. Il movimento interno all’applicazione resta quello del tree. Uscendo dall’applicazione si passa al drag nativo con effetto copia. Non ci sono maniglie o pannelli ausiliari.

Il drag usa il formato e le opzioni delle preferenze. La modalità “chiedi” e il doppio trascinamento sono stati rimossi, incluso il vecchio valore persistito `ask`, che viene ignorato. Nessun percorso viene chiesto durante il drag. La cartella/desktop ricevente gestisce destinazione e conflitti di nome. Tasto destro → Esporta… apre il pannello normale sugli splat richiesti, con tutte le sue opzioni e il percorso.

Il file viene prodotto **su richiesta dei dati**: enumerare i formati o entrare su un destinatario non avvia automaticamente l’esportazione. Il destinatario può tuttavia richiedere i dati anche prima del rilascio; il protocollo lo consente. I percorsi vengono restituiti solo quando tutti i file sono stati scritti e rinominati atomicamente. Non si consegnano placeholder, percorsi inesistenti o file vuoti. Il payload viene risolto una sola volta, anche con richieste ripetute.

Questo elimina il precedente prerequisito di attendere la preparazione prima di avviare il drag nativo e la richiesta di ripetere il gesto per modelli grandi. Se il destinatario legge i dati, la chiamata attende l’export; un destinatario che imponga propri timeout deve essere verificato sul desktop reale. Non viene inventato un gesto se il pulsante era già stato rilasciato prima dell’avvio del drag nativo.

## Avanzamento durante il trasferimento

Alla richiesta effettiva dei dati viene collegato ai writer esistenti un progresso complessivo della selezione. Il valore è monotono anche quando un encoder riparte da zero per una nuova fase; ogni oggetto occupa una porzione uguale della barra, senza promettere una stima temporale esatta. I callback del worker aggiornano soltanto atomiche, non widget o dati della scena.

Dopo 250 ms può comparire una finestra nativa con testo localizzato e annullamento. Gli export rapidi non fanno lampeggiare finestre. Su Windows si usa `IProgressDialog` della Shell, che aggiorna la propria UI su un thread dedicato; l'apertura attende anche il rilascio del pulsante sinistro per non interferire con il drag. Su Linux si usa un dialogo GTK non modale, senza acquisire il focus all'apertura. Su Wayland il destinatario può richiedere dati prima del rilascio e il progresso può quindi comparire prima: non si inventano seriali o un nuovo gesto per posticiparlo.

La normale coda SDL, il rendering Vulkan e Rml non vengono eseguiti ricorsivamente durante la richiesta dati. Linux itera solo il contesto GTK con lavoro limitato per tick; la connessione GTK della finestra non viene usata come sorgente del drag Wayland. Il lock dei modelli viene rilasciato prima di attendere il worker. Lo snapshot CPU rimane sincrono e non offre ancora un avanzamento granulare durante la copia.

Annulla o chiusura della finestra segnalano il token del writer. La finestra resta posseduta finché il worker ha concluso il rollback; poi viene distrutta. L'annullamento volontario non produce un toast di errore, mentre i veri errori del writer restano fallimenti. Un errore della UI cancella e attende il worker; se il backend nativo non può aprire la finestra, viene registrato un avviso e il trasferimento prosegue. I file parziali e i fratelli del batch non consegnato seguono il rollback già esistente.

## Preferenze e dati

Interfaccia → Albero della scena: PLY, SPZ v4 oppure SOG; qualità massima o grado SH per formato; iterazioni SOG personalizzate 1-100; conversione assi Blender per PLY. Qualità massima mantiene tutti gli SH memorizzati e tutti gli splat non eliminati, anche con SH ridotti nel viewport. PLY è binario float32. SPZ/SOG restano compressi con perdita; il preset SOG usa 100 iterazioni, senza promettere ottimalità matematica.

PLY per Blender converte RDF in `(x, z, -y)` e trasforma insieme geometria, covarianza e SH. SPZ e SOG mantengono le convenzioni dei writer. Blender 5.3 importa nativamente PLY e SPZ, ma non SOG. PLY conserva i valori float32 e resta la scelta di riferimento per i test senza addon. SPZ SH0 sul Blender originale usato in laboratorio può richiedere la correzione upstream separata. Non viene applicata una conversione nascosta da SOG a PLY: il formato scelto nelle preferenze resta quello consegnato.

Le etichette sono tradotte in tutte le dieci lingue del progetto. Errori tecnici dettagliati vanno nel log; la UI mostra un messaggio localizzato. Non viene esposto il testo non tradotto delle eccezioni.

## Componenti condivisi e gestione risorse

I writer sono quelli esistenti `io::save_ply`, `io::save_spz`, `io::save_sog`; le trasformazioni usano `core::transform`, l’estrazione LOD usa `extract_lod_leaves`, i percorsi attraversano `core/native_fs.hpp`. Non sono stati riscritti gli encoder. L’esportazione normale continua a usare il proprio pannello e `AsyncTaskManager`.

Il trasferimento nativo usa un servizio limitato separato: l’esportazione normale può fondere nodi e aggiornare il pannello globale, mentre il drag deve consegnare file distinti al ricevente durante un loop nativo. Il servizio ha un solo worker di processo e al massimo un job in coda. La selezione multipla è un unico job seriale, senza N thread o aggregatori `std::async`. I future usano promise/packaged_task: distruggerli non attende il worker sul thread UI. Il worker viene fermato e unito alla chiusura del servizio.

Il provider conserva identità UUID e generazione, non puntatori al tree o ai modelli. Alla richiesta dati ricontrolla la scena, acquisisce il lock del modello e produce uno snapshot CPU posseduto; poi rilascia il lock prima dell’IO. Lo snapshot richiede ancora tempo e memoria sul thread UI, ma non avviene per un semplice drag interno o annullato senza richiesta dati. Il limite conservativo degli snapshot di un batch è 2 GiB; non è una promessa sul picco totale degli allocator. Per modelli superiori va usata l’esportazione normale. L’addestramento attivo deve essere messo in pausa.

Windows: ownership COM con RAII, OLE bilanciato, ogni lettura riceve il proprio STGMEDIUM, niente eccezioni attraverso COM. GTK: widget, URI, target list, guardia di rientranza e timer hanno rilascio esplicito/RAII; nessuna eccezione attraversa i callback C. Il contesto nativo è posseduto da WindowManager, condiviso fra i pannelli e rilasciato esplicitamente prima di distruggere la finestra e chiamare SDL_Quit. Non si ribindano i global Wayland a ogni ricreazione del tree. Wayland: la connessione SDL è presa in prestito e non disconnessa; sono distrutti solo proxy/coda creati dal backend. I callback dei file descriptor chiudono sempre la pipe e gestiscono il lettore chiuso senza SIGPIPE fatale.

Le scritture sono staged in una cartella UUID e committate mediante rename sullo stesso filesystem. Errore o annullamento rimuovono partial e marker; se un elemento del batch fallisce, vengono eliminati anche i fratelli già completati ma mai consegnati al ricevente.

I file consegnabili rimangono disponibili per lettori differiti. Alla prima esportazione del processo si eliminano soltanto cartelle UUID non simboliche, con il nostro marker, più vecchie di sette giorni sotto la radice dei trasferimenti. File recenti, cartelle non marcate e vecchi artefatti del prototipo senza marker sono preservati. Non esiste cancellazione immediata dopo DoDragDrop, che interromperebbe lettori differiti.

## Linux e Wayland

X11/XWayland: GTK3 su connessione X11 separata, payload `text/uri-list`; non viene consumata la coda eventi SDL. Il timeout del loop di drag è due minuti. Se il display predefinito GTK è Wayland o appartiene a un altro server X, si apre una connessione GTK X11 corrispondente a quella SDL e un widget sul suo screen, poi si chiude solo quella connessione. Non si cambia il display GTK globale. Un GDK_BACKEND imposto dall’utente che vieti X11 viene rispettato e segnalato come incompatibile.

Wayland: collegamento nativo su **quella stessa connessione wl_display di SDL**, tramite proprietà pubbliche SDL3. Il backend crea registry, seat, pointer, keyboard e data-device propri, su una coda privata, senza rimpiazzare listener SDL. Osserva il seriale reale del press e la superficie originaria prima del gesto e li passa a `wl_data_device.start_drag`. Non inventa seriali, non usa connessioni GTK separate e non necessita coordinate globali. L’uscita dalla superficie è rilevata dalle coordinate locali del pointer durante il grab.

Offre solo URI e azione copia. Gestisce più seat, rimozione del seat, Escape, cancellazione, lettore interrotto, fine del trasferimento e timeout. Gli eventi SDL letti dalla connessione restano accodati ai proxy SDL; solo la coda privata viene dispatchata nel loop nativo. Non vengono cambiate clipboard o selezione. La versione minima del data-device manager usata è 3 e del seat 5; se non disponibili il trasferimento fallisce in modo controllato.

## Test e riproducibilità

Windows: configurare `BUILD_SPLAT_TRANSFER_TESTS=ON`, compilare e usare:

```text
ctest --test-dir .local/build --output-on-failure -R "^(splat_transfer_contract|native_drag_contract|file_transfer_progress_contract)$"
```

Il contratto export usa i writer reali: SH0-3, snapshot indipendente, maschera eliminati, assi e trasformazioni mondo, covarianza, opacità/DC, rotazione numerica SH1, LOD foglie, riduzione SH, nomi portabili, SPZ/SOG, input invalidi, annullamento durante scrittura, batch, rollback, errore IO, coda satura, future non bloccante e temporanei scaduti. SPZ/SOG sono verifiche di produzione del file, non round-trip numerici completi di tutti i coefficienti compressi.

Il contratto di avanzamento verifica aggregazione, valori non finiti, assenza di aggiornamenti per future già pronti, annullamento e attesa della chiusura del worker in caso di errore UI. Non apre finestre durante i test headless. Il contratto COM esegue 6.000 cicli: enumerazione senza generazione, generazione una sola volta, letture ripetute con storage indipendente, annullamento senza dati ed errore del provider. Dopo riscaldamento misura handle e memoria privata. Questo non certifica ogni percorso di ogni applicazione ricevente.

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

Non si promette trasformazione SH esatta per scale non uniformi, shear o riflessioni: SH>0 viene rifiutato in questi casi per evitare colori corrotti. SH0 supporta geometria affine non singolare. Trasformazioni non finite, singolari o prospettiche vengono rifiutate. Le LOD caricate parzialmente non vengono spacciate per l’intero modello.

Le applicazioni di test sono state avviate per l’utente; l’automazione non ha completato un drag fisico della nuova versione con avanzamento. Restano da verificare fisicamente: riga → Explorer/Desktop e Blender 5.3 originale su Windows; riga → file manager/Blender in X11 e Wayland GNOME/KDE; selezione multipla, Escape, cambio finestra, oggetti grandi, nomi Unicode e conflitti di nome. I sanitizer e i test di protocollo non sostituiscono questi destinatari reali. Una CI completa Linux Release/Debug non è stata eseguita su questa macchina Windows; sono stati compilati ed eseguiti i backend Linux isolati con GCC 14 e sanitizer.

## Risultati locali del 17 ottobre 2026

Build Windows completata; contratti CTest Windows 3/3 e Linux standalone 3/3 superati. Su Linux GCC 14 compila i backend con warning come errori e ASan/UBSan; il controllo delle perdite è abilitato. Il test COM copre 6.000 cicli; non sostituisce una prova fisica con Explorer. I test Python mirati a export e preferenze sono 89 superati.

Blender originale 5.3.0 Beta, build `346d005c9994`, importa senza addon i PLY SH0-3 generati dal contratto, inclusi nomi Unicode. Due punti per fixture; errore massimo sulle posizioni circa 1,91e-7. Questo verifica l’importatore nativo, non un gesto fisico tra finestre.

La suite CPU Python estesa su Windows ha prodotto 3.762 pass, 66 segnalazioni di fallimento, 40 skip e 410 deselezioni. I 61 identificatori unici falliti sono stati rieseguiti sui sorgenti della base `d161094a9eac9cd1f22f8c280549d2b5dc3c63e8`: tutti e 61 falliscono anche lì, usando lo stesso ambiente Windows e lo stesso runtime compilato locale. Non è una build nativa indipendente del master e non prova la correttezza di ogni percorso runtime. Questi problemi non sono stati corretti in questo branch, come richiesto dall’utente.

È stata inoltre osservata una race intermittente nel log del compilatore fittizio del test preesistente `test_shader_header_target.py`. Il test resta invariato: un’eventuale correzione appartiene a un branch separato. Nessuna baseline del debito o regola di controllo è stata indebolita.

Riferimenti per i backend di avanzamento: [IProgressDialog](https://learn.microsoft.com/en-us/windows/win32/api/shlobj_core/nn-shlobj_core-iprogressdialog), [Gtk.Dialog](https://docs.gtk.org/gtk3/class.Dialog.html). Compatibilità nativa: [note di rilascio Blender 5.3](https://developer.blender.org/docs/release_notes/5.3/rendering/).
