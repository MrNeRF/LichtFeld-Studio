# Esportazione degli splat tramite trascinamento

## Ambito e isolamento

Sviluppo esclusivamente locale. Branch `feat/splat-drag-export`, worktree `D:/repos/LichtFeld-Studio/.worktrees/splat-drag-export`, base master ufficiale `d161094a9eac9cd1f22f8c280549d2b5dc3c63e8` del 17 ottobre 2026. Nessun push o PR per questo lavoro. Non modifica Blender e non richiede l’add-on per il trasferimento LichtFeld → Blender. La correzione SPZ Blender pubblicata in precedenza è separata.

Il requisito Blender viene interpretato come 5.3, coerentemente con il lavoro precedente; la conferma del refuso “3.5” è stata richiesta. Blender 3.5 non ha l’importatore nativo di splat usato da questo progetto e non viene dichiarato supportato.

## Decisione sul formato al rilascio

Il drag interoperabile comunica un file con nome ed estensione già determinati. Windows Explorer copia un file offerto con CF_HDROP; le applicazioni ricevono il suo percorso. Linux usa URI di file tramite XDND. Non c’è un percorso di destinazione portabile ritornato alla sorgente con cui sostituire dopo il rilascio un file in un altro formato.

Windows supporta anche file virtuali e Linux X11 ha X Direct Save, ma non costituiscono un unico contratto condiviso con Blender e Wayland. Non si deve aprire una finestra modale da una richiesta dati OLE, che può già avvenire durante DragEnter, né simulare un rilascio per riprendere un gesto terminato.

Pertanto:

- Modalità predefinita: formato PLY, qualità massima, conversione assi per Blender. Si trascina direttamente la riga dello splat, senza maniglie.
- Modalità “chiedi”: l’uscita dalla finestra termina il gesto interno e apre solo la scelta del formato. Dopo la scelta viene preparato il file e si trascina di nuovo la stessa riga. Il dialogo dice esplicitamente che serve il secondo gesto e non contiene un percorso.
- Se la preparazione dura più del primo gesto, il rilascio non produce un falso trasferimento: una notifica invita a trascinare nuovamente il file pronto.
- Destinazione cartella o desktop: è il file manager a decidere dove copiare il file e a gestire un eventuale nome già presente.
- Destinazione Blender 5.3: PLY viene importato dal supporto nativo. Non occorrono add-on o associazioni del sistema operativo.
- Tasto destro → Esporta…: apre il pannello di esportazione normale sullo splat cliccato o sugli splat della selezione corrente. Qui si sceglie normalmente il percorso e restano disponibili tutti i formati dell’esportazione manuale.

## Preferenze

Interfaccia → Albero della scena:

- chiedi il formato solo per il drag;
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

Il primo snapshot CPU può richiedere tempo e memoria proporzionali al modello. La modalità “chiedi” richiede due gesti. Un drop accettato dal sistema non dimostra da solo che il destinatario abbia importato i dati.

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

Per la futura prova interattiva: eseguibile `.local/build/LichtFeld-Studio.exe`, fixture `.local/lab/source-splats.ply`. Prima provare con impostazioni predefinite: riga del tree → cartella vuota e riga → viewport Blender 5.3 senza add-on. Poi abilitare “chiedi il formato” e verificare il secondo gesto e l’annullamento; infine selezione multipla e tasto destro → Esporta… senza includere altri nodi. SPZ SH0 v4 nel Blender originale può ancora richiedere la correzione upstream separata; SOG non è proposto come formato nativo Blender. Per il trasferimento senza add-on usare PLY.

## Fonti tecniche

- https://learn.microsoft.com/en-us/windows/win32/shell/datascenarios
- https://docs.gtk.org/gtk3/method.Widget.drag_begin_with_coordinates.html
- https://wiki.freedesktop.org/www/Specifications/XDND/
- https://wayland.freedesktop.org/docs/html/apa.html#protocol-spec-wl_data_device
