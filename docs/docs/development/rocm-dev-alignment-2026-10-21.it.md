# Allineamento locale HIP a upstream/dev — 21 ottobre 2026

## Riferimenti immutabili

- Nuova base upstream/dev: `9e800d6f653ecf3127ffcfd24dd0bcd8e1620b57`, recuperata il 21 ottobre.
- Nuovo branch: `feat/rocm-dev-alignment`, checkout `D:/repos/LichtFeld-Studio/.worktrees/rocm-dev-alignment`.
- Storico: `feat/rocm-tensor-backend`, snapshot `c247a8bd16c65fb47c7e7069d859b8824e3db99f`, checkout precedente preservato.
- Documento storico: [Stato prima dell'allineamento](rocm-historical-handoff-2026-10-21.it.md).
- Nuovi riferimenti richiesti dall'utente: [GSplat AMD, PR studiate e aggiornamento driver](rocm-gsplat-amd-references-2026-10-21.it.md). Nessuna dipendenza Torch/PyTorch; ottimizzazioni da valutare dopo la correttezza dell'allineamento.
- Artefatti nuovi: `D:/repos/LichtFeld-Studio/.local/rocm-tensor-study/dev-alignment`.
- Non pubblicare branch o PR. Nessuna riscrittura dei riferimenti dello storico o modifica di checkout di altre chat.

Il trasferimento parte dal delta HIP successivo a `80edac55e`, escludendo i due vecchi commit dei selettori gia' superati dalla #3683. Inventario: 298 file modificati dal lavoro HIP, 478 da upstream rispetto all'antenato comune `4ef50cbbc`, 44 in comune. Patch iniziale a tre vie, seguita da revisione dei 18 conflitti; non e' un reimpiego integrale dei vecchi file. I prototipi MCMC e il contratto non ancora eseguito sono esclusi.

## Cosa preservare dalla nuova base

| Area | Sviluppo upstream studiato | Decisione per HIP |
| --- | --- | --- |
| Preferenze/GPU | #3683 (`3b1ecd482`): backend, dispositivi tensor/render indipendenti, preferenza attiva distinta da quella salvata, avviso di riavvio e ripristino della selezione attiva | Estendere con HIP; mantenere policy e test upstream, senza ripristinare i vecchi selettori |
| Hardware software | vendor_id, flag software, memoria di sistema e voce CPU per llvmpipe | Preservare campi e presentazione; aggiungere physical_id verificato per associare HIP/Vulkan senza usare il solo nome |
| Vulkan sort | Radix stabile ballot su AMD compatibile; controlli subgroup/ballot/arithmetic, shared memory e workgroup; cache e fallback generic con log; DeviceLost propagato | Sorgenti upstream invariati; non copiare vecchie varianti shader |
| Vulkan raster | Wave32 su AMD con default 64 solo se supportato; forward a 256 thread richiede almeno 8 subgroup, backward 4; fallback loggato se creazione pipeline fallisce | Preservare esattamente la selezione upstream e i suoi limiti |
| CUDA | Driver entry lazy e contesto ancora utilizzabile dopo eval (#3715, `b0b1145af`) | Non riportare indietro gli entry point o introdurre dipendenze CUDA nel percorso HIP |
| LPIPS | Letture dei punteggi dopo l'ultimo tile (#3713, `ff9ef2e37`) | Preservare aggregazione; estendere soltanto l'ammissione delle convoluzioni native FP16 HIP |
| Media ingest/cache | JPEG, PNG16 per immagini quantizzate, EXR float lossless, budget e shrink dei blob, decode prefetched JPEG su CPU; JXL e AVIF nuovi | Usare i codec e la cache upstream; mantenere il throttling dei grandi sorgenti anche per HIP senza duplicare cache/decoder |
| Nuove feature trainer | ROI/crop damping, crescita MRNF al cap, regolarizzazioni e POPSpa | Aggiornare firme e binding HIP; non ignorare silenziosamente nuovi output o modificatori |
| Metal | Nuove correzioni bilateral, exposure e SH (#3710/#3709/#3712) | Mantenere gli originali upstream, senza trasferire codice precedente sopra di essi |

Al primo controllo la diff rispetto alla nuova base e' vuota per `src/training/vulkan`, `src/training/ops/pair_sort_vulkan.cpp`, `src/core/cuda`, `src/visualizer/window/metal_graphics_context.mm` e `src/training/strategies/mcmc.cpp`. Ripetere questo controllo a fine allineamento.

## Compatibilita' dei file precedenti: requisito esplicito

La rimozione del vecchio kernel di training opacity-ADMM non elimina la lettura di file precedenti. `src/training/checkpoint.cpp` upstream conserva:

1. Consumo e validazione del blocco LFAS/versione 1 e dei tre tensori float32 [N,1].
2. Controllo che il numero di righe coincida con il modello, prima di scartare lo stato vecchio.
3. Avviso CLI quando il checkpoint contiene lo stato opacity-ADMM non piu' usato.
4. Lettura dei parametri precedenti e passaggio alle impostazioni correnti.
5. Resume senza sparsificazione, con avviso CLI, quando il checkpoint era gia' nella fase opacity-ADMM rimossa.

Questi file e i relativi test upstream restano invariati. In particolare preservare `CheckpointLegacySparsityTest.*`, inclusi `AdmmStateFromOlderBuildsIsConsumedAndDropped`, `AdmmSettingsBeforeTheTailNowRunPOPSpa` e `RejectsAdmmStateWithWrongModelRowCount`. I quattro casi equivalenti del contratto HIP dedicato sono ora passati sulla nuova build; questo non equivale ad aver eseguito il target GTest upstream completo. Un vecchio progetto che ora richiede POPSpa va distinto da un errore di lettura: la feature HIP richiede i suoi callback effettivi.

## Collegamenti aggiornati durante l'allineamento

- 18 conflitti iniziali riconciliati. Conservati vendor/software e i campi nuovi della UI, i test aggiunti upstream, i controlli CUDA DLPack sul dispositivo e le dipendenze ROI/POPSpa del trainer.
- Inventario HIP aggiunto ai selettori attuali; PCI/UUID restano distinti. Il limite Vulkan UMA mostrato non viene sostituito dallo spazio di indirizzamento HIP.
- Localizzazioni: conservate anche le nuove stringhe drag/export upstream; aggiunte le stringhe HIP.
- Il generatore ExtraLoss segue l'API corrente, che contiene regolarizzazione scale/opacity. Eliminati dal generatore e dai suoi binding i riferimenti al sorgente opacity-ADMM rimosso.
- Fast HIP aggiornato alle firme upstream che aggiungono IMPACT_SCORE, output sum/peak e cattura dei gradienti. La cattura non deve aggiornare Adam. I kernel multi-pixel sono selezionabili solo senza impact score; il percorso impact mantiene la riduzione e gli atomici originali, da verificare nei contratti.
- Conservati i nuovi modifier cropbox e screen-share. I puntatori/opzioni nuovi non devono essere ignorati.
- Riviste le differenze delle interfacce training tra la vecchia base e il nuovo dev: ExtraLoss, ROI, Fast gradient/impact e Gsplat camera_rays/pop_scores sono i cambi di contratto rilevati. Gsplat HIP ora rispetta anche camera_rays nel modo RGB, senza attivare profondita'/normali. Il kernel nativo supportava gia' questo output: aggiornati il binding e i casi di confronto HIP/Vulkan per camere e tile.
- La tabella della famiglia POPSpa e il callback pop_scores vanno completati prima di ammettere tale feature su HIP. Non equiparare la precedente copertura di 21 famiglie alla copertura di tutte le funzionalita' del nuovo dev.

## Verifiche attuali e limiti

- Configure nuova build: riuscito, directory separata, CUDA OFF / HIP ON / Vulkan ON, stesso set esplicito di ottimizzazioni native accettate nello storico.
- Cache compilatore C/C++: sccache abilitato (`ENABLE_COMPILER_CACHE=ON`, confermato nel configure log). Dipendenze vcpkg riutilizzate. `LFS_FAST_COMPILE=OFF` in questa prima build Release: i componenti C++ pesanti, inclusi i binding Python, sono ottimizzati e la compilazione completa e' lunga. L'utente ha chiesto di non interromperla per cambiarne la configurazione. Riutilizzare questa directory in modo incrementale; per eventuali build funzionali rapide distinte valutare FAST_COMPILE, senza attribuire tempi di training ottenuti con /Od a una build Release ottimizzata.
- Controllo sul CMakeCache dello storico `build-application`: Release, `LFS_FAST_COMPILE=OFF`, `ENABLE_COMPILER_CACHE=OFF`. La nuova cache ON non implica quindi riuso di tutti gli oggetti dello storico. Il livello di ottimizzazione C++ resta coerente; l'aggiornamento driver e le differenze di codice richiedono comunque nuove baseline controllate.
- Dipendenze Windows gia' disponibili usate in sola lettura; manifest install OFF. Prefissi privati JXL e libheif/AVIF esistenti non modificati. SDK HIP precedente invariato.
- Python: **133 test passati** in test_gpu_preferences, test_preferences_panel e test_keymap_bindings. Conflitti negli stub corretti mantenendo gli assert upstream; nessun test disabilitato per far passare l'allineamento.
- Generazione Fast HIP e compilazione di entrambi i runtime nativi (Core e trainer): riuscite nella terza build nativa, exit 0 (`build-native-r3.log`). Non equivale a validazione numerica dei kernel.
- Le prime compilazioni hanno identificato la dipendenza dal sorgente sparsity_optimizer_kernels.cu rimosso e la nuova firma ROI; entrambe corrette. ROI ora riceve anche raggi della camera, profondita' e alpha di superficie. Le costanti di soglia vengono estratte dal sorgente upstream.
- Compilazione completa di applicazione, binding Python e contratti: **riuscita, exit 0**, 1.324 attivita' (`build-application.log`), massimo due job. Anche il giro incrementale, incluso il nuovo contratto checkpoint, e' passato (`build-application-r2.log`, exit 0).
- Suite nativa: **39/39 contratti seriali passati**, senza compilazioni o altri nostri carichi GPU concorrenti; somma del tempo dei contratti 55,57 s. Manifest con hash di Core, moduli nativi ed eseguibili e snapshot driver: `alignment-correctness-r1-results.json`. Nessuna attribuzione di questo tempo alla velocita' di training.
- Fast: passati cattura/accumulo dei gradienti senza mutazioni di parametri/momenti/bounds Adam, confronto normale/impact, derivata numerica opacita' (-5,39980829e-5 contro -5,39910186e-5) e rimozione di un Gaussian (score 2,16906404 contro differenza immagine 2,16906469). Il contratto non prova ancora la cattura SH-rest in tutte le combinazioni di codec.
- ROI: passati raggi espliciti, superficie alpha-weighted, inversione, valori non finiti e riferimento CPU. Gsplat: passati modi RGB/depth/normal, SH0-3, camere/tile e nuova richiesta di raggi nel solo modo RGB senza modificare immagine, alpha o gradienti.
- lfs_hip_checkpoint_read_contract: **4 casi passati** con HIP selezionato, sul lettore upstream invariato: LFAS consumato, impostazioni pre-tail migrate a POPSpa, dimensioni errate rifiutate senza mutare il modello, checkpoint nella tail ripreso senza sparsificazione; riscrittura senza stato ADMM. Il caso pre-tail verifica anche il rifiuto esplicito del training POPSpa non implementato, distinto dalla riuscita della lettura.
- Contratto di ammissione: passato il rifiuto esplicito della famiglia POPSpa incompleta, senza consultare le tabelle di altri backend.
- Memoria/viewer: passati contratti di storage condiviso, accesso esterno e ownership; verificati anche staging, read/write Vulkan e copyback con consumer Radeon 890M e RTX 5060 Ti. Sono verifiche funzionali, non confronti prestazionali.
- Nessun risultato di velocita' o qualita' dello storico viene presentato come misura della nuova base.

## Verifica dell'applicazione reale sulla nuova base

- Profilo e porta MCP privati (45811), bridge del nuovo checkout; nessun collegamento a istanze di altre chat. API e risorse scoperte prima delle operazioni.
- Prima prova: Python non trovava la stdlib perche' il checkout usa dipendenze vcpkg esterne alla directory build. Corretto il solo staging privato, senza cambiare codice del prodotto: codec DLL verificati per hash nelle cartelle app/Python; CPython 3.12.13 nel layout `Lib/`, `DLLs/`, `build/python.exe` gia' previsto da getPythonHome. Manifest `private-codec-runtime-manifest.json` e `private-python-runtime-manifest.json`. Nessun site-packages copiato, nessuna dipendenza Torch/PyTorch. La prova finale r3 non usa PYTHONHOME esterno; `torch_loaded=false` verificato nell'interprete dell'app.
- Aperta una copia del seed storico, SHA verificato. Shell pronta, modello idratato e storage HIP/Vulkan attivo. Il progetto conserva il vecchio percorso Mac: appare correttamente la richiesta di ritrovare il dataset. Non dichiarare questa prova un resume completo del seed con percorsi riallineati.
- Importato poi il dataset locale in un progetto di test separato: 82 camere, 19.555 punti. Smoke GUI **20/20 step** MCMC, SH0, cap 1M, undistort, max_width 1920, resize_factor 1, cache CPU; senza eval (tutte le 82 camere train). Loss finale 0,387630; salvataggio finale .licht riuscito. Questo smoke non e' il protocollo prestazionale/qualitativo primario a 2.000 step e non dimostra la qualita' finale.
- Camera IMG_5416.JPEG: sorgente 6012x8017, immagine preparata 1440x1920. Decoder su CPU, grandi sorgenti GPU limitati a uno per volta. Il log riporta anche il fallback di un buffer ausiliario rendering non esportabile verso staging host sincronizzato; non confonderlo con una duplicazione dei buffer principali del modello, per cui lo storage condiviso risulta attivo.
- Verificato il viewer e ispezionate le schermate della finestra reale dopo il training. Preferenza salvata Vulkan mentre HIP resta attivo: restart_required=true e badge arancione `Restart needed` in basso a destra. Ripristinando HIP: restart_required=false e badge rimosso, senza riavvio. JSON e screenshot `gui-prefs-*-r3.json`, `gui-window-pending-r3.png`, `gui-window-restored-r3.png`; impostazioni del profilo di test ripristinate.
- Salvataggio esplicito e chiusura normale via API dell'app: **exit code 0**, letto prima di rilasciare il processo figlio del bridge (`gui-exit-status-r3.json`). Seed originale verificato invariato dopo le prove. Nessuna istanza di test lasciata aperta.
- Alcune prime chiamate di introspezione/configurazione dell'editor hanno generato errori di script (dir sull'oggetto dinamico, resize_factor 0 non valido, test_every read-only), corretti usando metadati/classi e valori ammessi. Non sono crash o fallimenti di training; i relativi artefatti restano disponibili.

## Stato e ordine del lavoro restante

1. **Completato su Windows HIP-on/CUDA-off:** generatori, runtime Core/trainer, applicazione e binding contro le firme correnti; massimo due job.
2. **Completati i casi descritti sopra:** contratti con oracoli numerici e ownership, cattura gradiente/Adam, impact e ROI/lens. Rimane da estendere la cattura SH-rest a tutti i codec prima di dichiarare copertura completa delle nuove funzioni.
3. **Completati i quattro casi HIP dedicati:** lettura checkpoint precedenti e migrazioni, distinti dal supporto POPSpa. Rimane la verifica del target GTest upstream completo nelle configurazioni appropriate.
4. **Completata la suite di base Windows e lo smoke GUI:** selezione/fallback, memoria condivisa e consumer misti. Restano HIP-off, coesistenza CUDA+HIP e piattaforme richieste, senza attribuire loro i risultati Windows precedenti.
5. **Ancora da implementare:** famiglia POPSpa e callback pop_scores Fast/Gsplat; oggi rifiutati esplicitamente e coperti da test negativo, senza fallback tacito a CUDA/CPU. Le nuove idee GSplat AMD sono raccolte separatamente e non inserite nel percorso di produzione senza verifiche.
6. Solo dopo correttezza e integrazione, nuova baseline controllata Vulkan e HIP: stesso seed, 2.000 step fresh, 1920/undistort/SH0/cap1M/split identico, cache cold/warm, misure per fase, qualita' per vista e controllo visivo.
7. DLPack foreign e gli altri metodi Tensor generali rimangono lavoro distinto da non dichiarare gia' concluso. Rivalutare il prototipo resume MCMC soltanto dopo il nuovo lifecycle upstream.
