# HIP/ROCm: stato storico prima dell'allineamento a dev

Data del congelamento: 21 ottobre 2026. Branch locale: `feat/rocm-tensor-backend`.
Base immediata prima dello snapshot: `949535f599eaffe32281610cc4e116955141e15d`.
Questo documento descrive lo stato precedente all'allineamento richiesto dall'utente; non costituisce una dichiarazione di integrazione completa o di assenza universale di regressioni.

## Decisione e arresto delle prove

L'utente ha chiesto di fermare gli esperimenti, documentare tutto, studiare l'ultimo `upstream/dev` e allineare HIP localmente, mantenendo questo lavoro come riferimento storico. Non pubblicare branch/PR e non modificare altri checkout o processi di altre chat. Il precedente rinvio del riallineamento e' superato da questa richiesta.

La prova di valutazione `real-hip-stopped-mcmc-eval-oct21-a1` e' terminata; la successiva `...-b1` e' stata interrotta su richiesta dell'utente. Il PID 31404 e il comando della cartella dedicata sono stati verificati prima di arrestarlo. Il runner e' terminato con codice non zero atteso. Non considerare completa la coppia qualitativa. Nessuna compilazione o prova GPU di questa chat resta in corso al congelamento.

## Percorsi e riproducibilita'

- Sorgenti: `D:/repos/LichtFeld-Studio/.worktrees/rocm-tensor-backend`.
- Artefatti: `D:/repos/LichtFeld-Studio/.local/rocm-tensor-study` (di seguito `R`).
- Build: `R/build-application`; applicazione `LichtFeld-Studio.exe`, Core `lfs_core.dll`, binding `src/python/lichtfeld.cp312-win_amd64.pyd`.
- Runtime HIP Core: `src/core/tensor/backend/hip/lfs_hip_runtime.dll` nella build.
- Trainer HIP: `src/training/hip/lfs_hip_training.dll` nella build.
- SDK privato: `R/sdk-env/Lib/site-packages/_rocm_sdk_devel`, compilatore clang 23 / SDK 7.14.1; runtime ordinario di sistema `C:/Windows/System32/amdhip64_7.dll`.
- Hardware: Radeon 890M, gfx1150, wave32 nel port HIP, HIP 0 / Vulkan 0; NVIDIA 5060 Ti Vulkan 1 / CUDA 0.
- Identita' PCI verificate: Radeon `0000:c8:00.0`, NVIDIA `0000:c6:00.0`.
- Build di verifica corrente: CUDA OFF, HIP tensor/trainer/Fast ON, Vulkan tensor/render ON. Questo non equivale alla verifica di ogni combinazione di build.
- Windows: VS 2022 Community, CMake `C:/Program Files/CMake/bin/cmake.exe`, Ninja `C:/Strawberry/c/bin/ninja.exe`.
- Python dei runner: `C:/Users/franz/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe`.
- Massimo due job di compilazione; test GPU seriali, mai sovrapposti alle compilazioni di questa chat.
- Ambiente ordinario: `HIP_VISIBLE_DEVICES=0`, `CUDA_VISIBLE_DEVICES=-1`, `LFS_HIP_FAST_ARENA_GROWTH=1`, `LFS_HIP_FAST_PHASE_HELPERS=1`, `LFS_HIP_FAST_TRIM_UNUSED=1`.

Non ricompilare nella vecchia cartella contro il nuovo checkout: conservarne manifest/binari come evidenza e usare una build separata per il nuovo allineamento.

## Protocollo primario e risultati accettati

Il confronto primario richiesto e' 2.000 step, undistort, max_width 1920, SH0, cap 1.000.000 e progetto/seed comune. Dataset `D:/gaussian_datasets/statua_budda_villaggio`, 82 camere, split originale 71 training / 11 valutazione, test_every 8.

Seed immutato: `D:/repos/LichtFeld-Studio/.local/gpu-backend-preferences/common-initial-2000.licht`; SHA-256 `4273c94a3c29daffc84c490651510d91ac71ba9cfb9eba8aa3abc0fced3f63f7`.
Configurazione risolta: `R/historical-resolved-2000.json`: steps_scaler 1, refine start 33, every 7, stop 1667, Morton 333, salvataggi 467/2000, valutazione 2000, cache CPU. Non applicare di nuovo lo scaler gia' risolto presente nel progetto iniziale.
Runner: `R/run-real-hip-historical.py`.

| Prova completa HIP | Media step 201-1900 | Mediana | p90 | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `real-hip-historical-2000-oct21-r2` | 328,718765 ms | 297,31295 | 597,0236 | 18,208645 | 0,437029 | 0,577670 |
| `real-hip-mcmc-pixels-quality-2000-oct21-r1` | 213,357566 ms | 204,9399 | 332,8532 | 18,290804 | 0,439301 | 0,576331 |

Riduzione della media: 35,094%. La finestra comprende l'evoluzione della topologia; non e' un microbenchmark a geometria fissa. Wall complessivo con eval: 1009,409 contro 815,6975 secondi. Il `steady_ms_per_iter` aggregato di perf_bench include lavori diversi dal solo trace degli step: non scambiarlo con la media 201-1900.

Entrambe terminano a 1M e valutano le stesse 11 viste. Picchi pool Core della coppia: 577.335.508 byte usati, 586.527.264 riservati. Il dato globale del driver (~6,24 GB) non e' VRAM fisica attribuita al processo.

Controllo visivo eseguito su tutte le 11 viste, tavole `mcmc-pixels-quality-all-views-{1,2,3}.png`. Non osservato peggioramento globale; non esiste parita' bit per bit e non migliorano tutte le viste. PSNR migliora in 6/11, SSIM 7/11, LPIPS 5/11; peggiori variazioni per vista: -0,111305 dB, -0,00305748 SSIM, +0,00511187 LPIPS. Dati e hash in `mcmc-pixels-full-quality-summary-oct21.json/.log`.

Un precedente ABBA sul medesimo checkpoint finale, 2000->2220, 142 warmup / 78 steady, aveva confermato 187,821233 -> 149,571449 ms (-20,365%) per la specializzazione multi-pixel MCMC. Questo e' un protocollo diverso dal test completo.

## Altri backend: valori storici e limiti

Valori gia' registrati, non rimisurati in questo giro: CUDA 46,9324 ms; Vulkan NVIDIA 103,1184 (ripetizione 124,3676); Metal 74,2845 (ripetizione 73,376); Vulkan macOS 436,4533; AMD Vulkan precedente 229,9766; AMD Vulkan ottimizzato 114,591 ms + 9,936 ms loader.

Il riferimento AMD ottimizzato era una ripresa 200->2200: non e' identico alla partenza fresh del confronto HIP da 2.000 step. Il rapporto indicativo 213/(115..125) e' circa 1,7-1,9, non una misura controllata del divario attuale. L'obiettivo di superare il Vulkan ottimizzato non e' dimostrato raggiunto. Il prototipo privato a topologia fissa da circa 68 ms non e' il trainer reale e non entra nella tabella.

Controllo aggiuntivo V-H-H-V sullo stesso checkpoint HIP finale: SHA-256 `f0894923c0f26b35756ec9fd3f96b6d6907664be15b7481f8621dc7cfed5030b`; 2000->2220, 1M dall'inizio, 142 warmup / 78 steady. HIP 144,170741 e 145,559773 ms; Vulkan di questo vecchio branch 256,554869 e 262,056609 ms. Medie 144,865257 e 259,305739 ms. Non usare questa prova per rivendicare il superamento della versione Vulkan piu' recente: cambiano base di codice e protocollo rispetto al riferimento ottimizzato. Split ricostruito senza eval: tutte le 82 camere vengono usate nella ripresa. Artefatto `same-checkpoint-hip-vulkan-oct21-summary.json`.

## Ultimi esperimenti

### Separazione RGB e statistiche MCMC: scartata

Il prototipo `LFS_HIP_FAST_SPLIT_MCMC_STATS=1` eseguiva RGB e statistiche in due kernel, con l'obiettivo di ridurre i registri. Sei casi numerici (19x17, 64x48, 1x1 e due livelli di gradiente) passano sulle tolleranze di immagine, parametri Adam, densita'/errore. ABBA reale: A1 145,970274, B1 164,117023, B2 164,904163, A2 146,319896 ms. Medie 146,145085 -> 164,510593, peggioramento +12,5666%. Loader 0,059-0,064 ms steady. Rimosso dal generatore di produzione e ricompilato il percorso precedente; contratto MCMC di ripristino passato.

Restano solo artefatti locali: `split-stats-prepare_fast-oct21.py`, `split-stats-training-oct21.dll`, `split-stats-abba-summary-oct21.json`, `split-stats-archived-contract-oct21-results.json`. L'ultima coppia di log numerici archiviati usa esplicitamente la DLL sperimentale; i log `split-stats-contract-*` sono stati ripetuti anche sul modulo ripristinato e non vanno confusi con essa.

### Statistiche scadute dopo resume: promettente, non accettata

MCMC inizializza statistiche anche dopo il limite di refinement, ma il rilascio ordinario scatta solo a iter==stop_refine. Il prototipo HIP-only `LFS_HIP_STOPPED_MCMC_CLEANUP=1` esegue quella transizione alla prima post_backward successiva al limite con statistiche vive. Non cambia noise, ottimizzatore o gli altri backend. E' OFF di default nello snapshot.

ABBA reale identico: A1 145,099910; B1 139,437376; B2 136,084974; A2 146,190624 ms. Medie 145,645267 -> 137,761175, -5,4132%. Buffer Core finali: usati 381.159.668 -> 365.159.668 byte, riservati 447.514.876 -> 431.514.876: -16.000.000 byte. Non attribuire questo guadagno al ciclo fresh che attraversa normalmente stop_refine.

Riepilogo `stopped-mcmc-abba-oct21-summary.json`. Loss finali controllo ~0,14714, candidato ~0,14704: non basta per affermare equivalenza qualitativa. La valutazione del solo controllo e' completa: PSNR 18,661440 / SSIM 0,453005 / LPIPS 0,565439; il candidato e' stato interrotto su richiesta dell'utente. Sono 11 viste del nuovo split di valutazione dopo una ripresa allenata sulle 82 immagini: non chiamarle holdout e non mischiarle con il protocollo primario.

Il nuovo `tests/hip/mcmc_resume_contract.cpp` e relativo target sono scritti ma NON compilati/eseguiti. Coprono prima/esatto/dopo il limite, idempotenza, estensione del limite e confronto con il comportamento Vulkan. Non presentarli come test passati. Studiare la nuova implementazione upstream prima di trasferire questo prototipo.

## Integrazione realmente completata

- 21 famiglie trainer native: Fast, MCMC, Refine, Photo, Adam, SH, Session, TrainingImage, SharedImage, ExtraLoss, Morton, LPIPS, Structure, Blob, Controller, MRNF, Bilateral, PPISP, Geometry, Masks, Gsplat/3DGUT.
- Trainer reale, media ingest e cache CPU, checkpoint/resume, parametri e selezione backend/GPU, viewer e selezioni.
- Layout SH/codec, Adam joint, loss, maschere, geometria, controller e riprese verificati nei rispettivi contratti; 3DGUT/Fast SH3 provati anche nell'app con test brevi, distinti dal confronto principale SH0.
- Identificazione fisica distinta dagli UUID delle API; associazione HIP/Vulkan solo con corrispondenza verificata, Windows D3DKMT e PCI Linux.
- Stessa GPU HIP/Vulkan: modello exportable/imported senza duplicazione, proprietari e viste, sincronizzazione/ownership EXTERNAL e scope che copre registrazione piu' submit. GPU diverse o import non disponibile: host staging con messaggio CLI esplicito.
- Coordinatore CPU con fairness, produttori HIP esclusi durante consumer Vulkan, hook per chiamate native, gestione errori e durata delle risorse.
- Corretto log iniziale del viewer HIP per non dichiarare fallback prima della verifica reale dell'import.
- Corretto il rilascio GUI HIP: cache frustum tratteneva 82 camere, 164 allocazioni, 6.232 byte. Sessioni successive con diagnostica non segnalano risorse HIP trattenute.
- Port nativi per filter_points, update_labels, and_live, histogram_u8, project_points, mark_points_2d e predicati Int32/Int64/UInt32/FP16, riutilizzando le formule Core condivise.

## Verifiche e prove GUI

`full-integration-oct21-r20-results.json`: 38 contratti seriali PASS, inclusi le 21 famiglie trainer, parita' statistiche MCMC, pool, hardware, NN, riduzioni, SH, immagini, warp, memoria esterna, coordinamento/fairness e viewer misto. Sono precedenti al prototipo di rilascio MCMC, il cui test dedicato resta pendente.

Sessione GUI: 2.000 step e crescita a 1M con viewer condiviso attivo, due cambi camera, salvataggio concorrente e checkpoint finale generation 4. Il CKPT di partenza e' estratto senza alterazioni dal seed comune: SHA-256 `55485e15b62870b8a32d960800693f299a49d58233b174b0a9a6db7135c6cd2e`. L'import GUI ricostruisce camere/split: tempi e metriche GUI non sostituiscono il confronto headless storico. Manifest `gui-shared-training-2000-oct21-r1/gui-validation-manifest.json`.

GUI r8, Model da 1M: replace 718.183, remove 336.521, undo 718.183, redo 336.521, intersect regione rimossa 0, add 381.662, clear 0; confermati con selection_get. GUI r9, camera dataset 0: rettangolo e poligono equivalente hanno esattamente gli stessi 7.906 indici, pennello 2.563 tutti contenuti nel rettangolo; clear e uscita normale riusciti. Artefatto `gui-hip-shared-dataset-selection-summary.json`.

## Lavoro ancora aperto

1. Import DLPack da allocazioni ROCm esterne: ancora esplicitamente rifiutato. Export presente. Probe diretto del driver (`foreign-memory-driver-probe-oct21-r1.json`) conferma hipPointerGetAttributes e hipMemGetAddressRange per base/interni; per puntatori CPU/one-past attrs puo' restituire success ma tipo unregistered e device -2: verificare tipo/device/range, non solo lo status. Servono proprietario, stream handshake, offset/overflow, viste, accounting senza doppio conteggio e deleter una sola volta. Nessuna implementazione di registrazione foreign adottata.
2. Circa 30 metodi Tensor generali ancora non implementati in hip_unsupported_ops.cpp. Non confondere una famiglia trainer nativa con copertura di tutta l'API Tensor.
3. Restano compilazioni/verifiche Linux, HIP-off e combinazioni production CUDA+HIP; i probe di convivenza e l'app CUDA-OFF non sostituiscono tali build.
4. Riconciliare selettori, preferenze, fallback, interop, budget UMA e memoria con le implementazioni piu' recenti upstream; non sovrascriverle con versioni vecchie.
5. Rimisurare il riferimento Vulkan aggiornato e HIP con protocollo identico solo dopo l'allineamento e la correttezza. Nessuna garanzia automatica che HIP nativo sia piu' veloce.
6. Completare oppure scartare esplicitamente il prototipo resume MCMC dopo lettura del nuovo dev e verifiche, senza presentarlo come gia' accettato.

Metodi Tensor pendenti al congelamento: compiled_expression, arg_extreme, cdist, nearest_point_indices, camera_frustum_counts, radius_neighbors, radius_neighbor_counts, radius_neighbor_min, point_neighbor_spacing, rasterize_points, ppisp_apply, environment_composite, affine_splat_geometry, morton_sort, kmeans_sh, assign_sh3, decimate_candidates, decimate_merge, nn_linear, nn_attention, nn_norm, nn_conv2d, multinomial, gather_fused_unary, scatter, index_fill, index_put, masked_scatter, strided_upload, pad.

## Regole per l'allineamento successivo

Conservare commit, sorgenti e artefatti di questo storico. Recuperare upstream/dev senza modificare il checkout di altre chat. Usare un nuovo branch locale con nome descrittivo, senza `codex`, e preferibilmente una nuova cartella di build. Prima studiare commit e diff di backend registry, hardware/preferences, ingest/cache, strategie, allocator/pool, Fast/Vulkan, shader, pubblicazione del modello e viewer.

Portare gli adattatori HIP sui contratti correnti; non reintrodurre copie obsolete di CUDA/Metal/Vulkan o dei selettori. Dove upstream ha gia' risolto un problema, mantenere la soluzione upstream e aggiungere soltanto la parte HIP necessaria. Documentare decisione per file/conflitto e provenienza della nuova base. Nessun push o PR. Prima benchmark: configure/build, contratti, fallback/identita'/mixed-GPU/resume; poi confronto 2.000 step identico, qualita' sulle stesse viste, immagini e memoria. Non riutilizzare come prove della nuova base i test passati su questa.
