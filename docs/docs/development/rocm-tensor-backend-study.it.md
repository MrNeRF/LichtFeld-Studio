# Backend tensor ROCm/HIP: fattibilità e piano

Studio del 18 ottobre 2026, branch locale `feat/rocm-tensor-backend`.
Base immutabile della chat primaria: `80edac55e`, branch
`feat/gpu-backend-preferences`. Nessuna modifica al checkout della chat primaria,
nessun push e nessuna PR di questo branch. Il branch contiene lo studio, probe compilati e i
moduli runtime e training opt-in (`LFS_HIP_RUNTIME=OFF`,
`LFS_HIP_TRAINING_RUNTIME=OFF`). Il backend tensor HIP
nell'applicazione e la futura opzione `LFS_TENSOR_HIP` **non sono ancora implementati**.

## Esito iniziale

La Radeon 890M di questa macchina esegue realmente operazioni HIP su Windows.
La condivisione di una singola allocazione Vulkan con HIP passa una prova GPU
bidirezionale. Passano anche kernel compilati e un bridge sperimentale tra code
GPU con fence nativa e contatore monotono. La sola attesa del semaforo importato
non è sufficiente nel test con dati reali; non è dimostrato alcun vantaggio
prestazionale del training. I dettagli dei primi fallimenti e dei probe successivi
sono conservati sotto.

Sono quindi soddisfatti i primi prerequisiti per studiare un backend nativo,
ma non quelli per promettere un backend veloce e completo. Prima si completano
toolchain, kernel e interoperabilità; poi si misura il training.

## Compatibilità e ambiente verificato

La nuova matrice ROCm Core SDK 7.14.1 comprende APU RDNA 3.5/gfx1150 e Windows.
La vecchia documentazione Radeon/Ryzen 7.2.1 distingue invece il supporto PyTorch
dalla disponibilità dell'intero HIP SDK. Le due matrici non sono intercambiabili:
una wheel PyTorch funzionante non dimostra che tutti i componenti C++ necessari
siano disponibili. La procedura moderna propone anche un archivio portabile
Windows installabile in una directory dedicata. Fonti:

- [Matrice ROCm Core SDK 7.14.1](https://rocm.docs.amd.com/en/docs-7.14.1/compatibility/compatibility-matrix.html)
- [Installazione ROCm Core SDK 7.14.1](https://rocm.docs.amd.com/en/docs-7.14.1/install/rocm.html)
- [Matrice Ryzen Windows precedente](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/windows/windows_compatibility.html)

Rilevamento locale, senza installazioni né cambiamenti al driver:

| Voce | Risultato |
|---|---|
| CPU rilevata da Windows | Ryzen AI 9 HX 470 con Radeon 890M |
| OS | Windows 11, DisplayVersion 26H2, build 26300.9711 |
| Runtime caricato esplicitamente | `C:\Windows\System32\amdhip64_7.dll` |
| Versione file | `10.0.3679.0` |
| `hipRuntimeGetVersion` / `hipDriverGetVersion` | `70260201` / `70260201`, valori grezzi |
| Dispositivi HIP | Uno: AMD Radeon(TM) 890M Graphics |
| PCI HIP | `0000:c8:00.0` |
| UUID HIP, byte grezzi | `30303030303230303030303030303030` |
| Memoria totale riportata da HIP | 63.339.544.576 byte, non VRAM dedicata fisica |
| Allocazione, memset e verifica completa di 1 MiB | Passano |

Non interpretare la memoria totale HIP come sostituto dei 4 GB BIOS, né sommare
automaticamente tale valore alla RAM del processo. Il primo probe nel sandbox
falliva `hipMemGetInfo` con argomento non valido; lo stesso probe nell'ambiente
normale passa. Non è una prova di incompatibilità della GPU.

Inizialmente non era presente una toolchain HIP configurata tramite
`HIP_PATH`/`ROCM_PATH`. È stato poi installato ROCm 7.14.1 (`rocm[devel]`) in
`.local/rocm-tensor-study/sdk-env`, senza cambiare driver o ambiente globale.
Il compilatore e gli header vengono selezionati esplicitamente nei soli processi
di build. I kernel compilati passano con runtime di sistema `70260201` e runtime
SDK `71460850`, registrando la DLL effettivamente caricata. Questo conferma una
prima compatibilità, non la disponibilità già verificata di ogni libreria ROCm.

## Prove di memoria e sincronizzazione

Probe autonomo MSVC/Vulkan; non collega librerie LichtFeld. Carica soltanto il
runtime HIP di System32 e usa le forme ABI pubbliche controllate sull'header
[HIP 7.2](https://raw.githubusercontent.com/ROCm/HIP/rocm-7.2.0/include/hip/hip_runtime_api.h).
Le dichiarazioni manuali servono al probe iniziale, non devono diventare l'ABI
di produzione: ripetere le prove con gli header dell'SDK scelto.

Una sola allocazione Vulkan device-local da 1 MiB, dedicata ed esportabile tramite
NT HANDLE, è importata e mappata in HIP. Vulkan esegue `vkCmdFillBuffer`; HIP legge
e verifica tutti i byte. HIP esegue un memset; Vulkan copia i dati della stessa
allocazione verso un buffer diagnostico di readback e ne verifica tutti i byte.
I passaggi Vulkan rilasciano/acquisiscono la proprietà della risorsa verso/da
`VK_QUEUE_FAMILY_EXTERNAL`. Attese esplicite separano le operazioni dei due runtime.
Passano 32 cicli alternati, compreso il rilascio finale degli oggetti.

Il buffer CPU di verifica non è un secondo modello del renderer. Questa prova
dimostra aliasing dell'allocazione, non la residenza fisica esatta né il costo
di un modello da un milione di gaussiani. L'assunzione iniziale di memoria
device-local direttamente host-visible/coerente non era valida per i memory type
del buffer esportabile: il probe corretto usa vere operazioni GPU e readback separato.

| Sincronizzazione esterna tentata | Risultato sul runtime corrente |
|---|---|
| Timeline Vulkan, OPAQUE_WIN32, import HIP tipo timeline | `hipImportExternalSemaphore`: invalid argument |
| Binario Vulkan, OPAQUE_WIN32 | Import e signal HIP restituiscono successo; submit Vulkan in attesa fallisce con `VK_ERROR_UNKNOWN` |
| Binario Vulkan, OPAQUE_WIN32_KMT | Stesso fallimento della submit Vulkan |
| Timeline Vulkan esportata come D3D12 fence | Creazione del semaforo Vulkan fallisce con `VK_ERROR_UNKNOWN` |
| Buffer condiviso, completamento host esplicito | Passano tutti i 32 cicli |

Non dedurre supporto dall'esistenza delle entry point HIP o dei valori enum.
Prima di classificare definitivamente i fallimenti occorre ripetere con SDK
coerente, interrogare le proprietà esterne Vulkan specifiche per tipo di handle
e semaforo, e aggiungere callback/layer di validazione. Non è stata stabilita
la causa tra driver, incompatibilità della combinazione e limiti del probe.

### Aggiornamento: kernel compilati e ordinamento GPU sperimentale

Il progetto autonomo `tests/hip` compila con SDK 7.14.1 e target `gfx1150`.
Tre test CTest sono passati; il test di identità fisica aggiornato è passato
anche separatamente. La configurazione OFF non cerca SDK o compilatore e non
compila nulla. Il CMake dell'applicazione non include questo progetto.

| Verifica | Risultato |
|---|---|
| Affine e gather, 1.048.595 elementi, offset 13, coda parziale | Coincidenza esatta con CPU |
| Prefisso/suffisso fuori dalla vista | Invariati |
| Adam, 32 aggiornamenti sugli stessi elementi | Errore massimo parametri `1.12e-8`, primo momento `8.94e-8`, secondo `4.66e-10` |
| Kernel compilati, runtime driver e runtime SDK | Passano entrambi; GPU rilevata `gfx1150`, wave32 |
| Host MSVC/CUDA + DLL Clang/HIP nello stesso processo | Kernel HIP riuscito; memoria CUDA conservata e ancora modificabile |
| Un buffer Vulkan importato HIP | 32 cicli alternati con verifica completa dei byte |
| HIP stream → coda Vulkan → HIP stream | 32 cicli con fence nativa e contatore monotono dedicato |

La direzione supportata per la fence D3D12 è **importazione in Vulkan**, non
esportazione da Vulkan: le proprietà rilevate sono `IMPORTABLE`, senza
`EXPORTABLE`. Il probe crea quindi una fence D3D12 sulla GPU identificata dal
LUID Vulkan e la importa nelle due API. Il semplice signal/wait passa, ma il
successivo test con scritture GPU mostra che la sola attesa HIP del semaforo
importato non protegge correttamente la lettura: il contatore è ancora al valore
del produttore quando HIP legge il contenuto precedente.

Il candidato aggiunge un contatore a 64 bit in una regione dedicata dello stesso
buffer (256 byte di headroom, separati dai parametri). Vulkan pubblica il contatore
dopo le scritture; HIP usa `hipStreamWaitValue64` prima del consumo. La capability
`hipDeviceAttributeCanUseStreamWaitValue` è presente sulla 890M. Passano 32 roundtrip
con veri dati GPU, stream esplicito e readback pinned. Le attese host per controllare
i risultati arrivano dopo il roundtrip, non tra i due handoff GPU verificati.

L'API di memory-wait è Beta nella documentazione dell'SDK. Non promuovere questo
probe direttamente a protocollo di produzione: provare guasti della coda Vulkan,
cancellazione, rilascio delle risorse, overflow dei contatori e crescita del modello.
Una memory-wait senza produttore può restare pendente; la gestione dell'errore va
dimostrata prima di abilitarla nell'app. Il primo percorso integrato può mantenere
lo stesso buffer con completamenti host limitati nel tempo, senza duplicarlo;
questo è distinto dalla successiva ottimizzazione delle attese.

### Identità fisica

L'UUID HIP osservato non coincide con quello Vulkan della stessa scheda.
`hipDeviceGetLuid` non è esportato dal runtime provato; anche i campi LUID delle
proprietà HIP sono vuoti, con node mask zero. Vulkan non espone `VK_EXT_pci_bus_info`.
Il primo probe usava una sola GPU AMD per API. Quello aggiornato abbina invece
PCI HIP a LUID Vulkan → `D3DKMTOpenAdapterFromLuid` → `D3DKMTQueryAdapterInfo`
con `KMTQAITYPE_ADAPTERADDRESS`: il confronto bus/device/function passa. Le query
e i campi sono quelli degli header Windows e della
[documentazione Microsoft](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/d3dkmthk/ns-d3dkmthk-_d3dkmt_adapteraddress).
Il probe rifiuta match ambigui e non assume una mappa PCI valida per domini nonzero.

Per la produzione: LUID quando entrambe le API lo forniscono; altrimenti mappa
PCI verificata tramite API Windows appropriate e proprietà del dispositivo.
Su Linux confrontare PCI/UUID solo quando effettivamente equivalenti. Non usare
il nome commerciale, l'ordinale o l'uguaglianza del solo vendor per autorizzare
un import diretto. Se la corrispondenza è ambigua, disabilitare l'interop diretta
con motivazione nel log. Conservare la richiesta salvata separata dalla scelta runtime.

## Architettura proposta

L'obiettivo è **tensor HIP + training HIP + viewer Vulkan**, non soltanto tensor
HIP con conversioni verso Vulkan a ogni operazione di training. Un backend tensor
da solo non sostituisce rasterizzazione forward/backward, loss e optimizer.

| Punto nel repository | Intervento da progettare |
|---|---|
| `src/core/include/core/gpu_backend_fwd.hpp` | Aggiungere HIP mantenendo i valori CUDA/Vulkan/Metal; aggiornare tabelle e serializzazione |
| `src/core/tensor/backend/backend_registry.cpp` | Dispatch esplicito HIP; eliminare l'assunzione che ogni backend restante sia Vulkan |
| `src/core/include/core/detail/gpu_backend_ops.hpp` | Coprire le 137 dichiarazioni virtuali censite, comprese memoria e sincronizzazione |
| `src/core/include/core/detail/descriptors.hpp` | `ExecContext` oggi contiene `cudaStream_t`: aggiungere un percorso opaco HIP senza reinterpretare uno stream HIP come CUDA |
| `src/core/tensor/backend/cuda/` | Usare come riferimento semantico; nuova implementazione HIP separata |
| `src/core/tensor/backend/cuda/expression_ptx.cpp` | Il generatore PTX è specifico NVIDIA: serve lowering HIP indipendente |
| `src/training/ops/` e registry training | Tabelle HIP native per raster, loss, Adam e strategie |
| `src/core/splat_exportable_storage.cpp`, `splat_block.cpp` | Layout del modello condiviso e crescita corretta; il VMM CUDA non diventa HIP per rinomina |
| `src/core/tensor/backend/tensor_upload.cpp` | Queue HIP, completion, lease del consumer Vulkan e sincronizzazione bidirezionale |
| `src/python/lfs/py_tensor.cpp` | DLPack ROCm corretto, con ordinali e ownership; mai presentare HIP come CUDA |
| Preferenze e inventario hardware | ROCm/HIP solo se compilato; GPU supportate e motivi di indisponibilità; riuso dell'indicatore di riavvio |

### Build opt-in e isolamento

Opzione prevista: `LFS_TENSOR_HIP=OFF` per default. Con OFF: nessuna ricerca SDK,
compilazione HIP, dipendenza HIP obbligatoria o cambiamento delle priorità Auto.
Con ON: SDK e architetture espliciti, inizialmente `gfx1150`, errore CMake chiaro
se manca un componente necessario. La build combinata CUDA+HIP+Vulkan deve essere
verificata come tale; non sostituire CUDA con HIP a livello globale.

Su Windows il resto dell'app deve poter conservare MSVC/nvcc. Compilare gli oggetti
HIP con il compilatore ROCm in un target/sub-build separato, con interfaccia C e
tipi POD se necessario; dimostrare compatibilità COFF/CRT, gestione eccezioni e
link prima del porting esteso. Non assumere supporto Windows di `enable_language(HIP)`
in qualunque versione CMake e non cambiare il compilatore C++ globale.
Su Linux verificare il percorso HIP CMake nativo con SDK fissato.

Le intestazioni HIP e CUDA non devono contaminare le stesse unità di traduzione
con alias globali. Il modulo HIP va caricato in modo controllato/lazy, così una
build completa continua ad avviarsi senza driver AMD. Backend richiesto ma non
disponibile: fallback all'avvio, con richiesta, scelta effettiva e ragione nel CLI.
Non migrare un training vivo tra backend dopo un errore di kernel.

### Porting e ottimizzazione

Prima una verticale completa: allocation/stream/event, fill/cast/gather, riduzioni,
scan/sort e random; poi GS forward/backward, loss RGB/SSIM, Adam, MCMC e salvataggio.
Gli strumenti HIPIFY possono aiutare sui sorgenti del progetto, ma ogni risultato
va revisionato. Audit esplicito di inline PTX, warp da 32, shuffle, ballot, atomiche,
cooperative groups, CUB e assunzioni di allineamento. Query e configurazione wave
per `gfx1150`, senza supporre equivalenza delle primitive CUDA.

Non utilizzare il generatore PTX CUDA nel backend HIP. Per le espressioni: prima
kernel nativi/fusioni controllate; eventuale HIPRTC solo dopo disponibilità e costo
misurati. Cache compilata per versione SDK, architettura e firma del programma.
Sort/scan attraverso rocPRIM/hipCUB se disponibili nella toolchain selezionata;
hipBLAS/rocBLAS per i carichi adatti, non come sostituto del raster backward.

Ottimizzare partendo dai costi reali: raster backward e contesa atomica, forward,
loss/SSIM, traffico dei gradienti e sincronizzazioni. Valutare riduzioni locali,
layout coalescenti, workspace riusabile, fusione Adam/proiezione dove semanticamente
equivalente e graph capture quando la forma è stabile. Evitare sincronizzazioni
globali per operazione e allocazioni per iterazione. Densificazione invalida il
graph quando necessario; non eseguire un graph con puntatori o capacità superati.

Nessuna riduzione nascosta di risoluzione, SH, gaussiani, precisione o funzioni
per ottenere un numero migliore. Le funzioni ancora non implementate devono
essere dichiarate e rifiutate nel preflight, non deviate silenziosamente alla CPU.

## Memoria del modello e viewer sulla stessa GPU

Obiettivo: una sola backing allocation per regione del modello, letta dal viewer
e scritta dal trainer con lease e completamenti espliciti. Gradienti, optimizer,
sort e framebuffer restano allocazioni distinte necessarie; non sono copie del modello.

Il primo percorso dimostrato è Vulkan-owner/HIP-importer. Non chiamare `hipMalloc`
per una seconda copia persistente degli stessi parametri. Il layout deve mantenere
offset/allineamenti, SH quantizzati, bounds e generazioni coerenti con il viewer.
Il trainer attende il termine dell'ultima lettura prima di modificare il modello;
il viewer attende il termine della scrittura prima di leggerlo. Zero-copy non
consente lettura/scrittura concorrente degli stessi parametri senza protezione.

In assenza di semafori esterni affidabili, è possibile mantenere lo stesso buffer
con attese host limitate al confine trainer/viewer, senza copiarlo. Questa modalità
è un riferimento di correttezza, non un risultato prestazionale definitivo.
Se si dimostrano semafori binari, un bridge verso le timeline Vulkan può utilizzare
submit e slot riusabili con ciclo di consumo rigoroso; va misurato e validato.

La crescita progressiva è un gate separato. Provare HIP VMM, export dei chunk e
import Vulkan prima di promettere indirizzi stabili e commit fisico progressivo.
In alternativa usare buffer Vulkan a capacità crescente e generazioni/lease:
una riallocazione può richiedere temporaneamente vecchia e nuova allocazione e
una copia durante la crescita, pur senza copia per frame a regime. Documentare
e limitare quel picco; non presentarlo come identico al VMM CUDA. Il probe da
1 MiB non dimostra questa parte. Errori di crescita devono lasciare valido il
modello precedente e rilasciare solo la riserva fallita.

Tra GPU fisiche diverse resta il percorso di copia esplicito, con budget e log;
non importare un handle soltanto perché entrambe le GPU appartengono ad AMD.
La cache JPEG preparata rimane quella esistente in RAM CPU. Monitorare memoria
allocata, riserva virtuale, capacità live, scratch, staging e memoria condivisa
separatamente, evitando doppi conteggi su UMA.

## Fasi e criteri di avanzamento

1. **Fattibilità locale (avanzata, non completa).** SDK isolato, kernel `gfx1150`,
   aliasing, identità fisica e un bridge GPU sperimentale passano. Restano la
   robustezza del bridge sotto errore e il protocollo di crescita della memoria.
2. **Build isolata opt-in.** Provare oggetti HIP + app MSVC/nvcc, runtime assente,
   macro OFF e dipendenze. Nessun cambio ai default CUDA/Mac.
3. **Tensor verticale.** Implementare le operazioni necessarie al primo training,
   con test di dtype/stride/offset, allocazione, stream, errori e teardown.
   Gli altri metodi restano errori espliciti, non fallback impliciti Vulkan.
4. **Training HIP completo RGB/MCMC.** Stesso modello e stesse formule;
   differenze finite, confronto CPU/CUDA/Vulkan e ripresa checkpoint.
5. **Viewer Vulkan senza copia persistente.** Validare generation, growth,
   cancellazione, resize, auxiliary context e milioni di gaussiani, quindi stress
   durante training. Provare esaurimento memoria e guasti di import/sync.
6. **Prestazioni e copertura.** Benchmark alternati, poi SH0-SH3, 3DGUT e strategie,
   depth/normals, PPISP/bilateral, mask e quantizzazione secondo capacità dichiarate.
   Windows per primo; Linux su hardware AMD nativo separatamente, non llvmpipe.

Ogni fase deve avere un risultato verificabile prima di attivare quella successiva.
L'opt-in non implica selezione automatica né sostituzione di backend funzionanti.

## Prestazioni e qualità: criterio proposto

Il controllo attuale Radeon Vulkan misura circa 247 ms/iterazione a un milione
di gaussiani (1920, SH0, 2.000 step). È un riferimento storico, non il controllo
di una futura build HIP: ripetere HIP/Vulkan sulla stessa revisione e hardware.
Il rapporto Metal/Vulkan sul Mac non predice il rapporto HIP/Vulkan sulla 890M.

Obiettivo minimo proposto per promuovere il percorso come vantaggio prestazionale:
mediana almeno 20% inferiore al controllo Vulkan nella finestra a modello stabile,
con ripetizioni alternate (almeno cinque coppie), p95 e costo del viewer riportati.
Sul vecchio riferimento 247 ms questo corrisponderebbe a circa 198 ms, ma non è
una previsione né un risultato ottenuto. Un obiettivo più ambizioso richiede dati
prima di assegnare un tempo. Se il vantaggio non c'è, non dichiarare raggiunto
l'obiettivo dell'utente e non promuovere il backend come default.

Stesso seed, dataset, resize, cap, SH, loss, strategia e evaluator; cold/warm
separati; misurare anche l'intera iterazione e il tempo al risultato, non solo
un microkernel. Non confrontare il numero storico da 1,4 secondi con una prova
ridotta per dichiarare il guadagno HIP. GPU timestamps e wall time vanno distinti.

Qualità: verifiche finite-difference RGB/depth/normals/rotazione, loss e gradienti
finiti, controlli ripetuti per stimare la variabilità, PSNR/SSIM/LPIPS a 2.000 step
e ispezione delle stesse viste. Conservare tutti i numeri; non promettere identità
bitwise tra riduzioni parallele. Provare SH0 e SH3 prima di concludere che il
backend completo sia più veloce, e misurare la fluidità con viewer attivo.

## Evidenze locali

Cartella separata: `D:\repos\LichtFeld-Studio\.local\rocm-tensor-study`.

- `probe_runtime.py`, `runtime-hip7.json`, `runtime-hip7-unrestricted.json`:
  prova runtime, errore nel sandbox e ripetizione riuscita nell'ambiente normale.
- `interop_probe.cpp`, `build-interop.cmd`: microprogramma autonomo MSVC/Vulkan.
- `interop-probe-memory-cycles.log`: 32 cicli con buffer condiviso, riusciti.
- `interop-probe.log`, `interop-probe-binary.log`, `interop-probe-kmt.log`,
  `interop-probe-d3d12.log`: tentativi di sincronizzazione, falliti e conservati.
- `sdk-install.log`, `kernel-smoke-summary.json`, `build-probes.log`:
  SDK e kernel realmente compilati, con entrambe le DLL runtime.
- `interop-capability-summary.json`, `interop-native-fence-gpu-streamsync.log`:
  capability e fallimento della sola attesa del semaforo su un vero roundtrip.
- `interop-native-fence-gpu-control-word.log`, `probes-ctest.xml`,
  `interop-physical-identity.xml`, `probes-final.xml`: bridge con contatore e suite riproducibile.

### Primo modulo riutilizzabile dal prodotto

`src/core/tensor/backend/hip/` contiene ora un modulo DLL/SO con ABI C privata
versionata: inventario HIP, contesti per dispositivo, code nonblocking, allocazioni,
upload/download, fill e affine float32. Le operazioni sullo stesso contesto devono
essere serializzate dal chiamante. Non passano oggetti C++, stream CUDA o header
HIP attraverso il confine del modulo. Gli errori non attraversano l'ABI come
eccezioni; ogni chiamante riceve codice nativo e messaggio in memoria propria.

La build principale offre `LFS_HIP_RUNTIME=OFF`. Con OFF non include la directory
del modulo e non cerca SDK HIP. Con ON richiede `LFS_HIP_ROOT` e un target esplicito
`LFS_HIP_ARCHITECTURE`; compila con il Clang dell'SDK senza sostituire MSVC/nvcc.
Questa opzione costruisce soltanto il modulo: **non è `LFS_TENSOR_HIP`**, non
registra un backend, non modifica la selezione automatica né i selettori GUI.
Il ramo Linux della compilazione non è ancora stato eseguito su Linux.

Il test MSVC `HipRuntime.OwnershipQueuesAndBounds` carica la DLL realmente compilata
e verifica sulla 890M: ABI, inventario, indici GPU inesistenti, contesti distinti,
rifiuto di buffer appartenenti a un altro contesto, overflow e limiti, offset con
guardie, ordinamento di kernel e copie, operazioni in-place, count zero e teardown
con buffer ancora vivi e lavoro accodato. 1.048.595 float controllati elemento per
elemento. La validazione preventiva degli indici evita di inviare al driver una
richiesta per un dispositivo inesistente: il primo test aveva rilevato un errore
`invalid device ordinal` e mancata terminazione del processo; dopo la correzione
il test termina correttamente. Non rappresenta una prova di fault recovery GPU.

Suite aggiornata: **4/4 pass**, 12,13 secondi complessivi, incluso il test precedente
di convivenza CUDA e il bridge Vulkan sperimentale. Risultati in
`runtime-contract-results.xml`. I buffer sono allocati su richiesta; non esiste
ancora un pool, una politica budget UMA o il blocco modello condiviso del trainer.
Le primitive non sono una misura delle prestazioni del training.

Le suite complete CUDA/Mac non sono state rieseguite: i percorsi applicativi
esistenti non sono stati modificati; la sola modifica alla build principale è il
nuovo modulo opt-in. Restano da implementare l'adapter Tensor, il collegamento al
trainer, la condivisione del modello con il viewer e i confronti su dataset.

### Prima ottimizzazione di una scena con rasterizzazione HIP

È disponibile una prova **autonoma**, controllata da
`LFS_HIP_PROBE_GSPLAT_TRAINING=OFF` nel progetto dei test. Genera nella directory
di build le versioni HIP dei kernel forward/backward gsplat già presenti nel
checkout: nessuna modifica ai sorgenti CUDA applicativi. Le trasformazioni sono
nel file riproducibile `src/training/hip/prepare_gsplat.py`. Mantengono le formule del
rasterizzatore, adattano le API e le diagnostiche host, usano maschere wave
a 64 bit e una riduzione tiled32 shuffle-XOR al posto di `cooperative_groups::reduce`.
Il processo rifiuta un dispositivo con warp size diversa da 32; wave64 non validata.
Sono compilate soltanto le istanze a tre canali.

La prova esegue forward, gradiente MSE, backward e Adam FP32 su posizione,
quaternion, log-scala, RGB e opacità. Usa un target prodotto da un renderer CPU
indipendente (geometria double, immagine float). Confronta il forward e controlla
15 derivate mediante differenze finite centrali, campionando tutte le famiglie.
Epsilon `1e-4`, tolleranza `1e-6 + 0,02 * abs(derivata numerica)`; il limite assoluto
conta per le derivate piccole, quindi non dichiarare errore relativo sotto il 2%
per ogni campione. Il massimo relativo osservato è circa 3,53% nella scena piccola
e 5,42% nella scena densa, su derivate di piccola magnitudine ammesse dal limite
assoluto. Nel primo tentativo con epsilon `1e-3` il controllo non passava: la
perturbazione più piccola ha risolto la discrepanza senza allargare le tolleranze.

| Scena sintetica | MSE iniziale | MSE a 400 step | Tempo wall/step in tre ripetizioni |
|---|---:|---:|---:|
| 6 gaussiani, 35×29 | 0,00182972856 | 2,12261–2,12264 × 10⁻⁷ | 0,061–0,063 ms |
| 263 gaussiani traslucidi, 35×29 | 0,00461940047 | 4,18126–4,22943 × 10⁻⁶ | 0,982–0,990 ms |

Su entrambe, il massimo errore assoluto forward rispetto alla CPU è
`2,38418579e-7`. La seconda scena attraversa i batch forward da 128 e backward da
256, con un ultimo batch parziale e tile immagine incompleti. Le variazioni fra
ripetizioni sono conservate; il backward usa accumulazioni atomiche parallele.
I tempi includono quattro ulteriori forward/readback di controllo e non comprendono
l'inizializzazione, il renderer CPU e le differenze finite.

Le prove iniziali hanno anche rilevato due errori del nuovo harness: mancava il
buffer obbligatorio del gradiente alpha (anche quando la loss non usa alpha) e
l'offset sentinella dopo l'ultimo tile. Entrambi corretti senza modificare i kernel
applicativi. Il secondo errore è emerso ampliando la scena; i primi numeri della
scena a sei gaussiani, precedenti alla sentinella, sono superati dai risultati
completi e ripetuti riportati nella tabella.

I render target/iniziale/finale sono salvati e sono stati controllati visivamente:
forma e colori si avvicinano al riferimento CPU. È una piccola scena artificiale,
non una prova della qualità sul dataset. La suite completa passa **6/6**, e i due
test training passano altre tre ripetizioni ciascuno. Evidenze:
`gsplat-training-suite.xml`, `gsplat-training-repeated.xml`, sottocartelle
`build-probes/gsplat-training-results` e `build-probes/gsplat-training-dense-results`.

Questa prova usa liste tile conservative fisse e include ogni gaussiano in ogni
tile. **Non include proiezione/binning/sort**, ingest, SH, SSIM, densificazione,
checkpoint, budget UMA, Tensor applicativi o viewer. Non usa l'ABI C del nuovo
modulo runtime: prepara il nucleo numerico che andrà esposto tramite quel confine.
Nessun confronto di velocità con i 247 ms del dataset Vulkan è valido a questo
stadio. Il rasterizzatore provato è gsplat 3D world-ray, non una prova del percorso
FastGS completo: anche questa distinzione va mantenuta nei futuri confronti.

### Estensione della verticale: binning dinamico sulla GPU

La terza prova, `HipTraining.GsplatDynamicBinningAndConvergence`, aggiunge i kernel
di proiezione UT e intersezione del checkout, con scan int64 e radix sort hipCUB.
Prima di ogni render ricalcola interamente proiezione, rettangoli dei tile, conteggi,
prefissi, chiavi tile/profondità, ordinamento e offset. Non riutilizza liste host
precostituite e non scarica i conteggi a ogni iterazione. Scratch e capacità sono
allocati prima del ciclo e riutilizzati sulla stessa coda HIP.

Le allocazioni hanno un limite esatto per questa piccola fixture: 263 gaussiani ×
6 tile + una sentinella = 1.579 slot. **Questo limite worst-case non va trasferito
al prodotto**: su immagini reali sarebbe troppo grande. La versione applicativa
richiede crescita limitata dal budget e capacità adattiva delle intersezioni.
La proiezione usa gli extent geometrici conservativi, senza restringerli in base
all'opacità. Non è ancora un confronto fra politiche ottimizzate di binning.

La trasformazione HIP gestisce anche la conversione dei limiti dei tile:
clamp in floating point prima della conversione unsigned, per impedire che un
limite negativo all'orlo dell'immagine diventi un indice molto grande. È una
regola esplicita del port HIP; non modifica né dichiara difettoso il percorso CUDA.
La lista viene verificata rispetto all'enumerazione CPU dei rettangoli proiettati.

Prima misura manuale della verticale completa: **1,013 ms/iterazione**, 400 step,
MSE `0,00461940047 -> 4,22201698e-6`, errore forward CPU `2,38418579e-7`.
Intersezioni reali: 1.128 iniziali, 1.132 dopo il training; capacità 1.579 slot.
Segue una prova con tutta la scena oltre il far plane: zero intersezioni, zero RGB,
zero gradienti dei parametri, senza errori di ordinamento o accesso alle sentinelle.
Questa misura comprende binning e checkpoint di controllo, ma rimane una scena
35×29 con 263 gaussiani. **Non permette di stimare il tempo a un milione di gaussiani**.

Le due prove senza binning passano anche con la DLL del solo SDK isolato:
runtime `71460850`, mentre la suite usa la DLL di sistema `70260201`; percorsi DLL
identificati nei log. MSE finali SDK: `2,12260241e-7` e `4,21063325e-6`.
Nessun driver o runtime globale è stato sostituito. Evidenze aggiuntive:
`gsplat-training-sdk-small.log`, `gsplat-training-sdk-dense.log`,
`gsplat-training-binning-first.log`, `gsplat-pipeline-build.log`.

Verifica finale della verticale dinamica: **7/7 test pass**, poi altre tre prove
dinamiche passate. Wall time `1,012–1,015 ms/step`, MSE finale
`4,17158–4,24376e-6`. Passa anche il confronto CPU di ogni coppia chiave/ID emessa,
dei conteggi e dei prefissi int64, oltre a ordine, range dei tile e sentinelle.
Con SDK `71460850`: `1,014 ms/step`, MSE `4,21860327e-6`, inclusa la verifica del
frame vuoto. Render della pipeline dinamica controllati visivamente; risultati
coerenti con il target e con la prova a liste fisse. Log finali:
`gsplat-pipeline-suite.xml`, `gsplat-pipeline-repeated.xml`,
`gsplat-training-sdk-binning.log`. In questa fase non sono cambiati sorgenti
applicativi: solo test opt-in, generatore e documentazione. Non sono stati
rieseguiti i training CUDA/Mac, e non è ancora disponibile un training HIP
richiamabile dal prodotto o una misura sul dataset dell'utente.

### Modulo sessione SH0 richiamabile dal codice host

`src/training/hip` contiene ora una DLL/SO con ABI C privata e versionata,
`training_api.h`. L'opzione applicativa `LFS_HIP_TRAINING_RUNTIME=OFF` richiede
`LFS_HIP_RUNTIME=ON` quando abilitata. Il build principale può quindi compilare
la sessione senza modificare i compilatori C++/CUDA. Il modulo non è ancora
collegato a `Tensor`, `TrainingOps`, ingest, selettori o viewer. Linux non validato.
Il modulo è stato compilato tramite lo stesso CMake dal progetto isolato; non è
stata eseguita una build completa dell'applicazione con il nuovo flag abilitato.
Il generatore è stato spostato sotto `src/training/hip`; la preparazione e gli
oggetti kernel sono condivisi fra test e modulo tramite `cmake/HipRasterKernels.cmake`.

La sessione supporta modello di dimensione fissa, camera pinhole globale e loss MSE.
SH0 è un vero coefficiente: attivazione `max(0, C0 * coefficiente + 0,5)` e gradiente
moltiplicato per C0, azzerato sotto il clamp. Parametri in un solo buffer con regioni
allineate a 256 byte; gradienti e momenti FP32 persistenti. Un kernel Adam aggiorna
tutte le famiglie. Non sono ancora implementati SH superiori, SSIM, regolarizzazioni,
densificazione, maschere frozen, checkpoint/ripristino dei momenti o condivisione
del modello con Vulkan. Il buffer parametri usa ancora `hipMalloc`; la precedente
prova di memoria condivisa non è integrata in questa sessione.

Il budget è esplicito e comprende tutti i buffer della sessione e la sovrapposizione
durante le sostituzioni, con conteggi live/peak. **Non è un contatore della VRAM o
dell'heap del driver**, né include runtime e altre applicazioni. Le intersezioni
vengono dimensionate sul conteggio GPU reale, con crescita limitata e una sentinella;
si ordinano soltanto le intersezioni vive, non tutta la capacità. Non viene allocato
il prodotto gaussiani × tile. Per ora il conteggio richiede un completamento host
per frame: eliminare questa attesa richiederà un protocollo di overflow verificato.

I fallimenti prima di Adam per budget, input o cancellazione conservano modello
e step. Si completa il lavoro GPU accodato prima di restituire un errore recuperabile.
La sostituzione di un frame fallita mantiene il precedente. Un errore GPU, un risultato
Adam non finito o un errore di risorse dopo il dispatch dell'optimizer rende la
sessione inutilizzabile, senza fallback silenzioso. Non viene promesso il rollback
di un kernel optimizer parzialmente eseguito. I test di budget non equivalgono a
prove di device loss o OOM reale del driver.

La cache degli attributi di shared memory HIP è protetta e indicizzata per dispositivo
e funzione kernel, evitando la condivisione di un singolo limite fra funzioni con
la stessa firma o fra sessioni su thread differenti. È un adattamento dei sorgenti
generati HIP: nessuna modifica all'implementazione CUDA o Mac dell'applicazione.

Test MSVC → DLL sulla 890M, runtime di sistema `70260201`:

- Forward SH0 contro CPU indipendente: massimo errore `2,38418579e-7`.
- Budget esaurito prima di Adam: parametri invariati, step zero, nessun buffer
  live rimasto dalla sostituzione fallita. Dopo aumento del budget, primo update
  coerente con una sessione nuova su tutte le famiglie (`max abs < 1e-5`).
- Sostituzione frame fallita dopo una prima allocazione riuscita: frame precedente
  ancora valido e conteggio live ripristinato.
- Validazione di GPU assente, quaternion nullo, camera non rigida/riflessa, target
  NaN e rate invalide; clamp SH0 e layout a un solo gaussiano verificati.
- Due sessioni su thread host diversi, cancellazione cross-thread e ripresa,
  camera traslata, frame vuoto e ripristino. Non è una prova con due GPU AMD fisiche.
- Memoria NVIDIA CUDA da 1 MiB viva nello stesso processo: ogni byte conservato e
  nuova scrittura riuscita dopo distruzione delle sessioni e unload della DLL HIP.
  Non è un confronto di training CUDA o una prova di carichi CUDA/HIP simultanei.

Training tramite C ABI, stessa scena sintetica 263 gaussiani/35×29: 400 step da
MSE `0,00461940046` a `4,33308–4,41525e-6` nelle tre ripetizioni, tempo wall
`1,176–1,212 ms/step`. Circa 178.568 byte live e 217.820 peak per i buffer della
piccola sessione; capacità intersezioni 1.654 con 1.132 vive. L'API completa ogni
step prima di tornare: questi tempi non sono il vecchio loop asincrono dei probe
né sono confrontabili con il dataset a un milione di gaussiani.

Prova separata di memoria con **un milione di elementi modello**, 128 MiB di budget
rifiutati senza pubblicare una sessione; con 512 MiB allocazione e un passo riusciti:
**280.089.496 byte live**, **280.093.928 peak**, 1.128 intersezioni e capacità 1.129.
I parametri dell'ultimo elemento invisibile rimangono invariati. Solo 263 gaussiani
sono visibili, a 35×29: questa verifica dimostra la gestione di memoria/capacità,
non il throughput con un milione di gaussiani visibili o un footprint entro 4 GiB
su immagini reali.

Anche il runtime SDK isolato `71460850` passa le due prove di sessione, inclusa
convivenza CUDA: MSE a 400 step `4,29838656e-6`, `1,184 ms/step`; stessa memoria
contabilizzata nella prova da un milione. Nessuna sostituzione di driver/runtime
globali. Suite completa **9/9 pass**, più tre ripetizioni della sessione. Evidenze:
`training-session-suite-final.xml`, `training-session-repeated.xml`,
`training-session-sdk.log`, `training-session-sdk-large.log` e
`build-probes/training-session-results`. Restano da collegare il modulo al trainer
applicativo e validare qualità/velocità sul dataset reale. I backend applicativi
esistenti non sono cambiati e i training CUDA/Mac non sono stati rieseguiti.

### Adapter host C++ del modulo di training

Aggiunti `src/training/hip/session.hpp` e `session.cpp`, compilati dal compilatore
host (MSVC nella verifica), senza includere header HIP/CUDA. Il target opzionale
`lfs_hip_training_host` carica il modulo da percorso assoluto, verifica ABI ed entry
point, conserva stato/codice nativo degli errori e gestisce durata della sessione
e del modulo. Le operazioni si serializzano; la cancellazione può arrivare da un
altro thread. Chiusura esplicita idempotente, rifiuto degli accessi dopo chiusura;
se la distruzione fallisce, diagnostica CLI e riferimento OS al modulo conservato
per evitare unload di codice con lavoro potenzialmente vivo. Non è una prova di
recupero da device loss reale. I chiamanti devono terminare i thread prima della
distruzione dell'oggetto C++.

Suite isolata **10/10 pass**, evidenza `host-adapter-suite.xml`. Il nuovo test usa
il codice host del prodotto: fallimento transazionale del caricamento frame per
budget, ripresa, cancellazione, accessi concorrenti step/info, readback del modello,
100 step e chiusura. Scena sintetica invariata: MSE `0,00461940046` →
`8,90355563e-6` col runtime di sistema, `8,90356028e-6` con quello SDK. Anche qui
l'allocazione NVIDIA CUDA viva rimane integra e riscrivibile dopo la chiusura.
Non sono tempi o metriche del dataset reale.

L'adapter non conserva copie del modello e non modifica i backend applicativi
preesistenti, ma **non registra ancora un Tensor backend HIP e non è ancora
collegato al trainer principale o al renderer**. Restano necessarie l'integrazione
con proprietà/storage Tensor e l'integrazione delle funzionalità di training oltre
SH0/MSE prima di esporlo nella UI. Linux e build applicativa completa con opzione
attiva restano da validare; le opzioni restano disabilitate per impostazione iniziale.

### Modello condiviso tra sessione HIP e allocazione Vulkan su Windows

Il modulo di prodotto ora espone il layout del modello e un costruttore per
importare un'allocazione Vulkan dedicata `OPAQUE_WIN32`, senza allocare una seconda
copia privata dei parametri. Gradienti, momenti e scratch rimangono HIP privati.
Il budget conta una volta l'intera allocazione importata, compreso il padding.
Il proprietario Vulkan deve conservarla fino alla distruzione della sessione.

La GPU viene verificata nel modulo: confronto LUID/node mask quando HIP li
pubblica; altrimenti risoluzione Windows del LUID Vulkan in indirizzo PCI e
confronto con HIP. Sul dispositivo locale serve questo secondo metodo. Identità
non risolvibili, ambigue o diverse vengono rifiutate, senza confronto per nome e
senza fallback silenzioso a un modello privato. Il chiamante deve fornire il LUID
effettivo del proprietario Vulkan: il descrittore non prova da solo l'origine di
un handle NT arbitrario. La risoluzione Windows è compilata come codice host
separato, senza header Windows nella compilazione device HIP.

Il passaggio usa completamento host e trasferimenti di proprietà della coda verso
`EXTERNAL` e ritorno. Non consente accessi simultanei al modello. Non usa le attese
GPU beta del precedente esperimento con semafori. Linux rifiuta esplicitamente
questo import finché non viene implementato il percorso corrispondente.

Nuovo test `HipTraining.VulkanSharedModelOneMillion`: modello da un milione di
elementi, 263 visibili a 35×29, tre sessioni (condivisa, privata e secondo controllo
privato), 100 aggiornamenti per sessione. Sei letture tramite copia GPU Vulkan
verificano **byte per byte tutte e cinque le famiglie** contro il readback HIP.
Verificati guard finale, rifiuto del descrittore della seconda GPU fisica, identità
nulla, dimensione insufficiente, budget insufficiente e riuso dell'allocazione
Vulkan dopo distruzione della sessione HIP.

Nel confronto iniziale una soglia assoluta RGB di `1e-5` falliva. Il secondo
controllo ha mostrato scarti analoghi anche tra due allocazioni private: dopo
100 step, max parametri circa `2,49e-4`, max RGB circa `0,002285`, MSE tra immagini
circa `3,11e-9`. Le accumulazioni atomiche dei gradienti non sono deterministiche;
non si dichiara equivalenza bit per bit fra training indipendenti. Il primo
aggiornamento condiviso/privato è invece coincidente o differisce sotto `4e-9`
nelle prove registrate.

Il test mantiene soglie numeriche esplicite: primo aggiornamento `<1e-5`, dopo
100 step max parametro `<1e-3`, max RGB `<1/255`, MSE tra immagini `<1e-8`, MSE verso
target entro 1% (più `1e-9` assoluto) dell'intervallo dei due controlli privati e
miglioramento di almeno 10× rispetto all'inizio. La verifica dell'alias della
stessa sessione rimane esatta. Nei controlli registrati l'MSE scende da
`0,000554199743` a circa `9,4616–9,4623e-6`; nessun peggioramento significativo
rispetto ai controlli privati in questa scena. Runtime SDK e di sistema passano.

Sessione condivisa: **291.211.168 byte live**, **291.250.636 peak**, inclusi
**67.109.120 byte** di allocazione Vulkan importata una sola volta. Questi sono
buffer contabilizzati dalla sessione, non consumo totale del driver/UMA. La prova
non misura il throughput con un milione di gaussiani visibili o il dataset reale.

Suite **11/11 pass**; evidenze `shared-model-suite.xml`,
`shared-model-suite-final.xml` e `shared-model-system.log`. I test ora compilano e
collegano anche il target CMake di prodotto `lfs_hip_training_host`, non una copia
dell'adapter. Configurazione e build dei probe con opzione OFF passano senza SDK o
compilatore. Non è ancora una build completa dell'applicazione. Fallimenti reali
di driver/device loss e cleanup non sono stati iniettati.

Rimangono da implementare storage/operazioni Tensor, famiglie TrainingOps mancanti,
collegamento al trainer/viewer e confronti sul dataset. Non viene esposto un
backend incompleto nel selettore; prima della registrazione devono essere chiusi
anche i dispatch che attualmente assumono solo CUDA/Vulkan/Metal. Le modifiche
sono confinate al modulo HIP opzionale e ai suoi test/documentazione; nessun
percorso applicativo CUDA, Metal o Vulkan preesistente è stato modificato.

### Riferimento da riallineare: PR #3683 in attesa di merge

Letta la [PR #3683](https://github.com/MrNeRF/LichtFeld-Studio/pull/3683), ancora
aperta al controllo, head `b52f8425116ade46cb091d37adc0500e973aefe7`, base `dev`.
Non è stata incorporata anticipatamente. Dopo il merge va aggiornato il branch
sul nuovo `dev`, preservando solo i commit HIP e rieseguendo i controlli interessati.

Il nuovo riferimento prestazionale è **114,591 ms/step**, con **9,936 ms** di attesa
loader media e **253,641 s** di processo, come riportato nella PR. Protocollo:
checkpoint immutabile allo step 200, altri 2.000 aggiornamenti fino a 2.200,
MCMC/Fast/3DGS, SH0, limite un milione, cache CPU, dimensione massima 1920
(immagini effettive 1440×1920). Non sostituire questo riferimento con i vecchi
229,977 ms, che rimangono soltanto evidenza storica. Nessun confronto HIP su quel
carico è ancora disponibile. Il modulo corrente usa il percorso gsplat: il futuro
confronto deve dichiarare e allineare anche il rasterizzatore, non solo risoluzione
e numero di gaussiani. Tempi dei piccoli test non sono utilizzabili come confronto.

Scelte da riutilizzare dopo il riallineamento:

- Identità persistenti normalizzate e selezione per contesto, senza affidarsi a
  nome/indice e senza forzare il dispositivo zero. Il descrittore HIP sceglie già
  esplicitamente l'ordinale e verifica il dispositivo dell'import condiviso.
- Compatibilità e trasferimenti decisi dal contesto consumatore; usare il percorso
  host per dispositivi diversi. Non duplicare la policy in un controllo globale HIP.
- Lease di esportazione immutabili, offset e durata dello storage conservati.
  L'allocazione mutabile di training del probe non può essere esposta come lease
  immutabile del viewer senza adottare i contratti del core.
- Backend scoped dei loader, memoria UMA distinta dalla VRAM dedicata, errori
  espliciti e fallback CLI. Prima della registrazione HIP chiudere i dispatch che
  assumono l'attuale insieme di tre backend, mantenendone i valori enum.
- Riprendere i sort/limiti di subgroup della PR solo dove applicabili. Il radix
  sort HIP corrente usa hipCUB; non sostituirlo automaticamente con shader Vulkan.

### Loss fotometrica L1/SSIM nel modulo HIP

Implementati `set_loss` e valutazione scalare GPU della loss. MSE resta il valore
iniziale e mantiene lo stesso kernel. Il percorso non mascherato L1/SSIM deriva
direttamente dai kernel del checkout (`ssim.cu`), senza cambiarne formule,
coefficienti, ordine delle convoluzioni o parziali FP16. Nessuna modifica ai
sorgenti CUDA/Metal. La configurazione valida tipo, peso finito in [0,1] e padding
prima di modificare lo stato.

Padding valido identico al percorso fused esistente: esclusione di cinque centri
per lato nelle dimensioni maggiori di dieci; dimensioni piccole e zero padding
esterno preservati. Peso SSIM zero usa un kernel L1 diretto senza workspace SSIM.
Con peso positivo, allocazione lazy di **22 byte per pixel RGB**: 18 per i tre
parziali FP16 e quattro per la mappa media. La memoria rientra nel budget, si
riutilizza e viene rilasciata tornando a L1 puro/MSE. Verificato anche il fallimento
della seconda allocazione dopo la riuscita della prima. La valutazione scalare
riusa il buffer del gradiente immagine; non scarica immagini né avanza Adam.

Controlli aggiunti:

- 60 casi di valore/gradiente contro convoluzione 2D e chain rule CPU double
  indipendenti, comprese differenze finite, segni L1 misti, uguaglianze esatte,
  dimensioni 1×1/pari-dispari, bordi e tile parziali. Massimo errore relativo L2 dei
  gradienti osservato `0,00178579873` (0,179%, con parziali FP16); massimo scarto
  loss `8,15893417e-7`. Soglie esplicite 1% e `5e-5`.
- 48 riduzioni scalari della sessione contro CPU, sette descrittori invalidi,
  ridimensionamento, budget completo/parziale, ripresa e parametri immutati durante
  le diagnostiche. Un passo L1 con target esattamente uguale al render lascia
  invariati tutti i parametri.
- Tre training da 300 step (solo L1, peso SSIM 0,2, solo SSIM) con allocazione CUDA
  viva nello stesso processo. Convergenza e riuso CUDA passano con runtime di
  sistema `70260201` e SDK isolato `71460850`.

Risultati del controllo SDK, scena sintetica 263 gaussiani a 35×29:

| Peso SSIM | Loss iniziale | Loss a 300 step | MSE immagine finale |
|---|---:|---:|---:|
| 0 | 0,079052344 | 0,00213381136 | 9,24638025e-6 |
| 0,2 | 0,104019314 | 0,00172932236 | 6,55249769e-6 |
| 1 | 0,203887224 | 0,00092611037 | 4,72159243e-6 |

La nuova prova di condivisione applica L1/0,2 SSIM a un milione di elementi (263
visibili), confrontando 100 aggiornamenti condivisi e due controlli privati. Con
i rate fissi calibrati per il vecchio test MSE, L1 oscillava e il confronto falliva
le soglie MSE: anche i controlli privati divergevano. Per questa nuova prova viene
fornita la stessa schedule `max(0,02; 0,95^step)` nei tre percorsi; il modulo non
cambia i rate del chiamante. Il vecchio test MSE e le sue soglie restano invariati.

Il confronto L1/SSIM verifica primo passo `<1e-5`, qualità verso il target e loss
fotometrica entro 1% (con i floor assoluti documentati) dell'intervallo dei
controlli privati, RGB RMSE `<1/255`, max RGB `<4/255` e miglioramento MSE almeno
10×. Non richiede identità dei parametri dopo traiettorie indipendenti con la
loss non liscia. Le sei letture del modello Vulkan/HIP restano **byte identiche**.
Nel controllo completo: primo passo max `5,96e-8`; MSE condivisa `2,68105768e-5`
contro `2,68086214e-5` e `2,68246035e-5`; loss fotometrica `0,00430834433` contro
`0,0043086065` e `0,00431144238`. Non è un test di throughput o del dataset reale.

Suite completa **14/14 pass** (`photometric-suite.xml`); successivo oracolo esteso
a 60 casi passato sui due runtime (`photometric-oracle-final.xml`); controlli SDK aggiuntivi
in `photometric-kernel-sdk.log` e `photometric-session-sdk.log`. Rimangono da
implementare Tensor HIP, SH superiori, strategie/refinement, percorsi fotometrici
mascherati/appearance, checkpoint e integrazione nel trainer/viewer. Il backend
non è ancora esposto nella UI e rimane opt-in.

### SH1–SH3: kernel e collegamento alla sessione

Adattati per HIP `SphericalHarmonics.h` e `SphericalHarmonicsCUDA.cu` del checkout,
senza modificare i sorgenti CUDA. Il target condiviso dei kernel ora compila anche
lo SH forward/backward e il calcolo della direzione di vista. Il test indipendente
usa polinomi SH reali in double e differenze finite, non la stessa implementazione
del device: **100 combinazioni** di grado attivo/layout 0–3, dimensioni
1/31/32/33/263, maschere e code di blocco, direzioni/maschere/storage opzionali nulli.
Massimi scarti osservati: colori `5,19201975e-8`, coefficienti `6,16498489e-8`,
derivata direzione `1,06334842e-8`.

La sessione espone `set_sh` e `set_sh_degree`, con upload/readback canonico
`[N, rest, RGB]`. Valori e momenti FP32 sono conservati nel layout swizzled
float4/gruppi da 32 già usato dai kernel. L'Adam SH-rest percorre lo storage fisico
per coalescere letture/scritture di valori e momenti e legge i gradienti canonici
del raster. Esiste una sola copia persistente dei parametri SH-rest; l'upload
canonico temporaneo rientra nel budget e viene liberato. Il readback riusa scratch
dei gradienti, senza una seconda copia persistente del modello.

Il percorso predefinito SH0 mantiene il suo kernel e non alloca stato SH-rest.
La prima allocazione può essere differita fino all'attivazione di SH, anche dopo
aggiornamenti SH0. Uno storage SH esistente può essere sostituito/azzerato soltanto
prima del primo aggiornamento; il cambio di grado attivo e rate non rialloca.
I coefficienti inattivi e i loro momenti rimangono invariati, anche disattivandoli
dopo averli già addestrati. La correzione del bias usa lo step globale della
sessione: non viene azzerato il contatore all'attivazione.

Il backward applica una sola volta la derivata del clamp colore e la conversione
SH0. Le direzioni sono staccate dal gradiente delle posizioni, come nel chiamante
gsplat CUDA esistente; il gradiente opzionale della direzione è verificato a parte.
Gaussiane escluse dalla proiezione o con direzione nulla vengono mascherate prima
della normalizzazione. Non sono stati modificati i percorsi applicativi CUDA,
Vulkan o Metal.

Controlli della sessione:

- Input/gradi/rate non validi e coefficienti non finiti; fallimento parziale del
  budget, pulizia, riconfigurazione e rilascio prima del training.
- Upload swizzled/readback canonico esatto; render SH0 identico con SH-rest
  configurato ma inattivo; confronto CPU per SH1–3 e camera traslata. Scarto
  massimo del render `2,68220901e-7`.
- Primo aggiornamento SH0 confrontato con quello del nuovo percorso a SH-rest
  zero: max delta circa `1,2e-7` (soglia `1e-5`). Attivazione differita senza
  azzerare lo step, gradi inattivi con momenti esistenti, rate zero e riattivazione.
- Derivata del clamp nero e gaussiana esclusa con centro coincidente con la camera.
- Quattro training SH-rest-only da 200 step con geometria/SH0 congelati e target
  sintetico CPU; non sono confronti multivista o del dataset reale.

| Controllo | Loss iniziale | Loss finale SDK |
|---|---:|---:|
| SH1, MSE | 8,3954113e-8 | 1,1844572e-9 |
| SH2, MSE | 3,27987664e-7 | 4,448657e-9 |
| SH3, MSE | 7,53021141e-7 | 1,03096855e-8 |
| SH3, L1/0,2 SSIM | 0,000825505238 | 0,0000653445022 |

La prova L1/SSIM riceve un rate SH-rest `0,002 * max(0,05; 0,98^step)`, identico
per ogni esecuzione del protocollo; non è un cambio implicito dei rate nel modulo.
Gli altri tre controlli usano rate SH-rest fisso `0,005`.
La tabella conserva la prima lettura SDK. La ripetizione finale porta la loss
SH3/L1-SSIM a `0,0000652426534`; non si dichiara identità tra training indipendenti.

Prova separata con un milione di elementi, solo 263 visibili a 35×29 e budget
software di 512 MiB: configurazione SH3 rifiutata transazionalmente senza perdita
di risorse o sessione avvelenata; successiva configurazione SH1 e un passo riusciti
nella stessa sessione. **485.089.496 byte live**, **521.004.436 peak**, includendo
lo staging iniziale. Non è un limite hardware a SH3 né un dato di VRAM dedicata:
sono buffer contabilizzati e un test deliberato di ammissione al budget.

Suite completa **17/17 pass** (`sh-suite.xml`), seguita dai tre controlli SH finali
(`sh-final-contracts.xml`). Kernel, sessione e prova grande passano anche col
runtime SDK isolato (`sh-kernel-sdk.log`, `sh-session-sdk.log`, `sh-large-sdk.log`).
Le prove host mantengono una allocazione NVIDIA CUDA viva, verificata integra e
riscrivibile dopo la chiusura di tutte le sessioni.

Limiti ancora aperti: SH-rest è privato HIP, anche quando le cinque famiglie base
usano l'allocazione condivisa Vulkan. Non si dichiara quindi un modello SH1–3
completamente condiviso col viewer. Mancano import/export SH-rest del core,
Q16/FP16, momenti compressi, crescita dello storage e ripristino dei checkpoint.
Restano inoltre Tensor HIP, strategie/refinement e integrazione nel trainer e
viewer. Al controllo la #3683 è ancora aperta allo stesso head `b52f84251`:
nessun riallineamento anticipato. La baseline di prestazioni resta il protocollo
da **114,591 ms/step**, non i tempi di questi piccoli controlli.

### Separazione dello storage dei parametri e di Adam (18 ottobre 2026)

Lo studio dei contratti `SplatExportableStorage`, `TensorUpload` e
`TensorWorkQueue` conferma che lo storage applicativo non è intercambiabile con
il blocco del prototipo: ordine delle famiglie, capacità riservata, generazioni
e lease del viewer devono essere rispettati esplicitamente. Non vengono
registrati backend Tensor parziali né reinterpretate code HIP come CUDA.

Nel modulo sperimentale il layout dei parametri esportati resta allineato a
256 byte; gradienti e due momenti FP32 hanno ora offset indipendenti, con
allineamento a 16 byte. Backward e Adam usano i rispettivi offset. Questo evita
che il futuro adattatore debba replicare negli stati dell'ottimizzatore i buchi
e le riserve dello storage esterno. Non abilita ancora l'importazione diretta
del layout applicativo, crescita/refinement o lease immutabili del viewer.
Non cambia ABI, formule, layout pubblico né codice CUDA/Metal/Vulkan esistente.
Nessuna nuova misura sul dataset o dichiarazione di vantaggio su Vulkan.

Verifica: suite completa 17/17 in 67,21 s (`optimizer-layout-suite.xml`
nella cartella locale degli artefatti), eseguita in sequenza. Sono inclusi
budget da un milione di gaussiani, interoperabilità Vulkan MSE/L1-SSIM,
concorrenza, cancellazione e CUDA attivo. Nei casi N=263 gli offset privati
sono effettivamente diversi da quelli esportati. Ripetuto SH1–SH3 sul runtime
SDK locale: render CPU massimo 2,68220901e-7, consistenza primo aggiornamento
SH0 1,16866431e-7, convergenza dei quattro casi di 200 step confermata e
allocazione CUDA viva invariata e ancora scrivibile. Sono test sintetici.
La #3683 risulta ancora aperta sul commit b52f8425116ade46cb091d37adc0500e973aefe7.

### Offset applicativi e aggancio senza ricaricare il modello (18 ottobre 2026)

Aggiunti due ingressi C ABI e i corrispondenti costruttori host: creazione
con offset esterni e aggancio a parametri già residenti. Il primo valida
allineamento, conteggi, overflow, intervalli e sovrapposizioni; il secondo
valida i contenuti sulla GPU e legge solo uno stato scalare. Non ricarica
i parametri dalla CPU, non li modifica durante la validazione e inizializza
un ottimizzatore nuovo: non è ripristino di un checkpoint Adam.

L'adattatore `splat_storage_layout.hpp` legge gli offset dal vero
`SplatExportableStorage::Control`, con mapping means/rotation/scaling/SH0/opacity
verso le famiglie HIP. Ammette per ora SH0 fisso e storage sano; rifiuta chunk
VMM esportati. Il chiamante deve ancora fornire il corretto handle Vulkan,
mantenere proprietario e generazione stabili e rispettare l'handoff esterno.
L'adattatore non è ancora invocato dal trainer principale; non introduce
operazioni Tensor HIP né crescita/refinement o SH-rest condiviso.

Verifica finale: **19/19 test in 71,76 s**, report locale
`core-resident-final-suite.xml`. Il nuovo test usa un Control sullo storage
Vulkan reale con ordine applicativo, padding iniziale e 50.000 righe riservate
per famiglia oltre al milione di righe attive. Dopo l'inizializzazione distrugge
la sessione e ne aggancia una nuova senza input host del modello. Nove modelli
residenti invalidi e sette layout malformati sono rifiutati; il fallimento
dell'aggancio preserva bit per bit i valori invalidi fino al ripristino del test.
Il contratto core ha inoltre dieci casi di rifiuto.

Il confronto di 100 aggiornamenti MSE con due sessioni private conserva
le soglie esistenti: primo aggiornamento delta massimo **1,16415322e-10**;
MSE target **0,000554199743 → 9,46224846e-6**, contro **9,46160184e-6** e
**9,46161106e-6** dei controlli. Massimo scarto RGB **0,00228527188**,
MSE tra immagini **3,11003212e-9**. Tutte e cinque le famiglie sono verificate
via copia GPU Vulkan bit per bit rispetto a HIP; padding, riserve e coda del
buffer mantengono la sentinella. Allocazione proprietaria viva dopo teardown.
Memoria della sessione **291.211.168 byte**, picco **291.250.636**, inclusa
l'allocazione importata **67.109.120 byte** conteggiata una sola volta. Solo
263 gaussiani visibili a 35×29: non è una misura di throughput sul dataset.

Configurazione con probe OFF ancora verificata senza SDK o compilatore HIP.
Nessuna modifica ai sorgenti dei backend CUDA, Metal e Vulkan esistenti; non
sono stati ripetuti qui training completi macOS/CUDA o validazione HIP Linux.


### Tensor applicativi nativi e training sul loro storage (18 ottobre 2026)

Implementato il collegamento reale a `lfs_core`, oltre al modulo isolato:
`LFS_TENSOR_HIP` resta OFF di default e richiede il runtime HIP esplicito.
L'adattatore host carica la DLL/SO con ABI verificata e implementa 68 operazioni
pointwise FP32, fusioni fino a 16 stadi, tutti i 17 operatori binari in broadcast
(inclusi predicati byte), copie, fill, arange, materializzazione strided e
riduzioni scalari. Le formule provengono dai functor del progetto; la sola
correzione privata ReLU conserva il segno di -0 come il riferimento CPU.
Le operazioni mancanti danno un errore esplicito. HIP non è ancora candidato
automatico/default né selezionabile in preferenze; TrainingOps non dichiara
famiglie che non implementa. Non viene delegato lavoro a CPU/Vulkan.

L'allocatore conserva temporanei di dimensione esatta entro 64 MiB, senza
sincronizzare ad ogni rilascio. Riuso ordinato sulla singola coda, contatori
attivo/riservato separati, allocazioni direct escluse dalla cache e trim limitato
ai buffer non vivi. In caso di OOM nativo svuota la cache opzionale e ritenta una
volta; questa condizione driver non è stata iniettata. Il test verifica riuso
con un consumatore ancora pendente, budget da 1 KiB, allocazione direct e trim
che preserva i valori di un Tensor vivo. Nessuna preallocazione dell'intera UMA.

`TensorModel` aggancia il vero Tensor HIP e il suo Control al training senza
ricaricare i parametri o duplicare il modello. Valida base/range/device nativi,
conserva una snapshot shallow dell'allocazione e controlla generazione, offset,
capacità e identità prima di ciascuna operazione. Attende esplicitamente i
produttori della coda Tensor prima della coda separata della sessione; le
operazioni della sessione sono completate al ritorno. Errori GPU o poisoning
Adam invalidano anche il Control applicativo. Il proprietario resta vivo fino
al teardown riuscito; un teardown fallito conserva proprietario e modulo.
Rimane obbligatoria la serializzazione delle mutazioni del modello.

#### Risultati e limiti della qualità

Compilati ed eseguiti i veri target core, senza CUDA e con CUDA 12.8 + HIP
insieme. La matrice di tre test per tre configurazioni è **9/9**: HIP-only,
CUDA+HIP con entrambe le GPU esposte, CUDA+HIP senza GPU CUDA esposte ma con
HIP visibile. Quest'ultima usa soltanto nell'ambiente figlio
`CUDA_VISIBLE_DEVICES=-1`, `HIP_VISIBLE_DEVICES=0`. Il controllo runtime CUDA
restituisce no-device (100), HIP enumera un dispositivo. Non equivale a una
macchina senza driver/runtime CUDA installato. La variabile CUDA da sola
nasconde anche la Radeon nel runtime HIP, come indicato nella
[documentazione AMD](https://rocm.docs.amd.com/projects/HIP/en/docs-7.14.0/reference/env_variables.html).
I controlli Tensor CUDA prima/dopo lo scope HIP passano senza modificare il
default di processo. Il test di ammissione rifiuta il training HIP incompleto.

Il confronto TensorModel usa **400 step**, 6 gaussiani attivi su 263 righe
logiche, SH0, 35×29, capacità 320 e sentinelle nelle riserve. Da step 200 il
solo test riduce progressivamente i learning rate. MSE target iniziale
**1,8062774e-5**, finale **2,4260234e-7–2,42602454e-7** nella matrice finale;
controlli privati **2,42602255e-7–2,42602368e-7**. Primo aggiornamento delta
massimo **0**. Massimo scarto RGB **1,86264515e-8–2,98023224e-8**, MSE tra
immagini **7,53950452e-18–1,13076523e-17**. Tutte e cinque le famiglie aliasano
bit per bit; padding/riserve preservati e proprietario rilasciato alla chiusura.
Restano le soglie originali max < 1/255 e MSE < 1e-8.

**Limite da non confondere con questo risultato:** il primo tentativo di fixture
con 263 gaussiani tutti attivi per 400 step non rispettava le stesse soglie di
immagine: max 0,00408338 e MSE 1,0894e-7. Anche i due controlli privati divergevano
(max 0,0043664, MSE 8,1101e-8), pur con primo aggiornamento quasi identico
(2,3283e-10). Il risultato non prova equivalenza qualitativa nel caso denso.
Una soglia adattiva provvisoria non è stata mantenuta: il test finale è mirato
allo storage/ownership su una fixture più stabile, con soglie strette. Nessuna
modifica alle formule del prodotto per far passare il confronto. La qualità
multiview/dataset e la stabilità densa restano da verificare prima dell'abilitazione.

Suite isolata completa dopo broadcasting/cache: **20/20 in 93,80 s**,
`hip-native-tensor-final-suite.xml`; matrice reale core in
`core-integration-matrix.json`, entrambi nella cartella locale degli artefatti.
Probe OFF configura senza SDK/compilatore. Il test dei produttori pendenti ha
inoltre rilevato e fatto correggere il dispatch HIP degli scalari in-place:
i metodi Tensor usano opcode Tensor con operando scalare separato, come nel
contratto CUDA. La correzione è soltanto nell'adattatore HIP.

#### Passi ancora necessari per il backend completo

1. Completare le operazioni Tensor richieste dal trainer: riduzioni per asse,
   conversioni, indicizzazione/gather/scatter, RNG, ordinamento e primitive di
   inizializzazione/refinement. Test CPU e test dei contratti per ogni famiglia.
2. Collegare TrainingOps e il ciclo applicativo: ingest CPU/resizing/cache,
   immagini residenti, schedule, strategie e crescita/pruning, checkpoint con
   stati Adam. Il prototipo fixed-count non sostituisce questi percorsi.
3. Collegare i lease immutabili del viewer e lo storage HIP/Vulkan applicativo,
   con identità fisica, higher-SH, crescita/generazioni e handoff verificato.
   Il test di import dedicato Win32 esiste; il viewer dell'app non è collegato.
4. Dopo il merge di #3683, riallineare la base e integrare rilevamento/selettori,
   fallback e indicatore riavvio; poi confrontare dataset alle stesse condizioni
   della baseline Vulkan aggiornata (114,591 ms/step + 9,936 ms loader wait).
   Servono verifiche Windows/Linux e regressioni CUDA/macOS pertinenti.

Non è ancora misurato un vantaggio HIP sul dataset. Non sono stati ripetuti
training completi CUDA/macOS né verificato HIP Linux in questa fase; i sorgenti
dei kernel CUDA/Metal/Vulkan esistenti restano invariati.


### Riduzioni Tensor per asse e per colonne

Dopo il checkpoint `20211212e`, implementate riduzioni FP32 sum/mean/min/max
per assi generici (anche non adiacenti e con stride), colonne e segmenti
outer/reduce/inner. I risultati restano sulla GPU. Per attributi fino a 32
componenti viene usato un lane per output; assi più lunghi usano workgroup.
Se gli output sono pochi e l'asse supera 4096 elementi, il lavoro è suddiviso
in più gruppi e ricombinato con un secondo kernel, entro 4096 float parziali.
Workspace condiviso con le riduzioni scalari: 16.388 byte, riutilizzato sulla
stessa coda. Non cambia i kernel del modulo di training né altri backend.

Verificati oracoli CPU double, extents 3/32/33/263/4096/4097/100003,
outer/inner diversi, colonne, assi non adiacenti/trasposti, offset, sentinelle,
NaN/Inf/zeri con segno, sum/mean vuote e rifiuti di extrema vuoti, assi
invalidi/duplicati, overlap, overflow e operatori non implementati.
Eseguiti anche `Tensor::sum(0)` e `Tensor::mean(1, true)` nel vero core.
La matrice aggiornata resta **9/9**, report `axis-core-integration-matrix.json`.
I tre confronti TensorModel a 400 step restano entro le soglie precedenti:
primo aggiornamento delta 0, massimo scarto RGB 1,86264515e-8–2,23517418e-8.
Sono verifiche di correttezza; non misure di throughput sul dataset.

Il primo blocco residuo del piano è quindi avanzato anche sulle riduzioni per
asse, ma richiede ancora conversioni, indicizzazione/gather/scatter, RNG e altre
primitive. Nessuna abilitazione del trainer completo o dei selettori è implicita.

Suite completa successiva alle riduzioni: **20/20 in 88,03 s**, report locale
`hip-axis-final-suite.xml`. Formattazione e `git diff --check` superati.


### Conversioni e letture indicizzate native

Dopo il ripristino della macchina il checkout risultava pulito sui due commit
precedenti. Implementate le 49 coppie di conversione fra i sette dtype del core,
con confronto ai contratti CPU: saturazione/troncamento degli interi, NaN a zero,
byte modulo 256, bool nonzero e half round-to-nearest-even con subnormali.
Le copie dello stesso dtype conservano i bit; overlap non identico rifiutato.
Non implica supporto alle operazioni aritmetiche integer/half ancora mancanti.

Aggiunti gather e index_select nativi, stride sorgente/indici, Int32 e Int64,
tutti i dtype dei valori e modalità Assert/Clamp/Wrap. Assert esegue validazione
GPU prima di scrivere e legge un solo stato da quattro byte; indici invalidi
preservano interamente l'output. Clamp/wrap non richiedono quel readback.
Conservare Int64 e stride nel solo percorso HIP dell'app evita narrowing e
interpretazioni errate degli indici trasposti. Gli altri backend mantengono il
loro percorso esistente; aggiunti controlli CUDA prima/dopo lo scope HIP.
La conversione controllata Int64→Int32 resta disponibile separatamente e rifiuta
anche extent oltre il range di indirizzamento Int32.

Verificati 49 coppie dtype e 252 configurazioni gather/select (7 dtype × 2 tipi
di indice × 3 assi × 2 operazioni × 3 modalità), inclusi Int64 min/max,
sentinelle, layout strided, duplicati, invalidazione transazionale e recupero.
Nel vero core passano roundtrip half/byte contro CPU, selezione righe Int64,
gather multidimensionale/indici trasposti e clamp/wrap di Int64 estremi.
La matrice core aggiornata è **9/9**, report `convert-index-core-matrix.json`.
Primo aggiornamento del confronto TensorModel a 400 step ancora delta 0;
scarto RGB massimo 1,67638063e-8–2,79396772e-8, soglie strette conservate.
Sono scene sintetiche di storage/ownership, non dataset o benchmark.

Rimangono scatter/scritture indicizzate, RNG, ulteriori operazioni e tutte le
integrazioni del trainer/viewer già elencate. Nessun push o PR di questo branch.

Suite completa conversioni/indici: **20/20 in 92,85 s**, report locale
`hip-convert-index-final-suite.xml`.


### Conteggio delle maschere e valori nonzero

Dopo il checkpoint `c675f56a3`, aggiunto count_nonzero nativo per Float32,
Bool/UInt8. Riduzioni e contatore sono interi a 64 bit, con un contributo atomico
per workgroup e un solo scalare da otto byte restituito alla CPU. Il workspace
fisso del contesto viene riutilizzato, senza buffer proporzionali all'immagine
o al numero di gaussiani. NaN/Inf contano come nonzero; +0 e -0 no.
Non implementa ancora compattazione delle maschere o pruning del trainer.

Verifica CPU su 0, 1, 263, 1.048.595 e 16.777.219 elementi. Nell'ultimo caso
tutti gli elementi sono nonzero: il risultato supera 2^24 e resta esatto.
Inclusi byte noncanonici nonzero, valori speciali e rifiuti dtype/overflow.
Nel vero Tensor passano float, maschere da predicati e viste non contigue.
Test mirati core/runtime **2/2 in 1,35 s**, report `hip-count-final-contracts.xml`.
La matrice core resta **9/9**, report `count-core-matrix.json`, incluso CUDA
nascosto solo ai processi e controlli CUDA gather/select prima/dopo HIP.
Nei tre confronti sintetici di 400 step: delta primo aggiornamento 0,
scarto RGB massimo 1,86264515e-8–2,04890966e-8; soglie precedenti conservate.
Rimangono da implementare 77 metodi dell'adattatore, con peso molto diverso:
il conteggio non è una percentuale di avanzamento del backend completo.

Suite finale completa dopo il conteggio: **20/20 in 98,35 s**, report locale
`hip-count-full-final-suite.xml`. Verificata inoltre la compilazione sintattica
del file condiviso `tensor_masking_ops.cpp` con HIP disabilitato, usando gli
stessi include/define della build core senza CUDA e output separati. Questo
non sostituisce una build completa macOS/Linux, ancora da eseguire.


### Scritture indicizzate e riscrittura delle viste

Implementati index_copy e index_add nativi. La copia privata ammette tutti i
sette dtype; l'accumulo Float32/Int32 usa atomiche per gli indici ripetuti.
Indici Int32/Int64 conservati, sorgente densa, destinazione con stride iniettivi
validati. Prima di ogni modifica vengono verificati layout, limiti, dtype,
allineamento, overflow e separazione da sorgente/indici. Una validazione GPU di
tutti gli indici legge un solo stato da quattro byte; un indice invalido
preserva interamente la destinazione. Workspace fisso riutilizzato, nessuna
allocazione ausiliaria proporzionale al modello. Copie con duplicati discordanti
non garantiscono un ordine deterministico; neppure la somma FP32 atomica ha un
ordine deterministico. I test copy con duplicati usano contributi concordanti.

Nel solo frontend HIP, sorgente o indici che condividono la destinazione sono
clonati prima della scrittura, conservando la snapshot dei dati consumati.
Gli altri backend mantengono il comportamento precedente. I limiti dtype del
frontend restano quelli esistenti; i dtype ulteriori della copia privata non
sono automaticamente esposti da index_copy_.

Implementati anche strided_scatter normale/immediate e Int32→Float32, necessari
al writeback delle viste tramite mutate_logical_view. Metadati passati per valore,
nessun upload di array shape/stride o sincronizzazione aggiuntiva per immediate.
Non è ancora lo scatter indicizzato generico, né compattazione/refinement.

Aggiunti 108 confronti CPU di copy/add (dtype, assi, tipi indice, duplicati),
controlli su byte/padding/riserve, invalidi transazionali, alias, overflow,
layout sovrapposti ed empty. Quindici casi di writeback strided controllano bit e
conversione Int32→Float32. Nel core reale passano snapshot di sorgente/indici,
accumuli ripetuti, rank 1, rifiuti che preservano il modello e writeback delle
viste trasposte. Aggiunti controlli copy/add CUDA prima/dopo lo scope HIP.
I test restano sintetici; non è ancora una misura del training sul dataset.

Verifica del blocco scritture: **20/20 in 92,63 s**, report locale
`hip-index-write-final-suite.xml`; matrice core **9/9**,
`index-write-core-matrix.json`. Primo aggiornamento TensorModel delta 0,
scarto RGB massimo 1,76951289e-8–2,04890966e-8 nei tre confronti a 400 step.
Compilazione del file condiviso con HIP OFF ancora verificata. Nessun nuovo
benchmark sul dataset, né test completo macOS/Linux in questo blocco.


### Generatori casuali nativi

Dopo il checkpoint `1e0348a6f`, implementati uniformi, Bernoulli, interi Int32
e normali FP32. Philox4x32-10 usa quattro estrazioni per contatore, senza stato
persistente per elemento, buffer proporzionali al modello o readback CPU.
Gruppi completi allineati scrivono quattro valori; code e viste allineate solo
a quattro byte usano store limitati alla regione valida. Il prefisso non cambia
al variare della lunghezza a parità di seme e distribuzione. I casi costanti
usano fill. Normali dispari non richiedono scratch o sincronizzazione aggiuntiva;
la sola condizione HIP evita lo scratch nel frontend. Gli altri backend
mantengono il loro percorso e la loro sequenza. La sequenza HIP non è promessa
bit-identica a CUDA/Metal/Vulkan; multinomial è ancora da implementare.

Verificati vettore noto Philox a chiave/contatore zero e mapping esatto CPU per
uniformi/Bernoulli/interi con due semi a 64 bit. 131.073 estrazioni per caso
controllano momenti, correlazione dei normali e frequenze degli interi. Inclusi
ripetibilità, prefissi da 263, empty, offset a quattro byte, code/sentinelle,
std nullo e segno del mean, limiti uniformi finiti estremi, intero signed a tutto
range, parametri invalidi e overflow. Per i normali verificati anche mean/std
non standard; i confronti statistici non provano qualità del training multiview.

Nel core reale passano reset/avanzamento del seme normale, isolamento del seme
esplicito, scrittura normale su vista trasposta e factory Bernoulli/interi.
Aggiunto controllo dei normali dispari CUDA prima/dopo HIP. Verificata inoltre
la compilazione sintattica di tensor_random_ops.cpp con HIP OFF. Nessuna modifica
a kernel o generatori CUDA/Metal/Vulkan e nessun nuovo benchmark sul dataset.

Verifica RNG finale: **20/20 in 102,08 s**, report locale
`hip-rng-final-suite.xml`; matrice core **9/9**, `rng-core-matrix.json`.
Il controllo normale CUDA dispari è passato prima/dopo HIP, così come HIP
con CUDA non esposto al processo. Nei confronti sintetici a 400 step il primo
aggiornamento resta delta 0 e lo scarto RGB massimo è
1,86264515e-8–2,42143869e-8. Soglie conservate e nessun dato di throughput
sul dataset. Restano 68 metodi esplicitamente non implementati, oltre alle
integrazioni trainer/viewer e alle verifiche multipiattaforma già elencate.


### Ponte Camera/immagine preparata verso la sessione HIP (18 ottobre)

Aggiunto `src/training/hip/camera_frame.hpp`: accetta la vera `core::Camera`
e un'immagine CPU float32 CHW già preparata dal loader. Deriva gli intrinseci
scalati dalle dimensioni effettive del target e dalla calibrazione originale,
ignorando le dimensioni mutabili dell'ultima immagine caricata sulla Camera.
Preserva R/T world-to-camera, clipping e valori colore senza applicare ulteriori
trasformazioni. Conserva una copia CPU del target per resistere al riuso del
buffer del loader; questo costo va misurato nel successivo percorso multiview.
Rifiuta camere non pinhole, distorsione non nulla, immagini HWC/dtype o device
errati, clipping e intrinseci invalidi. Non sostituisce l'undistortion del loader.
Il descrittore viene formato su richiesta per mantenere valido il puntatore
anche dopo move/copy del proprietario.

Il test `lfs_hip_tensor_training_smoke` ora allena per 400 step il modello Tensor
tramite questo ponte, confrontandolo con due sessioni col descrittore originale.
Passa in build HIP senza CUDA, build combinata e build combinata con CUDA
invisibile/HIP visibile solo nel processo figlio. MSE iniziale 1.8062774e-5,
finale circa 2.426024e-7; differenza massima immagine rispetto al controllo
<=2.24e-8. Verificati anche resize con dimensioni Camera obsolete, convenzione
R/T, proprietà del target, rifiuto di HWC, distorsione, ortografica e clipping.
Sono ancora dati sintetici, sei gaussiane attive su 263 righe, non qualità o
velocità su dataset reale. Evidenze locali: `camera-frame-*.log`.

**Stato preciso del collegamento al trainer:** `Trainer::train_step` usa famiglie
separate di TrainingOps (rasterizzazione, loss, Adam, SH, immagini, strategia),
mentre la sessione sperimentale esegue questi stadi come una singola operazione.
`gpu_ops::SessionOps` riguarda arena e profiling, non è un'interfaccia per questa
sessione: collegarvi il modulo non abilita il training. La tabella HIP resta
intenzionalmente vuota e il trainer principale continua a rifiutarla.
Per la prima prova reale servono ancora: adattamento dell'inizializzazione del
modello e del loader multiview, un loop esplicito a topologia fissa con schedule,
export/eval del risultato e confronto riproducibile. Per il trainer completo
servono inoltre famiglie TrainingOps reali e strategie/refinement; non basta
abilitare il selettore. Nessun comportamento o kernel CUDA/Metal/Vulkan è stato
modificato da questo ponte, che viene incluso soltanto dal test HIP opt-in.


### Primo training su dataset reale e limite di scala (18 ottobre, sera)

**Risultato:** esiste ora `lfs_hip_dataset_train`, eseguibile locale opt-in ed
escluso dalla build predefinita. Riutilizza il vero modulo I/O dell'app per
COLMAP, camere, seed PLY/progetto e salvataggio PLY. Non è ancora il `Trainer`
della GUI e non pubblica HIP nei selettori. La tabella TrainingOps HIP rimane
vuota: il controllo di ammissione continua a rifiutare le configurazioni complete
non implementate, senza fallback nascosto.

Il modello usa una singola allocazione Tensor HIP, regioni compatibili con il
controllo core e il ponte `TensorModel`; il modulo nativo prende in prestito i
parametri e possiede gradienti/momenti/workspace. Il seed `.licht` usa il CKPT
associato dal documento e ripristina **solo i parametri**, non i momenti Adam o
la strategia. SH superiori a zero, righe congelate e righe eliminate non vengono
scartati silenziosamente: sono rifiutati da questo runner. Il modo diagnostico
`points` inizializza invece i punti SfM con scala isotropa esplicita 0.01 e
opacità 0.1, senza pretendere equivalenza con l'inizializzatore dell'app.

Configurazione reale: dataset `statua_budda_villaggio`, 82 camere nel loader,
due viste selezionate agli indici 0 e 41, SH0, topologia fissa, sfondo nero,
L1+SSIM con peso 0.2 e valid padding, seed shuffle 42, resize massimo 1920.
Ordine delle viste e tempi sono in CSV. Le immagini CPU float CHW preparate
restano in RAM, cache del sottoinsieme limitata a 512 MiB; la sessione riceve
un frame GPU per volta. Il budget nativo è 2 GiB e non rappresenta un limite
sull'intero processo, sulle riserve del driver o sulla RAM UMA fisica.
Il loader del runner usa un'arena con concorrenza uno per conservare lo scope
HIP esplicito anche durante la costruzione delle camere. Non cambia il backend
di processo e non attiva un contesto Tensor Vulkan.

Sono stati necessari tre collegamenti mancanti:

- Inversione della lente pinhole in HIP: generata dalle formule CUDA già nel
  repository, senza modificare quei file. L'implementazione generica di
  `image_warp` rientrava ricorsivamente nella mappa inversa, causando stack
  overflow durante la calibrazione COLMAP. HIP ora implementa la mappa nativa
  e rifiuta esplicitamente gli altri warp GPU non portati. Test su identità,
  radiale positiva, fold radiale, tangenziale accoppiata, calibrazioni invalide,
  residuo di proiezione sotto 0.011 pixel e assenza di raggio oltre il fold.
- Undistortion del target CPU: riuso dell'identico kernel area/Lanczos esistente
  con righe indipendenti su otto worker, coordinate globali e ordine delle somme
  per pixel invariati. Il test confronta ogni float con il percorso CPU intero.
  Nessun kernel CPU/GPU degli altri backend viene modificato. Il primo tentativo
  seriale a 1920 è stato interrotto per la preparazione troppo lenta, prima
  degli step; non è un risultato di velocità del training.
- Import/export di modelli CPU in scope HIP: lo storage SH vuoto resta HIP,
  evitando l'inizializzazione indesiderata di Vulkan. La condizione è compilata
  solo con HIP e scatta solo nello scope HIP; CUDA/Metal/Vulkan mantengono il
  comportamento precedente. Corretta inoltre una dipendenza di compilazione
  PLY MSVC da `__restrict__`, già definita dagli header CUDA ma assente quando
  si costruisce senza CUDA. Nessuna modifica ai kernel CUDA/Metal/Vulkan.

Le esportazioni passano nel lettore PLY dell'app: means, scaling, SH0 e opacità
coincidono esattamente con i parametri letti dalla sessione. Le rotazioni devono
coincidere entro 3e-7 con la normalizzazione unitaria delle originali, perché
questa è la trasformazione già eseguita dall'exporter SplatData. Non si deve
pretendere che il PLY conservi la norma grezza del quaternione Adam: anche
`initial.ply` la normalizza. Per riprodurre esattamente l'inizializzazione usare
il seed originale, non il PLY normalizzato.

#### Misure conservate

| Caso reale | Step | Gaussiane effettive | Dimensioni | Step medio | Frame medio | Memoria viva nativa massima campionata |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Primo controllo | 100 | 19.555 | 360×480 | 5.521 ms | 1.059 ms | 19.839.188 B |
| Controllo a 1920 | 200 | 19.555 | 1440×1920 | 36.160 ms | 8.793 ms | 214.758.188 B |
| Prova estesa dopo correzione budget | 50 | 1.000.000 | 1440×1920 | 1007.675 ms | 10.276 ms | 1.364.844.664 B |

Primi due casi: scartati 20 step iniziali; 80/180 campioni rispettivamente.
Caso da un milione: scartati cinque step; 45 campioni, con marker diagnostici
non sincronizzati fra le fasi, quindi misura wall complessiva e non confronto
puro di throughput. Le attribuzioni intermedie di quel log non sono utilizzate.
Preparazione immagini, valutazione, lettura parametri ed export sono esclusi
dai tempi degli step. Il cap di un milione non è il numero effettivo dei primi
due casi: il seed iniziale ha 19.555 Gaussiane e qui non avviene densificazione.

Il seed iniziale comune conserva SHA-256
`4273c94a3c29daffc84c490651510d91ac71ba9cfb9eba8aa3abc0fced3f63f7`.
Una copia locale indipendente è in `.local/rocm-tensor-study/dataset-seed-19555.licht`.
Il caso da un milione legge invece il CKPT del progetto già usato per i test
pesanti, `final-adam-training-cuda/project.licht`, senza ripristinare Adam.

Qualità sulle sole viste usate per il training, **non holdout**:

- 19.555 Gaussiane, 480, 100 step: MSE 0.135345607 → 0.0866405702,
  PSNR finale 10.622787 dB.
- 19.555 Gaussiane, 1920, 200 step: MSE circa 0.139025 → 0.0505028804,
  PSNR finale 12.966839 dB. Controllo visivo prima/dopo/target eseguito:
  migliora la copertura della statua, ma restano sfocatura e buchi evidenti.
- 1.000.000 Gaussiane, 1920, 50 step: MSE 0.0735882874 → 0.0438907253,
  PSNR 11.331913 → 13.576272 dB. Non dimostra equivalenza con CUDA o MCMC.

Questi risultati non vanno confrontati direttamente con 114.591 ms + 9.936 ms
loader del percorso Vulkan Fast/MCMC: qui il raster è 3DUT world-ray, la
strategia è assente, i momenti ripartono e cambia il campionamento delle camere.
La prova da un milione mostra che **la prestazione richiesta non è ancora
raggiunta**; quella piccola non poteva dimostrarla.

#### Budget e diagnosi del caso da un milione

La prima esecuzione senza correzione si interrompeva dopo tre step: la crescita
geometrica dei buffer delle intersezioni non sottraeva lo spazio della
contemporanea sostituzione dello scratch radix-sort. Ora il dry-run hipCUB
calcola lo scratch prima di scegliere la capacità; il budget include vecchi e
nuovi buffer e vecchio e nuovo scratch fino al commit. Nessuna soglia di resa,
formula, parametro, gradiente o momento è stato cambiato per ridurre memoria.
La prova successiva completa 50 step sotto il medesimo limite di 2 GiB.
Un ulteriore controllo registra picco 2.147.483.628 B con limite 2.147.483.648 B.
Restano circa 19–22 milioni di intersezioni, quindi il costo non è spiegato
semplicemente da un milione di righe inattive: nel seed soltanto 1.816 opacità
su un milione sono sotto la soglia alpha del renderer.

Aggiunto `LFS_HIP_TRAIN_PROFILE=1`, solo per l'ambiente del processo figlio.
Default disabilitato: nessun evento o fence aggiuntivo nel percorso normale.
Il modo diagnostico sincronizza i confini per attribuire correttamente tempi
anche in presenza di allocazioni e fence del runtime. Eventi non sincronizzati
avevano attribuzioni ambigue fra le fasi e non sono usati per decidere interventi.
I numeri seguenti includono dispatch/attese del runtime; sono diagnostica,
non tempi puri di un kernel e non un benchmark del percorso senza profiler.
Media degli step 2–5, una vista a 1440×1920, un milione di Gaussiane:

| Fase | ms |
| --- | ---: |
| Proiezione | 1.409 |
| Conteggio/scan | 0.778 |
| Binning/ordinamento | 112.215 |
| Forward raster | 201.826 |
| Azzeramento gradienti | 1.590 |
| Gradiente L1/SSIM | 6.398 |
| Backward raster | 739.853 |
| Adam | 6.181 |

Il backward è quindi la priorità concreta. Non è stato attivato alla cieca il
bounding box dipendente dall'opacità già presente nella proiezione: usa anche
una compensazione che questo raster world-ray non applica e potrebbe tagliare
contributi. L'ottimizzazione deve preservare il contratto di rendering, non
ridurre il lavoro alterando la qualità. Il codice oggetto è nativo gfx1150,
wave32; raccolti anche i metadati delle risorse compilate per l'analisi successiva.

#### Verifiche finali e lavoro restante

Dopo la correzione del budget: 20/20 test nativi in 88.78 s e 9/9 controlli con
vero lfs_core (senza CUDA, CUDA+HIP, CUDA non visibile/HIP visibile). Includono
400 step e confronto con due controlli privati, alias dei parametri, code,
cache, RNG, dtype, indici, budget/ripresa, SH, loss e interop standalone Windows.
Passano la compilazione sintattica di SplatData senza macro HIP e quella del
lettore PLY con CUDA. Non sono nuovi benchmark CUDA o test runtime macOS.

Evidenze locali: `dataset-budget-final-suite.xml`,
`dataset-budget-final-core-matrix.json`, `dataset-source-manifest.json`,
`dataset-complete-480/`, `dataset-parallel-1920/`,
`dataset-million-profile-1920/`, `dataset-million-synchronized-profile/`,
`raster-gfx1150-metadata.txt`, tutte sotto `.local/rocm-tensor-study/`.

Priorità successive, prima di esporre il backend agli utenti:

1. Ottimizzare/portare raster e backward nativi per il percorso Fast/3DGS usato
   dalla baseline, senza confonderlo con il prototipo 3DUT. Confrontare pixel,
   gradienti e tempi a stesso modello/dataset/configurazione.
2. Riutilizzare il loader dell'app con cache e offload per tutte le viste,
   evitando la limitazione del sottoinsieme float residente di questo runner.
3. Implementare famiglie TrainingOps e strategia/refinement richieste dal
   Trainer, anziché allentare l'ammissione o simulare operazioni mancanti.
4. Collegare allocator del modello, interop viewer e crescita storage, poi
   selettori/fallback/restart sulla base concordata del lavoro precedente.
5. Eseguire controlli di qualità multiview lunghi e build complete Windows/Linux;
   ripetere i controlli dei backend esistenti in base alle parti condivise toccate.


### 19 ottobre 2026 — batch HIP ridotto e verifica del percorso Media Ingest

Il backward dual-pixel mantiene 128 thread e ora prepara 128 Gaussiane per
batch, invece di 256, soltanto nella copia generata per HIP. Lo staging viene
adattato insieme alla capacità; il generatore rifiuta sorgenti non più conformi.
La LDS RGB senza edge score scende da 24 KiB a 12 KiB per blocco. Non cambia
l'ordine inverso delle intersezioni dentro ogni pixel, né la formula dei
contributi, le soglie alpha, la loss o Adam. Può cambiare l'ordine globale delle
somme atomiche: non viene promessa identità bit per bit delle traiettorie.
Nessun sorgente dei kernel CUDA, Metal o Vulkan è stato modificato.

Riferimenti consultati: [linee guida HIP AMD](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)
per il compromesso fra LDS, registri e occupazione; [GSplat ROCm AMD](https://github.com/AMD-Ecosystem/gsplat-amd)
come riferimento pubblico di porting. Quest'ultimo dichiara MI300X e Ubuntu:
non costituisce una misura per Radeon 890M/Windows. Non è stato importato codice
esterno. Il Fast del repository contiene già culling per sottotile/warp: resta
un candidato per il successivo percorso nativo Fast/3DGS, distinto da 3DUT.

#### Contratto delle immagini e tempi del loader

L'audit ha individuato un errore nel runner sperimentale: ridimensionava in byte
prima della correzione lente, invece di usare un solo resampling combinato dal
sorgente originale. Ora usa il decoder CPU Media Ingest (`core::load_image` →
`image_codecs`), conserva il sorgente RGB nativo in RAM come byte, e riusa il
resampler CPU area/Lanczos templato UInt8. Il target viene riportato sulla griglia
16-bit prevista dal ramo portable per decode non float. Il test confronta ogni
float con il riferimento CPU per sorgente byte e griglia ridotta; rimane anche
il precedente confronto completo float. Questo allinea il contratto, non prova
identità bit per bit con tutte le implementazioni GPU o con la cache JPEG lossy.
I vecchi risultati del runner con resize preliminare restano storici: non vanno
confrontati come misura di qualità a target invariato con questi nuovi target.

Non è ancora un collegamento completo a `PipelinedImageLoader`: il suo percorso
richiede TensorUpload asincrono e SharedImageOps HIP ancora assenti. Nessun bypass
verso CUDA/Vulkan viene usato. Cache JPEG preparata, spill e prefetch del loader
applicativo non sono disponibili in questo runner. I log CLI dichiarano questi
limiti. La preparazione avviene una volta, prima delle iterazioni, e i target float
ridotti restano nella cache di sottoinsieme CPU con limite 512 MiB. Il sorgente
nativo non viene caricato sulla GPU o espanso in float a piena risoluzione.
Il preflight limita ciascuno dei due buffer transitori RGB byte a 256 MiB e
controlla il budget del target prima della preparazione. Questi limiti non sono
un tetto all'intero processo o al driver.

`preparation.csv` separa calibrazione, decode/resize, planarizzazione, warp e
copia di ownership; `steps.csv` mantiene separati bind/upload e step. Riutilizzare
lo stesso CameraFrame immutabile evita un altro bind/upload nel solo runner.
Il C ABI pubblico continua sempre a leggere, validare e caricare il target
prestato: modifiche in-place dei chiamanti non vengono ignorate.

Nel confronto con due viste native 6048×8064, target 1440×1920:

| Preparazione CPU una sola volta | Vista 0 | Vista 1 |
| --- | ---: | ---: |
| Decode | 110 ms | 315 ms |
| Planarizzazione byte | 48 ms | 96 ms |
| Resize/undistort combinato | 16.15 s | 19.54 s |
| Ownership target | 6.1 ms | 6.4 ms |

Circa 36.3 s totali all'avvio, 66.355.200 B di target persistenti in RAM.
La lente usa l'accurata quadratura del codice esistente; questo costo CPU non è
ripetuto a ogni iterazione. Con una vista soltanto non serve alcun nuovo upload
nel loop. Con due viste e shuffle 42 vengono effettuati 19 upload su 30 step.
Per il loader completo occorrono ancora il warp nativo HIP, upload asincrono e
riuso della cache esistente, evitando una seconda implementazione dei protocolli
di invalidazione/concorrenza/budget.

#### Confronto a parità di input, senza profiler

Seed originale del progetto con **1.000.000 Gaussiane reali**, SH0, Adam fresco,
3DUT world-ray, L1/SSIM 0.2, 30 step, due viste a 1920. Stessi target preparati
con il percorso corretto, verificati identici byte per byte nei PPM di controllo;
MSE iniziale identico 0.0736701273193. I primi cinque step sono esclusi dai tempi
riportati (25 campioni); nessun profiler o fence diagnostico aggiuntivo.

| Misura | HIP batch 256 | HIP batch 128 |
| --- | ---: | ---: |
| Step medio | 1038.564 ms | 791.448 ms |
| Step mediano | 1016.860 ms | 804.795 ms |
| Bind/upload medio nel loop | 7.436 ms | 7.199 ms |
| MSE finale sulle viste di training | 0.0528338158365 | 0.0528338372662 |
| Picco nel budget sessione | 2.147.483.628 B | 2.147.483.640 B |

Riduzione del tempo medio dello step **23.79%** in questa coppia. Il picco è
contabilità della sessione sotto limite 2 GiB, non VRAM totale. Non è un confronto
con Fast/MCMC Vulkan o CUDA; il prototipo resta molto più lento della baseline
Fast precedente. La distinta diagnostica sincronizzata a cinque step e una vista
mostra backward circa 747 → 512 ms; quei tempi includono i fence diagnostici.

Ispezione visiva di entrambe le viste: vecchio e nuovo appaiono coerenti, con gli
stessi limiti di un training molto breve (buchi, sfocatura, sfondo incompleto).
Differenza massima render esportato 2/255; errore assoluto medio 0.00202 e 0.00284
livelli byte, rispettivamente. I parametri non sono tutti identici: per esempio
logit opacità max 0.10899/RMSE 0.000465, posizione RMSE per componente circa
5.5–6.6e-7. Non si confonde questa misura con equivalenza universale dopo training
lunghi o con una verifica su viste di holdout.

#### Controlli

- 20/20 CTest nativi, 86.79 s, incluso interop Windows e budget a un milione.
- 9/9 controlli core: HIP senza CUDA, CUDA+HIP entrambe visibili e CUDA nascosta
  con HIP esplicitamente visibile. Nessun nuovo benchmark di training CUDA.
- Fixture core 400 step: MSE 1.8062774e-5 → 2.42602397e-7, due controlli privati
  coerenti; alias esatti di tutte e cinque le famiglie e righe riservate intatte.
- Confronto CPU UInt8 + resize/warp e precedente confronto CPU float esatti.
- Runner finale con preflight: 19.555 Gaussiane, 480, due step completati;
  MSE 0.160193799169 → 0.157264501643, zero upload ripetuti per una vista.
- Output esistente rifiutato con exit 1; hash PLY/CSV/summary invariati.
- Nessuna build runtime macOS/Linux nuova; modifiche confinate al prototipo HIP.

Evidenze aggiuntive in `.local/rocm-tensor-study/`: `batch128-suite.xml`,
`batch128-actual-core-matrix.json`, `batch128-ingest-comparison.json`,
`batch128-parameter-comparison.json`, `batch128-quality-preview.png`,
`batch256-ingest/`, `batch128-ingest/`, `ingest-preflight-final-check.json`.


Ripetizione di controllo con il vecchio batch, stessi 30 step e target: tempo
medio 1021.640 ms, MSE finale 0.0528337684407. La differenza di MSE fra le due
esecuzioni vecchie è 4.74e-8, contro 2.14e-8 fra prima vecchia e nuova.
Anche il vecchio percorso da solo presenta differenze parametriche: logit
opacità max 0.10252/RMSE 0.000462, scala 0 max 0.019259/RMSE 0.0000571,
posizione RMSE circa 6e-7; render max 1/255 e 2/255 per le due viste.
Sono dello stesso ordine delle differenze osservate con il nuovo batch.
La seconda misura del riferimento conferma un vantaggio dello step nell'ordine
del 22.5–23.8%, senza dimostrare equivalenza statistica su scene o training
lunghi non provati. Evidenze: `batch256-ingest-repeat/` e
`batch256-repeat-comparison.json`.


### 19 ottobre 2026 — preparazione immagini nativa HIP e controlli estesi

#### Esperimenti sul raster: conservare soltanto vantaggi verificati

Sono stati provati in successione, con DLL di riferimento conservate:

| Esperimento | Riferimento | Variante | Decisione |
| --- | ---: | ---: | --- |
| Limitare lo staging all'ultimo contributo del forward, 30 step/2 viste | 801.461 ms | 791.863 ms | ~1.2%, non distinto con sicurezza dalla variabilità; scartato |
| Batch 64 anziché 128, 10 step/1 vista, scarto primi 2 | 833.048 ms | 839.470 ms | Scartato |
| Gradiente scale ricavato dalla matrice già disponibile, 30 step/2 viste | 789.407 ms | 791.387 ms | Scartato; formule originali ripristinate |
| Un pixel per thread, riserva LDS artificiale rimossa, 10 step/1 vista | 834.303 ms | 965.706 ms | Scartato |

Le formule equivalenti e i layout alternativi passavano le differenze finite,
ma ciò non dimostra velocità migliore sul modello reale. Il raster finale resta
quello di `949535f59`: verificati **16 file generati identici byte per byte**.
Il generatore ora conserva i timestamp degli output invariati; CMake collega
separatamente i sorgenti delle unità raster/binning/SH e tutti gli header comuni.
Una modifica del raster non ricompila binning e SH senza motivo. Il generatore
rifiuta inoltre un numero di thread non conforme al caso wave32 verificato.

Resta nel test un nuovo controllo utile a prescindere dall'esperimento: prefisso
opaco di 2 Gaussiane su 6/263, coda non accettata senza gradienti, confronto
prima/dopo disattivazione della coda, tile parziali e completamente trasparenti.
Il primo tentativo del nuovo test leggeva una coda senza sincronizzare il suo
stream nonblocking; corretto il readback del test, senza allentare tolleranze.

#### Warp RGB nativo: componente concreta per il loader

Aggiunto `lfs_hip_warp_rgb` al C ABI privato, con collegamento ai veri
`HipBackendOps::image_warp` per RGB pinhole UInt8/F32 CHW. Usa una copia generata
delle funzioni di campionamento e del kernel CUDA già presenti, non una nuova
formula di filtro. La sorgente CUDA non viene modificata. La proiezione usa FMA
esplicito come il riferimento CPU, e ogni batch mantiene le coordinate globali.
Sono verificati tipo, forma, contesto proprietario, offset/allineamento,
sovrapposizione, indici signed, calibrazione e lavoro massimo. Le coordinate
proiettate non rappresentabili sono escluse prima delle conversioni intere.
L'identità normalizza byte o copia float con loop grid-stride, verificato anche
sopra il limite di 4096 blocchi della griglia del runtime.

L'originale viene decodificato dal codec CPU Media Ingest e mantenuto in byte in
RAM. Una copia byte temporanea sulla GPU viene usata per il warp e rilasciata
prima di passare alla vista successiva; nessuna espansione float a risoluzione
nativa sulla GPU. Il risultato ridotto torna nella cache CPU esistente del
runner, con la medesima griglia 16-bit del ramo portable. È verificato che i byte
GPU attivi del core tornino al valore precedente dopo ogni readback.

Limiti di lavoro nativi: footprint massimo 64 e massimo 2.097.152 valutazioni di
quadratura per batch di righe, fino a 32 righe; l'ultima riga di ogni batch ha
un limite esplicito anche quando il numero non è multiplo di 16. Il runner usa
il riferimento CPU, con avviso CLI e contatori, se il lavoro è oltre questi
limiti. Questo fallback viene scelto prima di allocare immagini GPU e riguarda
soltanto la preparazione; il training rimane HIP. Non vengono nascosti errori
GPU, invalidità di calibrazione o di ownership. Il C ABI continua a rifiutare
richieste non supportate. La cache dei target resta 512 MiB; i sorgenti byte sono
limitati a 256 MiB ciascuno, con al massimo due buffer host e uno GPU transitori.

`LFS_HIP_CPU_IMAGE_PREP=1` sceglie esplicitamente il riferimento CPU.
`LFS_HIP_VALIDATE_IMAGE_PREP=1` confronta invece ogni float GPU con un ulteriore
resampling CPU, prima della quantizzazione. È diagnostica: il suo costo CPU è
incluso nel tempo di preparazione e non va citato come velocità normale.
I log e `preparation.csv` distinguono upload, warp/readback, validazione, byte
GPU attivi e riservati. Summary e CSV riportano il backend effettivo per vista
e il numero di fallback.

#### Risultati sul dataset reale

Stesso progetto seed con 1.000.000 Gaussiane, SH0, Adam fresco, 3DUT world-ray,
L1/SSIM 0.2, due viste (0/41 nell'ordine di 82) e target 1440×1920. Nessun nuovo
benchmark CUDA e nessun confronto improprio con Fast/MCMC.

| Preparazione una sola volta | CPU | HIP |
| --- | ---: | ---: |
| Totale due immagini, run 30 step | 36.860 s | 1.442 s |
| Totale due immagini, run 200 step | 36.567 s | 1.453 s |
| Upload sorgente per vista (run 200) | — | 16–17 ms |
| Warp + readback per vista (run 200) | — | 489–536 ms |
| Target persistenti CPU | 66.355.200 B | 66.355.200 B |

Preparazione circa **25 volte più veloce**, non un miglioramento del raster per
iterazione. Nella diagnostica completa ogni float delle due immagini reali è
stato controllato: max 6.795e-6 / 8.285e-6, MSE 4.194e-15 / 4.898e-15. La prova
sintetica indipendente riporta max 3.576e-7 per float e 5.364e-7 per byte.
Le tolleranze del test (max 4e-5, MSE 1e-10) erano fissate prima della misura e
non sono state allentate. Nei target PPM ci sono differenze massime di un livello
byte su pochi pixel: non sono immagini dichiarate bit-identiche.

Nel run diagnostico i byte GPU attivi del core sono 56.006.232 prima e dopo il
warp; il pool riservato dopo il readback è 89.296.732 B. La differenza comprende
il piccolo buffer di output ridotto riutilizzabile nel pool generale da 64 MiB.
Il sorgente originale da 146.313.216 B (139.5 MiB) supera quel pool e viene
rilasciato. Questi contatori riguardano il contesto Tensor del core, non tutta la
VRAM o le allocazioni del modulo training; il modello condiviso non va contato
due volte sommando contabilità logiche di moduli differenti.

| Training a 200 step | Preparazione CPU | Preparazione HIP |
| --- | ---: | ---: |
| MSE iniziale sulle viste di training | 0.0736701273193 | 0.0736701274027 |
| MSE finale | 0.0170283651307 | 0.0170281177689 |
| PSNR finale | 17.688270 dB | 17.688334 dB |
| Step medio, scartati i primi 10 (190 campioni) | 700.989 ms | 699.899 ms |
| Bind/upload medio | 9.124 ms | 8.658 ms |
| Picco nel budget sessione | 2.147.483.628 B | 2.147.483.640 B |

Nessun peggioramento aggregato o visibile osservato nella coppia. Le traiettorie
non sono identiche: errore medio dei render finali 0.044/0.051 livelli byte,
RMS 0.216/0.243 e massimi locali 12/19. Sono dati completi, non una promessa di
uguaglianza pixel per pixel dopo training lunghi. Ispezionate entrambe le viste:
statua e basamento migliorano rispetto ai 30 step, le due varianti risultano
coerenti; lo sfondo rimane incompleto e sfocato, come atteso da un modello fisso
ottimizzato per sole due viste. Non è una validazione di holdout o del Trainer
completo. Il tempo dello step cala durante i 200 passi con il modello evoluto;
non va attribuito all'ottimizzazione della preparazione.

#### Collegamento restante

Il decoder è quello di Media Ingest e il warp Tensor HIP è ora nativo, ma il
runner non è ancora `PipelinedImageLoader`. Restano upload asincroni, semantica
delle code/fence e registrazione SharedImageOps per riusare cache JPEG, spill,
prefetch e invalidazione dell'app, senza inventare una seconda cache. I target
persistenti restano CPU e il training non ottiene vantaggi simulando offload.
Per abbassare sostanzialmente il costo delle iterazioni la priorità è il percorso
Fast/3DGS nativo, che porta la derivata della proiezione fuori dal loop sui pixel;
le microvarianti world-ray provate non hanno dato altri guadagni affidabili.
TrainingOps, strategie/refinement, allocator crescente e interop viewer vanno
poi collegati e verificati prima dell'esposizione nelle preferenze. Le opzioni
HIP rimangono OFF di default; nessun comportamento CUDA/macOS modificato.

Evidenze: `tile-last-comparison.json`, `batch64-comparison.json`,
`cached-vjp-comparison.json`, `single-bwd-comparison.json`,
`native-raster-inputs-unchanged.json`, `native-warp-comparison.json`,
`native-warp-200-comparison.json`, `native-warp-validated/preparation.csv`,
`native-warp-200-quality-preview.png`, tutti sotto `.local/rocm-tensor-study/`.

### Porting FastGS HIP: primo strato matematico (19 ottobre 2026)

`prepare_fast.py` genera copie private di sei dipendenze matematiche del percorso
FastGS CUDA (riduzioni warp/blocco, effective rank, codec Adam/SH, screen share,
scala del passo delle medie). Non modifica gli originali, non registra un backend
e non costituisce ancora il porting del rasterizzatore FastGS. Maschere HIP a
64 bit; l'ammissione del test richiede wave32. Il test opt-in
`LFS_HIP_PROBE_FAST_MATH` compila tutti e sei gli header e verifica sulla Radeon
le riduzioni scalari/vettoriali da 32 a 1024 thread, i gradienti effective-rank
contro differenze finite e i limiti del passo delle medie. CTest superato.
I codec e screen-share sono per ora verificati in compilazione, non ancora con
parità funzionale completa. Nessun nuovo guadagno sui tempi di iterazione misurato.

Chiusi anche i controlli della preparazione immagini: 20/20 test del modulo,
9/9 controlli delle configurazioni core HIP/CUDA, invarianza delle allocazioni
live dopo la preparazione, preservazione di output esistenti e fallback per
footprint eccessivo con avviso CLI. Il training del fallback rimane HIP.
Restano da portare orchestrazione FastGS, binning, forward/backward e relativo
ciclo di memoria; poi strategie dinamiche, loader asincrono e trainer principale.

### FastGS HIP: forward/backward nativi e sessione FP32

Il porting privato ora comprende forward FastGS con visibilita compatta,
scan/sort hipCUB, tile culling, backward blend/proiezione e un adattatore
esplicito per estrarre i gradienti FP32 verso l'Adam HIP esistente. I sorgenti
CUDA originali rimangono invariati. `LFS_HIP_FAST_RUNTIME` e OFF per default;
la selezione C ABI deve avvenire prima del primo frame/step. Il runner la
richiede soltanto con `LFS_HIP_FAST_RASTER=1`, annunciandola nel log.
Non cambia automaticamente il rasterizzatore o backend dopo un errore.

Controlli iniziali: forward/gradienti analitici su tile parziali e interi,
mip-filter e warp culling attivi/disattivi; convergenza su SH0, SH1, SH2 e SH3,
cancellazione e ripresa. Tre nuovi test CTest superati. Restano da estendere
parita CUDA, condizioni limite, budget e integrazione nel trainer principale.

Confronto reale locale: un milione di gaussiane, SH0, resize 1920, due viste,
fresh Adam, L1/SSIM 0.2, medesimo seed e sequenza delle viste. Media al netto
dei primi 10 step su 200, senza profiler:

| Misura | HIP 3DGUT precedente | HIP FastGS |
| --- | ---: | ---: |
| Step medio | 699.898942 ms | 220.470032 ms |
| MSE viste training | 0.0170281177689 | 0.0170526653847 |
| PSNR | 17.688334 dB | 17.682077 dB |
| Allocazioni sessione live | 1,364,745,144 B | 825,276,532 B |
| Picco allocazioni sessione | 2,147,483,640 B | 988,714,992 B |

Sono rasterizzatori diversi: le immagini non sono pixel-identiche. Scarto medio
tra render finali 1.51/1.34 livelli byte nelle due viste, massimo 62/28. Ispezionata
la tavola `fast-native-200-quality.png`: statua/base e composizione coerenti;
restano le limitazioni delle sole due viste e della topologia fissa. Il dato non
e una validazione holdout o del ciclo completo con densificazione. La baseline
Vulkan aggiornata precedentemente registrata (114.591 ms/step + 9.936 ms loader)
non e ancora superata; non confondere il vantaggio su 3DGUT con un vantaggio
su Vulkan. Prossimo controllo: profilazione per fase e ottimizzazioni misurate.

### Ottimizzazioni Fast HIP e SH residente, 19 ottobre

Profilazione: il backward del blending domina il costo; la sola proiezione costa
circa 3 ms nel test breve. Gli accumuli FP32 generati inizialmente usavano CAS.
Gli atomici hardware sono applicati soltanto ai gradienti privati allocati con
hipMalloc, non ai parametri condivisi o alla memoria host. Non viene attivato un
flag globale di matematica non sicura. Seguono riduzioni native wave32, emissione
coalescente dei campi e staging del backward per wave con registri, eliminando
le barriere dell'intero blocco. Gli originali CUDA/Metal/Vulkan non cambiano.
La trasposizione completa dei gradienti ha peggiorato il tempo a circa 285 ms
nel test breve ed e stata scartata. Intrinseci exp/div e shuffle alternativi non
hanno dimostrato un guadagno affidabile individuale.

Stesso protocollo reale sopra, 200 step, medie escluse le prime 10:

| Misura | Fast iniziale | Fast ottimizzato FP32 | Fast ottimizzato fuso |
| --- | ---: | ---: | ---: |
| Step medio | 220.470032 ms | 121.243584 ms | 118.286947 ms |
| MSE viste training | 0.0170526653847 | 0.0170524515254 | 0.0170537332163 |
| Allocazioni sessione live | 825,276,532 B | 825,276,860 B | 713,589,108 B |
| Picco allocazioni sessione | 988,714,992 B | 988,715,508 B | 877,028,144 B |

La configurazione misurata abilita NATIVE_WAVE, DEVICE_ATOMICS, WAVE_REDUCE,
INTRINSIC_MATH, SCATTER_REDUCE, PACKED_GRADIENTS e WARP_STAGING, tutti sotto il
prefisso LFS_HIP_FAST_. Gli switch sperimentali restano OFF per default;
non attribuire il risultato aggregato al solo staging. La differenza di circa
3 ms tra FP32 e fuso deriva da una singola coppia di prove, non da una stima
statistica della variabilita. I parametri restano FP32; il fuso comprime i momenti
Adam e risparmia circa 112 MB. Non e ancora dimostrato un vantaggio su Vulkan,
che richiede un confronto attuale con identico protocollo e misurazione loader.

Ispezionata fast-optimized-200-quality.png: nessun degrado visibile tra i tre
render Fast nelle due viste. Scarto medio rispetto al Fast iniziale inferiore
a 0.055 livelli byte per vista; massimo 9 FP32 e 14 fuso. Queste sono viste di
training, due sole immagini, topologia fissa: non sostituiscono test holdout,
densificazione o validazione del Trainer completo.

SH1–SH3 FP32 possono ora essere presi in prestito dalla stessa allocazione dei
parametri base, senza buffer valori aggiuntivo. Test separati FP32/fuso coprono
alias esatto, aggiornamento dei coefficienti, padding/guardie, descrittori errati,
sostituzione non finita senza perdita dello stato e lifetime. Il collegamento
TensorModel richiede una vista Float32 esplicita e controlla anche la sua
sostituzione prima di lanciare lavoro GPU. Il controllo storage rifiuta SH
compressi, sovrapposizioni delle regioni riservate e crescita della topologia.

Suite nativa completa: 25/25; dopo ulteriori controlli di recovery SH, 4/4 test
mirati ripetuti. Integrazione con veri Tensor: test da 400 step superato, incluso
alias SH3 e rifiuto della vista sostituita. Restano chiusi l'accesso al Trainer
principale e la selezione automatica: operazioni Tensor mancanti, strategie di
crescita, checkpoint e protocollo asincrono loader/viewer richiedono altro lavoro.

### Porting scansioni inclusive

Implementato cumsum HIP per Float32/Int32 tramite il normale confine Tensor.
Il contratto pubblico resta invariato, compresi i tipi accettati da CUDA/Metal/
Vulkan. La C ABI HIP supporta anche Int64/UInt32. Vettori contigui usano hipCUB
DeviceScan, con workspace conservato nel contesto e contabilizzato insieme al
workspace delle riduzioni; gli assi strided usano BlockScan a tile di 256 valori
con carry. Non vi sono copie CPU dei dati o allocazioni scratch a ogni chiamata.
L'overflow intero e modulo 2^32/2^64; la somma parallela FP32 puo differire per
arrotondamento dalla somma CPU sequenziale in double.

Controlli prima del lancio: appartenenza al contesto, tipo, asse, allineamento,
span, overflow dimensioni, limite INT32_MAX elementi e assenza di sovrapposizioni
nel layout scrivibile. I descrittori errati preservano il contenuto. Verificati
assi multipli, trasposti, segmenti strided con padding, code oltre 256 elementi,
1,000,003 valori Int32, Int64 sopra la precisione FP32, overflow intero, view
vuote e riuso/contabilita del workspace. Il teardown libera il workspace solo
dopo avere completato la coda.

Validazione finale: suite nativa 25/25; matrice core 9/9 nelle build solo HIP,
CUDA+HIP e CUDA+HIP con NVIDIA nascosta. Artefatti scan-native-full-suite.log e
scan-final-core-matrix.json nella cartella locale dello studio. Restano 67
metodi esplicitamente non implementati nell'adapter Tensor: questo porting non
abilita ancora strategie di crescita o admission del Trainer. I tempi Fast
precedenti non sono rimisurati per attribuire un guadagno a un'operazione che
quel benchmark a topologia fissa non utilizza.

### Ordinamento nativo e esperimenti prestazionali successivi

Portati sort_1d/sort_2d: Float32, indici Int64 relativi all'asse, entrambe le
direzioni, ordine stabile per valori finiti uguali. Radix sort per vettori,
packing/segmented radix sort/unpacking per gli altri assi. Sorgente Tensor
immutato. La C ABI verifica proprieta, dimensioni, overflow, allineamento,
direzione e disgiunzione dei valori/indici prima di allocare o scrivere.
Il workspace resta nella coda privata e viene riutilizzato; contabilita e
picchi comprendono insieme scratch scan, sort e riduzioni. Il trim esplicito
o per soglia libera scan/sort dopo i consumatori pendenti e consente una nuova
allocazione alla chiamata successiva. Conservato il precedente comportamento
del piccolo workspace delle riduzioni. Restano 65 metodi non implementati.

Profilazione diagnostica del percorso ottimizzato: backward blending circa
70–75 ms, proiezione backward circa 5 ms, binning/ordinamento circa 20–30 ms,
forward circa 19 ms. Sono misure con sincronizzazioni di profilazione, non
tempi throughput da sommare ai benchmark ordinari.

Tre esperimenti privati scartati dopo test di correttezza e benchmark reale
breve identico (30 step, SH0, 1M, resize 1920, due viste, Adam fuso, media
escluse le prime cinque iterazioni):

| Esperimento | Tempo step | Decisione |
| --- | ---: | --- |
| Riferimento ripetuto | 137.09856 ms | Conservato |
| Hint occupazione backward 4 anziche 8 | 136.32468 ms | Meno dell'1%, nessun vantaggio convincente |
| Forward con staging per wave in registri | 160.89524 ms | Peggioramento, scartato |
| Filtro log-alpha conservativo nel backward | 144.02980 ms | Peggioramento, scartato |

Non confrontare queste medie brevi con 118.286947 ms su 200 step: il lavoro
cambia durante l'ottimizzazione. Nessuna di queste tre varianti e rimasta nel
codice. Artefatti e DLL sperimentali sono conservati nella cartella locale dello
studio. Il porting sort Tensor non sostituisce il sort privato gia presente
nel rasterizzatore e non produce da solo un guadagno del ciclo a topologia fissa.

Durante la rimozione degli esperimenti un blocco preesistente del generatore
privato era stato eliminato accidentalmente. Ricostruito dalle modifiche locali
precedenti: il confronto con la versione recuperata differisce soltanto nella
docstring iniziale (generator-recovery-comparison.json). Nessuna modifica ai
kernel originali CUDA/Metal/Vulkan. La validazione finale comprende anche il
ciclo scan/sort/reduce/trim ripetuto e il mantenimento del workspace riduzioni.

Esito finale di questo blocco: suite nativa 25/25 (sort-final-native-suite.log),
matrice core 9/9 con CUDA disponibile/nascosta (sort-final-core-matrix.json).
Il generatore produce 22 file identici ai sorgenti della build; una seconda
esecuzione conserva byte e timestamp (generator-final-idempotence.json).


### Verifica locale 19 ottobre: selezione, copie e specializzazione RGB

In parallelo è stato avviato lo studio dei sorgenti SDK richiesto dall'utente: fork separati di TheRock, rocm-libraries e llvm-project, provenienza esatta dal manifest 7.14.1 e confronto con upstream. Evidenze e valutazione PR in `.local/rocm-tensor-study/sdk-sources/README.it.md`; nessun SDK o driver sostituito. Il primo confronto ABBA del radix sort con header installati/upstream, compilatore invariato, non misura un miglioramento: circa 17,5–17,8 ms su 6.224.786 coppie uint64/uint32. Il tuning specifico richiede una verifica separata e non viene automaticamente integrato nel backend.

Sono ora native HIP anche masked_select, nonzero/nonzero_bool e cat/stack (60 metodi Tensor ancora non implementati). La selezione usa hipCUB DeviceSelect e trasferisce alla CPU solo il conteggio; capacità insufficiente non modifica la destinazione. Le copie concatenate validano tutta la lista prima di lanciare il primo trasferimento. Le viste HIP vuote ammettono operazioni di zero byte senza allocazione, mantenendo il rifiuto di puntatori estranei, offset invalidi e copie non vuote su puntatore nullo.

La verifica di finitezza dei target CPU mantiene il controllo IEEE di ogni elemento, ma usa una riduzione sui bit che il compilatore vettorizza. Nel confronto 30 step la fase frame media scende da 8,1603 a 4,0992 ms. Questa media include i riusi senza caricamento: non misura il tempo del singolo upload cambiato. Test aggiuntivi coprono zeri con segno, subnormali, estremi finiti, Inf e NaN, anche modificando lo stesso buffer CPU tra chiamate.

`LFS_HIP_FAST_RGB_SPECIALIZATION` è opt-in e specializza il backward privato quando depth ed edge sono assenti; normal e statistiche rimangono indipendenti. Non modifica i kernel originali CUDA/Metal/Vulkan. Le combinazioni analitiche raster sono passate da 20 a 72.

| Confronto ripetuto ABBA | Generale, ms/step | RGB, ms/step | Riduzione |
| --- | ---: | ---: | ---: |
| 30 step, scartati i primi 5 | 142,026 | 133,941 | 5,69% |
| 200 step, scartati i primi 10 | 122,370 | 117,871 | 3,68% |

Protocollo: due viste, lato massimo 1920, un milione di gaussiane effettive, SH0, topologia fissa, ottimizzatore fused, quattro esecuzioni per durata. Memoria live circa 713,59 MB e picco 877,03 MB in entrambe le varianti. Il primo confronto lungo ha PSNR 17,681687 contro 17,681559 dB. Le immagini sono state confrontate visivamente; la differenza media è circa 0,04–0,05 livelli su 255, dello stesso ordine delle ripetizioni della stessa variante. Non è parità bit a bit, né una verifica del training completo con crescita o viste holdout. Non confrontare direttamente questi tempi con la baseline Vulkan storica, che usa un protocollo diverso.

Scartato un esperimento con due shard dei gradienti: circa 150,35 ms/step nel run breve e 64 MB aggiuntivi. I sorgenti sperimentali sono stati ripristinati dal manifest prima dell'esperimento.

Verifica finale della configurazione RGB: 25/25 test nativi; 9/9 matrice Tensor (HIP, CUDA+HIP, CUDA nascosta). Artefatti in `.local/rocm-tensor-study`: `rgb-final-native-suite.log`, `selection-final-core-matrix.json`, `rgb-specialization-repeated-200.json`, `sdk-sources/audit/rgb-quality-repeat-context.json`. Le opzioni restano disattivate per default; l'ammissione al Trainer principale resta chiusa.


### Tuning rocPRIM nel training, confronto ripetuto da 200 step

L'opzione `LFS_HIP_FAST_RADIX_TUNING` (default OFF, ammessa solo con target gfx1150) usa configurazioni pubbliche rocPRIM senza patch dell'SDK: onesweep a 8 bit, ranking match, histogram 256x12, sort 256x16, soglia merge zero. Il wrapper privato conserva i DoubleBuffer e aggiorna i selettori soltanto dopo successo; query ed esecuzione usano la stessa configurazione. Le altre architetture e i sorgenti originali restano invariati.

Con RGB e le altre opzioni già attive in entrambe le varianti, confronto ABBA con un milione di gaussiane, due viste a 1920, SH0, fused, 200 step, esclusi i primi 10:

| Variante | Run A, ms/step | Run B, ms/step | Media |
| --- | ---: | ---: | ---: |
| Sort di riferimento | 119,672 | 121,309 | 120,490 |
| Sort configurato | 115,103 | 115,395 | 115,249 |

Riduzione media dell'iterazione: 4,35%. Non sommare percentuali di confronti eseguiti in momenti diversi. Memoria live circa 713,59 contro 714,91 MB, picco 877,03 contro 879,67 MB: il miglioramento comporta circa 1,32 MB live e 2,64 MB di picco aggiuntivi.

MSE finale riferimento 0,0170542657 / 0,0170538768; configurato 0,0170541306 / 0,0170543009. Confronto visivo target/riferimento/configurato/differenza x8 senza degradazioni evidenti; differenza media per pixel RGB tra le prime coppie 0,0417 e 0,0491 livelli su 255, massimo 10/25. Differenze dello stesso ordine delle ripetizioni della stessa implementazione, senza promessa di identità bit a bit. Rimangono i limiti del test a due viste e topologia fissa.

Artefatti: `sdk-sources/audit/radix-training-repeated-200.json`, `radix-training-200-analysis.json`, `radix-training-200-quality.png`. Cinque test mirati di matematica/raster/convergenza/SH superati prima del benchmark.

Verifica finale anche con radix tuning attivo: 25/25 test nativi superati (`sdk-sources/audit/radix-final-native-suite.log`). Generatore idempotente per byte e mtime su 22 file, identici ai sorgenti compilati (`radix-generator-validation.json`). I dati live/picco sopra sono contabilità della sessione, non una misura della VRAM fisica residente. Nessuna modifica ai sorgenti originali FastGS/CUDA e ai kernel condivisi rispetto al checkpoint precedente.


### 19 ottobre: confronto SDK e riserva temporanea limitata

La variante pubblica `rocprim::warp_reduce<float,32,true>` è stata confrontata con la riduzione device-library dopo la fine delle attività concorrenti: media ABBA 122,912 contro 120,767 ms/step, senza vantaggi. Ripristinati esattamente i tre sorgenti dal manifest SHA256 dell'esperimento; conservati i test matematici aggiuntivi che verificano ogni lane con contributi sparsi e negativi. Non è stato modificato il driver e non si adotta questa variante SDK. Evidenze: `sdk-sources/audit/sdk-reduce-clean-training-repeated-200.json` e `sdk-reduce-clean-decision.json`. Il precedente confronto interrotto resta escluso.

Nel runtime privato FastGS, il riuso dei buffer temporanei usava capacità esatta: incrementi piccoli del massimo numero di intersezioni comportavano una nuova allocazione e la liberazione del vecchio buffer grande. La nuova prova `LFS_HIP_FAST_ARENA_GROWTH=1`, disattivata per default, aggiunge headroom soltanto quando un buffer già esistente deve crescere, per richieste di almeno 1 MiB. Riserva massima min(6,25% della richiesta, 8 MiB), soltanto se il budget ammette anche il vecchio buffer ancora vivo. Se non c'è margine contabile sufficiente, alloca la dimensione esatta precedente; non altera il budget e non prealloca un heap massimo. Un fallimento reale hipMalloc continua a propagarsi secondo il contratto esistente: non è una nuova garanzia di successo in condizioni di pressione sulla memoria fisica.

Confronto ABBA, stesso binario e stesso protocollo da 200 step, due viste 1920, un milione di gaussiane, SH0/fused, scartati i primi 10:

| Variante | Run A, ms/step | Run B, ms/step | Media |
| --- | ---: | ---: | ---: |
| Capacità esatta | 122,188 | 121,104 | 121,646 |
| Riserva limitata | 111,522 | 112,646 | 112,084 |

Riduzione media 7,86%. Iterazioni >150 ms: 13+13 nei riferimenti e 0+0 con riserva; p95 166,280/153,934 contro 124,103/133,373 ms. Contabilità sessione live 714,91 contro 727,88 MB (+12,97 MB), picco 879,67 contro 889,79 MB (+10,12 MB). Non sono misure della VRAM fisica residente. Non sommare questi guadagni alle percentuali di esperimenti precedenti condotti in momenti diversi.

MSE finale riferimento 0,0170544695 e 0,0170538663; riserva 0,0170542817 e 0,0170542960. Controllo visivo target/riferimento/riserva/differenza x8 senza degradazioni evidenti; differenza media RGB 0,0434/0,0483 livelli su 255, massimi 13/11. Non è parità bit a bit e non verifica crescita della topologia, holdout o il Trainer principale.

Artefatti: `sdk-sources/audit/arena-growth-training-repeated-200.json`, `arena-growth-analysis.json`, `arena-growth-quality.png`, `arena-growth-targeted-tests.log` (5/5). CUDA, Metal e Vulkan originali non modificati; HIP resta opt-in e non ammesso al Trainer principale.

Verifica finale con riserva attiva: 25/25 test nativi passati (`sdk-sources/audit/arena-growth-final-native-suite.log`). Verificata anche la rimozione completa della prova SDK: tutti i 22 sorgenti generati sono byte per byte uguali al riferimento precedente (`sdk-reduce-removal-parity.json`). Nessun benchmark GPU è rimasto in esecuzione.


### 19 ottobre: riuso dell'adiunto colore nel backward

`LFS_HIP_FAST_FACTOR_ADJOINT` (default OFF) riusa `dot(color, grad_c) + depth * grad_d + dot(normal, grad_n)` nel calcolo di dL/dalpha e nella ricorrenza della trasmittanza. La derivata è algebricamente equivalente, ma cambia l'ordine delle operazioni FP32: non si dichiara identità bit a bit. Nessun fast-math globale, quantizzazione, cambio di layout o memoria aggiuntiva. Modifiche solo nelle copie HIP private; tutti gli altri backend conservano i sorgenti originali.

Con riserva temporanea, RGB e radix già attivi in entrambe le varianti, confronto ABBA da 200 step per run, scartati i primi 10, un milione di gaussiane reali, due viste a 1920, SH0/fused:

| Variante | Run A ms/step | Run B ms/step | Media |
| --- | ---: | ---: | ---: |
| Ricorrenza precedente | 110,040 | 109,269 | 109,655 |
| Adiunto riusato | 106,606 | 105,052 | 105,829 |

Guadagno medio 3,49%; memoria sessione live circa 727,885 MB e picco 889,786 MB invariati. Il codice macchina del kernel RGB None/cull0/senza canali ausiliari contiene 1.398 istruzioni statiche contro 1.426: questa è dimensione del codice, non un conteggio delle istruzioni dinamiche eseguite (`factor-adjoint-isa-analysis.json`, disassemblati conservati).

MSE finali precedenti 0,0170542327 / 0,0170545959, nuovi 0,0170547098 / 0,0170542654. Controllo visivo delle immagini senza degradazioni evidenti. Differenza media RGB 0,0417 e 0,0544 livelli su 255; ripetizioni della stessa variante circa 0,041–0,055. Le differenze puntuali massime non sono usate come prova di equivalenza (fino a 44 nella coppia, 50 fra ripetizioni). I confronti restano limitati a due viste, SH0 e topologia fissa; non provano qualità su holdout o Trainer completo.

Quattro alternative sono state scartate dopo altrettanti confronti ABBA da 200 step: reciproco nativo raffinato con controllo del dominio (112,711 -> 131,902 ms), reciproco nel solo dominio già garantito dal raster (114,949 -> 131,265 ms), riuso del limite dell'ellisse e conica normalizzata (107,744 -> 108,024 ms, nessun vantaggio), riduzione wave shuffle al posto della device library (107,115 -> 131,407 ms). Correttezza e test mirati passavano, ma non bastano a giustificare una regressione prestazionale. Ripristinati i sorgenti dai manifest; i binari e le misure delle prove restano separati. Non applicare queste percentuali ad altri confronti.

L'SDK installato definisce `__fdividef(x,y)` come `x/y`: il nome da solo non prova una divisione approssimata più veloce. Questo ha motivato una prova isolata, non una modifica accettata dell'SDK o del driver. I test matematici aggiuntivi conservati verificano la divisione della trasmittanza su 65.536 input entro un errore relativo 3e-7 e la somma in ogni lane. Il culling precomputato aveva passato 65.536 confronti esatti, compresi i casi alla soglia degenere, ma anche quel codice sperimentale è stato rimosso.

Artefatti in `.local/rocm-tensor-study/sdk-sources/audit`: `backward-optimization-comparison.json`, `factor-adjoint-training-repeated-200.json`, `factor-adjoint-analysis.json`, `factor-adjoint-quality.png`, `backward-final-source-parity.json`. In totale cinque confronti da quattro run di 200 step (4.000 step dataset); tutti con GPU serializzata. Nessuna sostituzione di SDK/driver o modifica dei sorgenti CUDA/Metal/Vulkan.

Verifica finale della configurazione conservata: 25/25 test nativi passati (`sdk-sources/audit/factor-adjoint-final-native-suite.log`), compresi budget/recupero, coesistenza CUDA e condivisione HIP–Vulkan. Generatore verificato idempotente per byte e mtime su 22 file, identici ai sorgenti effettivamente compilati (`factor-adjoint-generator-validation.json`). Tutti i benchmark di questo confronto sono conclusi.


### 19 ottobre: disposizione locale dei sotto-tile e nuovi confronti SDK

La variante con un pixel per thread è stata verificata e scartata: 110,661 -> 130,946 ms/step nel confronto ABBA da quattro run di 200 step. LLVM riporta 80 -> 61 VGPR, gruppi 128 -> 256 thread e zero spill SGPR/VGPR in entrambi i kernel RGB; ridurre i registri non ha quindi prodotto un guadagno. Non si attribuisce automaticamente tutto il costo alle atomiche: non è stato misurato un contatore hardware che lo dimostri. Cinque sorgenti ripristinati dal manifest `one-pixel-source-reference`; nessuna opzione one-pixel conservata nel repository. Metadati e decisione in `one-pixel-reference-backward.json`, `one-pixel-candidate-backward.json`, `one-pixel-decision.json` sotto `sdk-sources/audit`.

Le nuove opzioni private `LFS_HIP_FAST_ADJACENT_PIXELS` e `LFS_HIP_FAST_FORWARD_ADJACENT`, entrambe default OFF, mantengono quattro wave e due pixel per thread. Cambiano l'abbinamento dei sotto-tile da (0,4)/(1,5)/(2,6)/(3,7) a (0,2)/(1,3)/(4,6)/(5,7): ogni coppia copre una regione locale 8x8. Nessun cambio di formato, allocazione o matematica del singolo contributo; la somma dei gradienti può cambiare ordine FP32. Il backward adiacente richiede esplicitamente lo staging wave già disponibile. Verificata anche la copertura geometrica senza duplicati su 289 forme di bordo (0..16 per asse), oltre ai test raster su tile completi/parziali.

Il solo backward aveva un vantaggio medio piccolo (109,314 -> 106,490 ms) con riferimenti variabili; non si usa quel dato da solo per dichiarare una percentuale acquisita. La combinazione forward/backward è stata confrontata direttamente con il riferimento precedente, dopo un run separato di riscaldamento da 60 step, escluso dai dati:

| Variante | Run A ms/step | Run B ms/step | Media |
| --- | ---: | ---: | ---: |
| Coppie distanti precedenti | 107,717 | 106,847 | 107,282 |
| Coppie adiacenti forward/backward | 104,524 | 104,300 | 104,412 |

Protocollo invariato: due viste, lato massimo 1920, un milione di gaussiane effettive, SH0/fused, 200 step per run, primi 10 esclusi. Vantaggio 2,67% della combinazione, non sommare ad altre percentuali storiche. Contabilità sessione invariata: circa 727,885 MB live, 889,786 MB picco. MSE finale riferimento 0,0170541998 / 0,0170546030; combinazione 0,0170540168 / 0,0170537455.

Le immagini iniziali esportate a 8 bit sono identiche; questa non è una prova di identità dei buffer float. Dopo 200 step: differenza media RGB 0,0423/0,0491 livelli su 255, confrontabile con le ripetizioni della stessa variante (circa 0,044–0,052). Massimi locali fino a 45 contro 44 fra ripetizioni del riferimento. Controllo visivo senza peggioramenti evidenti. Rimangono i limiti a due viste, SH0 e topologia fissa: nessuna garanzia estrapolata a holdout o strategie complete.

Confronto SDK aggiuntivo: sette configurazioni rocPRIM onesweep (8 bit, ranking match, histogram 256x12), usando API DoubleBuffer, chiavi uint64/valori uint32, 46 bit e input sintetico non ordinato ripristinato in GPU prima di ogni campione. Il ripristino resta fuori dagli eventi cronometrati. Verificati esattamente ordinamento, corrispondenza degli indici e stabilità con il 20% di chiavi duplicate, per un milione e 6.224.786 elementi. Ogni eseguibile è stato eseguito in ordine diretto e inverso; quattro warmup più venti campioni. Il microbenchmark usa conteggio size_t, mentre il chiamante training usa int: non si trasferiscono automaticamente questi tempi al training.

| Configurazione sort | 1M, ms | 6,225M, ms | Scratch 6,225M byte |
| --- | ---: | ---: | ---: |
| 256x16 attuale | 2,149 | 13,044 | 1.570.816 |
| 128x16 | 2,279 | 13,693 | 3.127.296 |
| 128x24 | 2,729 | 16,382 | 2.089.984 |
| 128x32 | 2,384 | 13,278 | 1.570.816 |
| 256x24 | 2,738 | 16,159 | 1.052.672 |
| 256x32 | 2,345 | 12,923 | 792.576 |
| 512x16 | 2,202 | 13,082 | 792.576 |

Mantenuto 256x16: la variante 256x32 guadagna meno dell'1% sul caso grande ma peggiora sul piccolo; non giustifica un cambio del training. I valori di scratch DoubleBuffer non vanno confusi con quelli dell'API out-of-place del precedente microbenchmark, che richiede ulteriore workspace. Nessuna patch di SDK o driver applicata.

Verifica finale combinazione adiacente: 25/25 test nativi passati, inclusi budget/recupero, coesistenza CUDA e condivisione HIP–Vulkan. Generatore idempotente per byte e mtime su 22 file, identici ai sorgenti compilati. Artefatti in `sdk-sources/audit`: `paired-adjacent-training-repeated-200.json`, `paired-adjacent-analysis.json`, `paired-adjacent-quality.png`, `paired-adjacent-final-native-suite.log`, `paired-adjacent-generator-validation.json`, `adjacent-pixels-layout-verification.json`, `radix-double-repeated.json`, `radix-double-analysis.json`. Sorgenti originali CUDA/Metal/Vulkan invariati; HIP e nuove ottimizzazioni rimangono opt-in.


### 19 ottobre: verifiche aggiuntive sulle riduzioni del backward

Nuovo profilo della configurazione conservata, 30 step, primi cinque esclusi: blending backward 60,55 ms, proiezione backward 5,78 ms, forward 20,39 ms, binning/sort 18,81 ms, loss/gradient 6,11 ms, Adam 1,18 ms. La profilazione sincronizza le fasi: i tempi non sono direttamente confrontabili con il throughput senza profiling e le misure annidate non vanno sommate due volte. Non prova che un componente specifico del driver o le atomiche siano l'unica causa.

Tre nuove varianti private, tutte con cinque test mirati passati, sono state confrontate ciascuna con quattro run ABBA da 200 step dopo warmup separato da 60 step. Stesso dataset/protocollo delle misure precedenti (due viste, 1920, 1M, SH0/fused, primi dieci step esclusi):

| Esperimento | Riferimento ms | Variante ms | Decisione |
| --- | ---: | ---: | --- |
| Riduzione dedicata a due lane | 105.929 | 106.656 | Scartata |
| Culling preliminare solo AABB | 105.462 | 110.863 | Scartata |
| Riduzione nativa anche con una sola lane | 104.269 | 104.787 | Scartata |

Nessuna variante ha prodotto un guadagno materiale ripetibile. Conservati binari e risultati separati; generatore ripristinato byte per byte dal riferimento. Nessuna nuova opzione sperimentale mantenuta. Questi confronti aggiungono 2.400 step misurati, più 180 di warmup. Il riferimento rimane circa 104–106 ms: dimezzarlo è un obiettivo non ancora dimostrato, che richiederebbe ulteriori cambiamenti nell'organizzazione del raster, non soltanto queste scorciatoie. Nessuna modifica SDK/driver o ai sorgenti CUDA/Metal/Vulkan.

Artefatti: `sdk-sources/audit/adjacent-next-profile.json`, `sparse-two-analysis.json`, `bbox-only-analysis.json`, `full-reduce-analysis.json` e rispettivi `paired-*-training-repeated-200.json`/`*-tests.log`. Ogni percentuale vale solo per il proprio confronto alternato. Le prove scartate non richiedono dichiarazioni di equivalenza qualitativa né vengono adottate sulla sola base della MSE.

Verifica finale dopo il ripristino: 25/25 test nativi passati (`after-reduction-final-native-suite.log`); 22 file generati identici a quelli compilati, generatore identico al riferimento e idempotente per byte/mtime (`after-reduction-generator-validation.json`). Tutti i benchmark e le compilazioni di questo ciclo sono terminati.


### 19 ottobre: completamento del riuso degli helper tra fasi HIP

L'audit CUDA/HIP ha trovato due adattamenti incompleti nel gestore della sessione sperimentale: callback della fase forward senza riuso nel backward; helper dimensionati sul totale delle gaussiane anziché sul workset visibile. Il raster e le formule erano portati, ma la gestione delle allocazioni non seguiva queste ottimizzazioni del percorso CUDA.

`LFS_HIP_FAST_PHASE_HELPERS=1` abilita ora il dimensionamento sul workset visibile (una riga reale minima per costruire puntatori validi anche con workset vuoto) e il riuso della più grande allocazione temporanea della fase forward quando sufficiente. La capacità deriva dal Buffer posseduto, senza superare il budget né creare una seconda proprietà. Indici ordinati trattenuti, stato dei tile, visibilità e attributi proiettati persistenti non sono candidati al riuso. Il forward è già completato sulla coda della sessione prima dell'azzeramento degli helper; tutto il preprocessing backward continua a visitare le righe del modello, incluso il decadimento dei momenti delle gaussiane invisibili. Se nessun buffer è sufficiente, rimane un buffer dedicato con sostituzione transazionale e controllo del budget. Default OFF; nessuna modifica dei kernel matematici o dei backend originali.

Il solo riuso con dimensione full-N ha eliminato 64 MB, senza un miglioramento di throughput (106,270 -> 107,526 ms nel suo confronto ABBA); la combinazione con workset compatto è stata poi confrontata separatamente:

| Variante | Primo run ms | Secondo run ms | Media ms |
| --- | ---: | ---: | ---: |
| Riferimento precedente | 111.174 | 111.005 | 111.089 |
| Helper compatti e riuso | 109.973 | 108.997 | 109.485 |

Guadagno 1,44% in questo confronto, non sommare alle percentuali storiche né sostituire la precedente misura da 104,41 ms ottenuta in un'altra serie. Stessi parametri: due viste, massimo 1920, un milione di gaussiane, SH0/fused, quattro run da 200 step con primi dieci esclusi e warmup separato da 60 step. Contabilità allocazioni della sessione: live circa 727,885 -> 663,885 MB, picco 889,786 -> 825,786 MB, risparmio 64.000.000 byte. Non sono misure di residenza fisica VRAM.

Immagini iniziali esportate identiche a 8 bit. Dopo training, differenza media RGB 0,0449/0,0495 livelli su 255; riferimento contro sua ripetizione 0,0478/0,0509. Massimi delle coppie 9/11, ripetizioni del riferimento 12/39. Controllo visivo senza peggioramenti evidenti. Restano i limiti a due viste, topologia fissa e SH0: non prova qualità su holdout o Trainer completo.

Aggiunto test di otto transizioni vuoto/visibile con learning rate zero, per verificare recupero del buffer dedicato, riuso, output vuoto, parametri e salute della sessione. Il test di esaurimento budget ora forza il fallimento a freddo prima delle allocazioni forward, senza assumere che il backward debba allocare separatamente gli helper. Suite completa con riuso attivo: 25/25; test di convergenza, budget e cancellazione anche con opzione assente: passato.

L'audit ha inoltre contato 60 metodi Tensor ancora esplicitamente non implementati: generano errori, non fallback CPU silenziosi. Non sono chiamati nel percorso di raster/training misurato. Rimangono lavoro reale per Trainer/TrainingOps, strategie di crescita/potatura, checkpoint e integrazione loader/UI; non descrivere il backend come integrato completamente.

Artefatti: `sdk-sources/audit/phase-helper-analysis.json`, `compact-phase-helper-analysis.json`, `compact-phase-helper-quality.png`, `compact-phase-helper-tests.log`, `compact-phase-helper-final-native-suite.log`, `compact-phase-helper-default-session.log`, `hip-porting-audit-phase-helpers.json`. Nessuna patch di SDK o driver e nessun push/PR.


### 19 ottobre: compilazione matematica selettiva dei kernel privati

Il confronto delle impostazioni di build ha trovato FastGS CUDA Release con `--use_fast_math`, mentre le unità HIP private usavano O3 e alcune intrinseche esplicite. Non è una prova che l'intero insieme fast-math sia opportuno in HIP. La nuova opzione `LFS_HIP_FAST_APPROX_FUNCTIONS` (default OFF) abilita esclusivamente `-fapprox-func -freciprocal-math -ffp-contract=fast` sulle due unità FastGS e sui relativi probe matematici/raster. Nessun `-ffast-math`, nessuna assunzione generale di assenza di NaN/Inf, nessuna modifica del denormal mode o dei flag degli altri target. I controlli della sessione rimangono compilati separatamente con le impostazioni precedenti. Un probe con lo stesso SDK produce IR con i controlli `fcmp uno`/confronto infinito ancora presenti e divisione `arcp contract afn`, senza `nnan`, `ninf` o riassociazione generale (`selective-math-flags.ll`). Questo controlla le assunzioni del compilatore, non dimostra identità numerica delle funzioni approssimate.

Con riuso compatto attivo in entrambe le varianti, quattro run ABBA per ciascuna durata, warmup separato di 60 step e primi dieci step esclusi:

| Durata | Riferimento ms | Variante ms | Riduzione |
| --- | ---: | ---: | ---: |
| 200 step | 107.984 | 103.471 | 4.18% |
| 500 step | 99.856 | 97.193 | 2.67% |

Le durate restano confronti distinti: il costo cambia con l'evoluzione del modello e non si attribuisce il passaggio 103 -> 97 ms interamente a un'ulteriore ottimizzazione. Stesso dataset, 1M gaussiane, due viste, 1920, SH0/fused; nessuna modifica di precisione dei parametri o dei codec dell'ottimizzatore. Contabilità live/picco circa 663,885/825,786 MB, sostanzialmente invariata rispetto alla variante compatta.

A 200 step MSE finali riferimento 0,0170544740 / 0,0170541809, nuove 0,0170538446 / 0,0170540807. A 500 step riferimento 0,0104700122 / 0,0104654174, nuove 0,0104633190 / 0,0104664847. I valori di ogni run rimangono nei JSON grezzi. Differenze medie RGB tra varianti dopo 500 step 0,7788/0,7142 livelli su 255; fra ripetizioni del riferimento 0,7355/0,7070, della candidata 0,7118/0,7021. Massimi locali rispettivamente 70/87, 81/80 e 89/87: nessuna pretesa di equivalenza bit per bit o garanzia su ogni pixel. Anche le immagini iniziali cambiano leggermente: massimo un livello su 255, media circa 0,000013–0,000015. Controllo visivo delle quattro immagini per vista senza peggioramenti evidenti. Rimangono i limiti di due viste, SH0, topologia fissa, assenza di holdout e integrazione Trainer non completata.

Cinque test mirati passati (comprese 72 combinazioni raw raster/gradienti, matematica e convergenza FP32/fused SH0–SH3). Suite nativa completa: 25/25, compresi condivisione del modello HIP–Vulkan e coesistenza CUDA. Aggiunto poi un test dedicato con 10.007 gaussiane visibili nello stesso tile e immagine 1x1: verifica che lo scratch insufficiente richieda helper dedicati, che il rifiuto per budget conservi tutte e cinque le famiglie dei parametri e il contatore, e che ripristino del budget e successivo riuso senza margine funzionino. Il test ampliato passa con riuso sia attivo sia disattivo. Nessuna modifica produttiva successiva alla suite completa: solo esplicitazione dell'inizializzazione delle liste CMake e commenti.

Metadati SDK del kernel RGB: VGPR 80 -> 79, SGPR 46 -> 43; zero spill SGPR/VGPR, gruppi da 128 e wave32 invariati. Questo non prova un aumento di occupancy né attribuisce da solo il guadagno a una specifica risorsa hardware. Nessuna patch di SDK/driver o ricompilazione del compilatore.

Artefatti sotto `sdk-sources/audit`: `approx-functions-analysis.json`, `approx-functions-long-analysis.json`, relative immagini `*-quality.png`, `approx-functions-tests.log`, `approx-functions-final-native-suite.log`, `phase-helper-dedicated-fallback.log`, `phase-helper-dedicated-default.log`, `selective-math-flags.ll`, `approx-functions-{reference,candidate}-backward.json`. Build locale di studio: `build-probes-approx-functions.cmd`; il precedente script adiacente forza ora esplicitamente l'opzione matematica OFF, così non eredita accidentalmente il valore ON dalla cache CMake. Entrambe le nuove ottimizzazioni rimangono opt-in e locali, nessun push/PR.


### 19 ottobre: aggregazione per tile, SSIM e buffer esclusivi del raster

Tre famiglie di esperimenti isolate non sono state adottate:

| Esperimento | Riferimento ms | Variante ms | Esito |
| --- | ---: | ---: | --- |
| Gradienti aggregati tra quattro wave | 97.931 | 102.730 | Scartato |
| SSIM con canali su blocchi indipendenti | 99.578 | 100.605 | Scartato |

Ogni confronto training usa quattro run ABBA da 200 step, primi dieci esclusi e warmup separato da 60 step, con 1M gaussiane, due viste, 1920 e SH0/fused. L'aggregazione dei gradienti aveva cinque test mirati passati e manteneva il clipping per wave prima della somma, ma introduceva sincronizzazioni e non giustificava il costo. Anche SSIM parallela passava i quattro test mirati, incluso l'oracolo CPU, senza migliorare il training. La variabilità osservata non viene trasformata in un guadagno.

Probe SSIM indipendenti su immagini sintetiche CHW 1440x1920: quattro warmup e venti campioni con eventi GPU, in ordine alternato. Con canali indipendenti forward/backward medi 3,468/3,456 ms contro 3,051/3,270 ms del riferimento. Provati anche blocchi larghi 32x16, con canali seriali/paralleli, e una disposizione swizzled XOR della memoria condivisa. Nessun vantaggio convincente: il layout largo seriale resta vicino al rumore (6,484 ms complessivi contro 6,660 del proprio riferimento, con deriva nei due run), gli altri sono più lenti. Nessun contatore hardware ha provato che i conflitti LDS siano il collo di bottiglia. In tutti i probe, tutti i partial FP16 e gradienti FP32 sono identici byte per byte fra riferimento e varianti; ciò non basta a giustificare un cambio senza un guadagno misurato. Queste varianti restano solo negli artefatti di studio. Sorgenti, template, CMake e test photometrici ripristinati; nessuna nuova opzione SSIM nel repository.

Il secondo audit ha invece individuato sette buffer privati usati soltanto da 3DGUT che restavano allocati scegliendo FastGS: color (12N byte), radii (8N), means2d (8N), depths (4N), conics (12N), counts (4N), prefix (8N). La nuova variabile `LFS_HIP_FAST_TRIM_UNUSED=1`, default assente/OFF, li rilascia quando si seleziona esplicitamente FastGS prima di associare un frame. Parametri condivisi/importati, gradienti dell'ottimizzatore e scratch della scan rimangono separati; nessun rilascio di memoria di proprietà del Tensor/renderer.

Tornando a 3DGUT prima del frame, i sette sostituti sono costruiti completamente sotto lo stesso budget prima del commit. Un fallimento restituisce ResourceExhausted lasciando FastGS e la contabilità precedente intatti. Il preprocessing 3DGUT ha anche una guardia che ricostruisce i buffer se mancanti. Il parametro color di Adam riceve nullptr solo su FastGS, dove il ramo che potrebbe leggerlo è già disabilitato. Default e kernel degli altri backend non cambiano.

Confronto ABBA da 200 step del solo rilascio:

Riferimento 101.198 -> variante 100.553 ms, circa 0.64%. Non viene dichiarata un'accelerazione significativa da una differenza inferiore all'1%. Il beneficio verificato è di 56.000.000 byte in meno, live circa 663,885 -> 607,885 MB e picco 825,786 -> 769,786 MB. Sommati al precedente riuso degli helper: 120 MB di allocazioni in meno rispetto a prima delle due modifiche di memoria; non sono misure della residenza fisica VRAM.


Le immagini iniziali esportate sono identiche a 8 bit. Dopo 200 step differenza media RGB 0,0467/0,0494 livelli su 255, contro 0,0468/0,0504 tra ripetizioni del riferimento e 0,0427/0,0540 tra ripetizioni della variante. Massimi locali rispettivamente 49/42, 8/45 e 49/48: non si dichiara identità del singolo pixel. Controllo visivo senza peggioramenti evidenti. Restano i limiti del test su due viste, SH0 e topologia fissa.

Test aggiunto: verifica il rilascio esatto di 56 byte per gaussiana, il fallimento immediato e a ricostruzione parziale per budget insufficiente, il rollback della contabilità, il successivo ripristino e un vero step 3DGUT. Suite completa con rilascio e riuso helper attivi: 25/25 passati; sessione predefinita senza entrambe le variabili: passata. Dopo l'aggiunta del caso di ricostruzione parziale viene rieseguito il test di sessione in entrambe le modalità. Nessun kernel produttivo cambiato dopo la suite completa.

Artefatti sotto `sdk-sources/audit`: `block-aggregate-analysis.json`, `photo-channels-analysis.json`, `photo-{channels,wide,swizzle}-probe-{results,parity}.json`, `block-photo-experiments-restoration.json`, `block-photo-final-generator-validation.json`, `trim-geometry-analysis.json`, `trim-geometry-quality.png`, `trim-geometry-tests.log`, `trim-geometry-final-native-suite.log`, `trim-geometry-default-session.log`, `trim-geometry-partial-rollback.log`, `trim-geometry-partial-default.log`. Verificati 22 file FastGS identici a quelli compilati e idempotenti per byte/mtime, più identità del sorgente photometric generato e assenza delle varianti scartate. Tutto locale e opt-in, nessun push/PR né modifica di SDK o driver.

Esito finale: anche il test ampliato di rollback parziale passa con entrambe le modalità. Benchmark, probe, compilazioni e verifiche di questo ciclo sono conclusi.

### 19 ottobre: backward RGB a quattro pixel con riduzione protetta

La nuova opzione privata `LFS_HIP_FAST_FOUR_PIXELS`, default OFF, usa 64 thread (due wave32) e quattro pixel per thread nel backward RGB. Richiede staging indipendente, disposizione adiacente e scatter delle riduzioni; CMake rifiuta combinazioni incomplete. Depth, normali, statistiche di densificazione/edge e modalità uncull di riferimento mantengono il kernel precedente. Guardie statiche vincolano il kernel a tile 16x16 e alla specializzazione RGB senza canali ausiliari. Forward, optimizer, codec dei parametri e sorgenti CUDA/Metal/Vulkan non cambiano.

La prima variante manteneva sempre due riduzioni separate: 98,140 -> 98,924 ms, senza guadagno, e non è stata adottata. La variante conservata unisce le riduzioni solo quando ogni lane dimostra che la somma assoluta di tutti i campi delle due vecchie quad è inferiore a clamp/64. Ogni somma originale di 32 lane resta sotto clamp/2, con ampio margine di arrotondamento: il clipping è allora un'identità. Altrimenti si riducono e limitano separatamente le due quad originali 8x8 prima di sommarle. NaN/Inf rendono falso il confronto; nessuna assunzione finite-only. Questo evita la cancellazione di gradienti saturi di segno opposto prima del clipping. L'ordine delle somme FP32 può cambiare: nessuna promessa di identità bit per bit.

Confronti ABBA, warmup separato da 60 step, primi dieci step esclusi, nessuna compilazione durante le misure. Stesso dataset statua, 1M gaussiane, due viste, lato lungo 1920, SH0, fused Adam, budget 2 GiB, identici flag di memoria:

| Durata per run | Riferimento ms | Variante ms | Riduzione |
| --- | ---: | ---: | ---: |
| 200 step | 98,556 | 92,788 | 5,85% |
| 500 step | 92,318 | 87,213 | 5,53% |

Le durate restano protocolli distinti; non si sommano questi guadagni alle percentuali storiche. Le singole medie a 500 step sono 92,905/91,731 ms riferimento e 86,713/87,713 ms candidata. Contabilità sostanzialmente invariata: circa 607,885 MB live, 769,786 MB picco, più target CPU da 66,355 MB. Non sono misure di residenza fisica VRAM.

MSE finali a 500 step: riferimento 0,0104620315 / 0,0104713229, nuova 0,0104674607 / 0,0104666952. Differenze medie RGB fra varianti 0,7239/0,7128 livelli su 255; fra ripetizioni del riferimento 0,6902/0,6995, della candidata 0,7576/0,7172. Massimi locali 77/89 contro 97/89 e 96/93. Ispezionate entrambe le viste e crop a risoluzione nativa, senza peggioramenti evidenti. A 200 step differenza media circa 0,048 livelli, comparabile alle ripetizioni. Restano due viste, SH0, topologia fissa e nessun holdout: non è una garanzia per ogni dataset né una misura comparabile alla vecchia baseline Vulkan.

Suite completa sulla variante: 25/25, incluse convergenza FP32/fused SH0–SH3, budget/cancellazione, memoria HIP–Vulkan e coesistenza CUDA. Aggiunte poi 24 coppie ottimizzato/riferimento (48 invocazioni) al probe raster: due larghezze parziali, gradienti piccoli/intermedi/saturanti positivi e negativi, anche con segni opposti fra quad e canali. Confrontati 16 helper più 14 gradienti dei parametri; si controlla anche che il caso saturante raggiunga davvero tale regime. Tutte le coppie passano insieme ai 72 casi analitici originali. Ricompilata l'opzione OFF: altri cinque test mirati passati, compresi clipping e convergenza. Nel relativo oggetto il simbolo del nuovo kernel è assente; il vecchio conserva 79 VGPR/43 SGPR, gruppo 128, zero spill.

Metadati SDK della variante: 107 VGPR, 55 SGPR, wave32, gruppo massimo 64, zero spill e scratch privato zero. I maggiori registri non provano un incremento di occupancy. Nessuna patch di driver/SDK. Il readback temporaneo dei bounding box proiettati è stato rimosso integralmente prima dei benchmark: quei box sono limiti superiori, non conteggi di contributi accettati.

Artefatti sotto `sdk-sources/audit`: `four-pixel-guarded-{200,500}-analysis.json`, immagini `*-quality.png`, `four-pixel-gated-all-tests.log`, `four-pixel-mixed-clipping-tests.log`, `four-pixel-option-off-tests.log`, `four-pixel-guarded-backward.json`, `four-pixel-layout-proof.json` e `four-pixel-generator-validation.json`. Provati 256 pixel unici, 289 forme di bordo e conservazione delle quad originali. I 22 file generati sono idempotenti per byte/mtime e corrispondono ai compilati. Build di studio `build-probes-approx-functions.cmd`; lo script precedente forza la nuova opzione OFF per non ereditarla dalla cache. Tutto locale, nessun push/PR, ammissione al Trainer principale ancora chiusa.

Verifica conclusiva: anche dopo le guardie statiche e i test con segni misti, il raster e la sessione con variabili di memoria assenti passano (2/2). Build ottimizzata ripristinata con opzione ON; generazione finale verificata nuovamente su 22 file. Tre combinazioni CMake incomplete respinte come previsto. Artefatti aggiuntivi: `four-pixel-final-default-session-tests.log`, `four-pixel-config-guards.json`, `four-pixel-option-off-symbol-check.json`. Nessun benchmark o compilazione di questo ciclo rimane attivo.


### 19 ottobre: letture uniformi dei payload nel backward HIP

Aggiunta l'opzione privata `LFS_HIP_FAST_UNIFORM_BROADCAST`, default OFF. Richiede native wave e staging indipendente, con rifiuto CMake delle combinazioni incomplete. Soltanto le letture dei payload dal batch, con indice sorgente `k` uniforme per l'intera wave32 attiva, usano `__builtin_amdgcn_readlane`. Il generico shuffle, i percorsi con indice variabile per lane, le riduzioni e il clipping non cambiano. Le componenti FP32 viaggiano come bit unsigned, senza conversioni numeriche. Il batch conserva l'ordine dei contributi accettati. Anche il kernel precedente a due pixel usa la nuova lettura quando esplicitamente abilitata, inclusi i suoi canali ausiliari. Nessuna nuova allocazione, modifica dell'optimizer, del codec dei parametri o dell'ammissione al Trainer principale. Nessuna modifica di CUDA, Metal o Vulkan.

Confronti ABBA, warmup separato 60 step, primi dieci esclusi, nessuna compilazione concorrente. Statua, 1M gaussiane, due viste, 1920, SH0/fused Adam, budget 2 GiB, stessi flag di memoria. Il riferimento è la precedente variante a quattro pixel con guardia a somma assoluta, ricampionata nello stesso confronto:

| Durata per run | Riferimento ms | Variante ms | Riduzione |
| --- | ---: | ---: | ---: |
| 200 step | 92,806 | 90,901 | 2,05% |
| 500 step | 87,031 | 85,428 | 1,84% |

A 500 step riferimento 87,0077/87,0537 ms, candidata 85,4086/85,4467 ms. Le durate restano protocolli distinti e le percentuali non si sommano alle precedenti. Memoria circa 607,885 MB live, 769,786 MB picco, target CPU 66,355 MB, invariata entro le piccole differenze dei conteggi visibili. Nessuna misura di residenza fisica VRAM. I tempi restano del probe di training a topologia fissa e non del Trainer applicativo completo.

MSE finali a 500 step: riferimento 0,0104650310 / 0,0104621376, candidata 0,0104653568 / 0,0104718077. La media della candidata è circa 0,048% superiore in questo piccolo campione: non si dichiara un miglioramento qualitativo né equivalenza esatta. A 200 step riferimento 0,0170539867 / 0,0170538349, candidata 0,0170534138 / 0,0170542992. I quattro export iniziali (due viste per ciascuna durata) sono identici byte per byte. Dopo 500 step differenze medie RGB fra varianti 0,7844/0,7245 livelli su 255, fra ripetizioni del riferimento 0,7590/0,6899 e della candidata 0,7331/0,7308. Massimi locali rispettivamente 79/73, 78/78 e 65/82. Ispezionate entrambe le viste e crop nativi senza peggioramenti evidenti. Restano i limiti di due viste, SH0, topologia fissa e nessun holdout; gli atomici non deterministici impediscono di presumere identità del training dal solo trasporto esatto dei payload.

Il test matematico verifica esattamente i bit float e unsigned per tutte le 32 lane sorgenti e blocchi da 32/64/128/512 thread, con indici distinti per wave nello stesso blocco. Copre finite, subnormali, estremi, zeri con segno, infiniti e payload NaN: 23.552 risultati float e altrettanti unsigned. Passa sia con la variante attiva sia in un eseguibile separato senza alcun flag di ottimizzazione matematica/native wave. Suite completa 25/25, inclusi i 72 casi analitici raster, le 24 coppie di gradienti/clipping, convergenza FP32/fused SH0–SH3, budget/cancellazione, HIP–Vulkan e coesistenza CUDA. Ricompilata anche l'opzione OFF: cinque test mirati passati.

Nel kernel RGB a quattro pixel le istruzioni statiche `ds_bpermute_b32` passano da 47 a 35, con 12 `v_readlane_b32`; non sono conteggi dinamici. VGPR 107 -> 106, SGPR 55 -> 63, zero spill e scratch privato zero. Questi numeri non dimostrano un incremento di occupancy; il guadagno dichiarato proviene dai confronti di training.

Due ulteriori studi non sono stati adottati:

| Esperimento | Riferimento ms | Variante ms | Esito |
| --- | ---: | ---: | --- |
| Guardia del clipping con limite IEEE a OR di bit | 92,549 | 92,699 | Nessun guadagno |
| Radix onesweep a 10 bit, istogramma con due stripe | 92,486 | 91,676 | Meno dell'1%, non giustifica la dipendenza sperimentale |

Entrambi hanno cinque test mirati passati e quattro run da 200 step. La guardia IEEE ha anche 100.000 vettori di verifica del limite, inclusi NaN/Inf, ma il sorgente conservato usa la precedente somma assoluta. Il radix 10 bit standard non compila: l'istogramma richiede 114.688 byte LDS oltre il limite 65.536; cambiare soltanto il block size non risolve. Nella copia isolata dei soli header rocPRIM si è reso configurabile `atomic_stripes` dell'helper onesweep, senza modificare l'SDK installato. Probe con conteggi int32, chiavi FP32 complete e 20% di pareggi: valori/indici e stabilità esatti a 1M e 6.224.786 elementi. Su quest'ultima taglia mediane ripetute 12,872 ms per 8 bit/4 stripe, 12,032 per 10 bit/2 stripe, 11,797 per 10 bit/1 stripe; sul milione di elementi 2 stripe era migliore di 1. Il vantaggio isolato del sort non si trasferisce interamente al training. La variante a due stripe aggiungeva circa 5,08 MB live e 10,16 MB di picco rispetto al riferimento, quindi è stata esclusa dal codice finale. Tutte le modifiche rocPRIM restano nei soli artefatti, il radix produttivo privato è tornato a 8 bit e l'hash dell'header SDK installato è invariato. Nessuna patch di driver o dipendenza da percorsi di studio nel repository.

Artefatti sotto `sdk-sources/audit`: `paired-uniform-broadcast-{200,long-500}-analysis.json`, immagini di qualità, `uniform-broadcast-tests.log`, `uniform-full-native-suite.log`, `uniform-default-math-test.log`, `uniform-option-off-tests.log`, `uniform-broadcast-backward.json`, `uniform-broadcast-instructions.json`, `uniform-initial-image-parity.json`, `uniform-config-and-sdk-guards.json`, `four-pixel-bit-bound-analysis.json`, `radix-bits-analysis.json`, `radix-stripes-sdk-patch.json` e `wide-radix-training-analysis.json`. I JSON mantengono tutti i run, compresi quelli senza beneficio.

Build di studio aggiornata: `build-probes-uniform-broadcast.cmd`. Gli script precedenti mantenuti forzano la nuova opzione OFF per evitare eredità accidentali dalla cache CMake. Tutto locale e opt-in, nessun push/PR.

Verifica conclusiva: ricompilata e ripristinata la variante ottimizzata, passano anche matematica, raster e convergenza con le variabili di memoria assenti (3/3). Verificati nuovamente 22 file generati idempotenti per byte/mtime e identici ai compilati, guardia assoluta e radix 8 bit ripristinati, sorgente training privo del readback diagnostico. Artefatti finali `uniform-final-default-session-tests.log` e `uniform-final-generator-validation.json`. Nessun training, probe o compilazione di questo ciclo rimane attivo.


### 19 ottobre: tile RGB intero per wave nel forward e backward

Aggiunte due opzioni private, entrambe default OFF: `LFS_HIP_FAST_EIGHT_PIXELS` e `LFS_HIP_FAST_FORWARD_EIGHT_PIXELS`. Un gruppo da 32 thread copre tutti i 256 pixel del tile 16x16, con otto pixel per thread. Il backward richiede la disposizione/clipping già usata dal percorso a quattro pixel; il forward richiede le letture uniformi native. CMake rifiuta combinazioni incomplete. Le due opzioni sono indipendenti e sono state provate anche separatamente.

Nel backward rimangono quattro accumulatori logici, uno per ciascuna quad originale 8x8: pixel (0,2), (1,3), (4,6), (5,7). Il limite assoluto include tutti i campi delle quattro quad. Solo se ogni lane resta sotto clamp/64 si possono unire le riduzioni; altrimenti ogni quad viene ridotta e limitata separatamente prima della somma. Non si cancella un gradiente saturo prima del suo clipping. Il prefisso inverso comune del tile può essere più lungo di quello delle precedenti mezze tile, ma le guardie `last` conservano la sequenza dei contributi accettati di ciascun pixel. La modifica è confinata al backward RGB privo di canali ausiliari.

Nel forward, RGB senza profondità/normali usa batch da 32 primitive in registri privati. Ogni lane carica una primitiva e le letture uniformi vengono eseguite prima di qualsiasi `continue` divergente dei pixel: la lane sorgente rimane attiva anche se i suoi pixel sono fuori immagine o già terminati. La sincronizzazione di blocco è sostituita dal ballot dell'unica wave; non rimane staging LDS. Le modalità di test con culling disattivato/negativo, canali ausiliari o batch esplicitamente richiesti diversi da 32 mantengono il vecchio forward. Il batch predefinito può cambiare senza cambiare l'ordine di blending di ciascun pixel. Nessuna nuova allocazione e nessuna modifica al layout dei parametri, all'optimizer o ai kernel CUDA/Metal/Vulkan.

#### Confronto principale pulito

Statua, 1M gaussiane, due viste, lato lungo 1920, SH0/fused Adam, budget 2 GiB, shuffle 42, stessi flag di memoria. Quattro run ABBA da 500 step, primi dieci esclusi, warmup separato da 300 step. Nessuna compilazione, query dei contatori GPU o analisi immagini durante questa serie. Il riferimento è la precedente build a quattro pixel con letture uniformi, non una media storica ricostruita:

| Configurazione | Run 1 ms | Run 2 ms | Media ms |
| --- | ---: | ---: | ---: |
| Riferimento | 84,5179 | 84,8220 | 84,6700 |
| Tile intero | 80,4164 | 80,4282 | 80,4223 |

Riduzione misurata **5,02%**. I p95 sono circa 97,11/97,05 ms per il riferimento e 91,57/91,37 ms per la variante. Nessuno dei quattro run contiene step oltre 150 ms. Live/picco circa 607,885/769,786 MB, target CPU 66,355 MB, senza nuove allocazioni. Non sono misure di residenza fisica VRAM né tempi del Trainer applicativo completo: rimane il probe a topologia fissa.

La prima serie combinata da 200 step è stata mantenuta nei dati ma esclusa come confronto principale: riferimento 90,884/95,543 ms, candidata 85,617/91,792 ms; tre step della seconda candidata superavano 150 ms, con massimo 403,295 ms. Il carico in istanze era quasi identico. Il controllo dei processi non ha mostrato altre build o sessioni LichtFeld, e la RAM disponibile era abbondante; non si attribuisce la deriva a una causa hardware provata. Le query WMI dei contatori erano in sola lettura. Anche la successiva serie esplorativa da 500 step, con diagnostica durante parte dell'esecuzione, resta distinta: 87,181 -> 83,215 ms. Non si selezionano singoli campioni veloci né si sommano le percentuali delle varianti isolate.

#### Esperimenti isolati e risorse

Tutti i confronti seguenti hanno quattro run da 200 step e warmup separato da 60 step; i loro tempi non sono confrontabili direttamente con la serie da 500:

| Variante isolata | Riferimento ms | Variante ms | Decisione |
| --- | ---: | ---: | --- |
| Backward otto pixel, vincolo minimo 8 gruppi | 90,824 | 88,715 | Superato: 23 spill, 80 byte privati |
| Backward otto pixel, vincolo minimo 4 gruppi | 90,940 | 88,721 | Conservato: zero spill |
| Forward otto pixel, staging condiviso da 128 | 91,260 | 92,170 | Scartato |
| Forward otto pixel, staging in registri da 32 | 90,788 | 88,735 | Conservato |

La variante backward senza spill usa 154 VGPR/71 SGPR, il forward 87 VGPR/96 SGPR. Entrambi hanno wave32, gruppo massimo 32, scratch privato zero e zero spill; il forward ha anche LDS zero. La prima variante backward aveva il limite di registri indotto dal vincolo più aggressivo. Eliminare gli spill non ha prodotto da solo un ulteriore guadagno dimostrato rispetto a quella variante: è stato scelto il percorso senza scratch a prestazioni comparabili. Queste risorse statiche non provano da sole un incremento di occupancy.

#### Qualità e regressioni

Nel confronto pulito, MSE finali riferimento 0,0104660488 / 0,0104637243, candidata 0,0104667142 / 0,0104674683. Nella serie esplorativa da 500 step erano invece riferimento 0,0104687815 / 0,0104684523 e candidata 0,0104680233 / 0,0104644252. Su tutti e quattro i run di ciascun tipo la media è rispettivamente 0,0104667517 e 0,0104666577: nessuna pretesa di miglioramento qualitativo o equivalenza esatta.

Le coppie di export iniziali controllate nelle due serie da 500 step sono identiche byte per byte nei PPM a 8 bit. Dopo il training pulito, differenze medie RGB fra varianti 0,7474/0,7165 livelli su 255, fra ripetizioni del riferimento 0,7219/0,7344 e della candidata 0,7370/0,7316. Massimi locali 60/78 contro 77/83 e 82/94. Ispezionate le due viste e crop nativi senza peggioramenti evidenti. Anche il controllo visivo della serie esplorativa è coerente. Restano due viste, SH0, topologia fissa e nessun holdout; l'ordine delle somme atomiche e delle riduzioni FP32 non è deterministico.

Il test raster include ora quattro disposizioni dei segni (uniforme, X, Y, diagonale), due larghezze parziali e sei scale positive/negative: **48 coppie** riferimento/ottimizzato, oltre ai 72 casi analitici originali. Confronta 16 helper più 14 gradienti dei parametri e verifica che i casi saturanti raggiungano davvero quel regime. Prova geometrica su 256 pixel unici e 289 forme di bordo, conservando tutte le quad di clipping. Suite nativa completa: **25/25**, compresi convergenza FP32/fused SH0–SH3, caso denso 1x1, budget/rollback, cancellazione, alias dei parametri, HIP–Vulkan e coesistenza CUDA. Ricompilate entrambe le opzioni OFF: cinque test mirati passati. Ripristinate ON: matematica, raster e convergenza con le variabili di memoria assenti passano (3/3).

Verificati 22 file generati idempotenti per byte/mtime e identici ai compilati. Il corpo generato del precedente backward a quattro pixel e quello del forward originale sono ancora identici ai rispettivi riferimenti; il sorgente host `training.hip` è invariato e privo del vecchio readback diagnostico. Gli altri backend non sono stati modificati da questo ciclo. Nessuna patch di SDK o driver, nessun cambiamento dell'ammissione al Trainer principale.

Artefatti sotto `sdk-sources/audit`: `paired-fulltile-clean-500-analysis.json`, relativa immagine `*-quality.png`, `fulltile-clean-variability.json`, `fulltile-quality-all-500.json`, `fulltile-initial-image-parity.json`, `fulltile-combined-200-variability.json`, `eight-pixel-{spill,relaxed}-analysis.json`, `forward-eight-{shared128,register}-analysis.json`, `fulltile-native-suite.log`, `fulltile-options-off-tests.log`, `fulltile-default-session-tests.log`, `fulltile-config-guards.json`, `fulltile-final-{forward,backward}.json`, `eight-pixel-layout-proof.json`, `fulltile-final-generator-validation.json`. Conservati anche sorgenti e binari degli esperimenti scartati.

Build locale corrente: `build-probes-fulltile-combined.cmd`. Gli script precedenti mantenuti forzano esplicitamente le nuove opzioni OFF; quello del solo backward forza il nuovo forward OFF. Tutto locale e opt-in, nessun push/PR. Nessun benchmark, query dei contatori o compilazione di questo ciclo rimane in esecuzione.

Possibile prossimo studio, non implementato né misurato: i payload proiettati sono ancora tre array separati da 16 byte (`PackedMeanBBox`, conica/opacità, colore). Un layout privato interleavato da 64 byte potrebbe ridurre letture sparse, al costo teorico di circa 16 byte per elemento visibile più allineamento. Andrebbero preservati gli indici e i pareggi del sort, il layout dei Tensor condivisi e tutti i consumatori, compreso il colore nel preprocess backward. Prima di giudicarlo servono test di stride/padding su più righe e dimensioni non multiple, budget/rollback, confronto numerico multi-gaussiana e misure contro la build a tile intero; non si presume che sia più veloce.


## 2026-10-19 — Layout interlacciato dei dati proiettati: esperimento scartato

Verifica successiva alla versione accettata con forward/backward RGB a otto pixel per thread. Solo generazione privata HIP: nessuna modifica ai sorgenti CUDA/Metal/Vulkan, al modello o ai parametri di training. Geometria/bbox, conica/opacita e colore accorpati in righe da 64 byte (16 di padding), poi da 48 byte. Ordine, indici, formule e ordinamento delle primitive conservati.

Protocollo: stesso dataset e seed, 1M gaussiane, resize 1920, SH0, due viste 0/41, topologia fissa, 500 step per run, ordine A-B-B-A, primi 10 step esclusi; riscaldamento separato da 300 step per serie. Nessuna compilazione o diagnostica GPU durante le misure. Le misure storiche da 80,42 ms restano valide nel loro confronto; i riferimenti contemporanei qui sono circa 81,4 ms.

| Layout | Riferimento medio | Variante media | Variazione | Decisione |
|---|---:|---:|---:|---|
| 64 byte | 81.411 ms | 83.932 ms | +3.097% | Scartato |
| 48 byte | 81.403 ms | 82.703 ms | +1.597% | Scartato |

Layout 64 byte, run A-B-B-A: 81.300382, 84.207633, 83.656337, 81.521513 ms. MSE finali, stesso ordine: 0.0104637185701, 0.0104617196927, 0.0104665740414, 0.0104646173913.

Layout 48 byte, run A-B-B-A: 81.295482, 82.686732, 82.719399, 81.510491 ms. MSE finali, stesso ordine: 0.010471740809, 0.0104663278135, 0.0104661917585, 0.0104618248888.

La variante da 64 byte passa da circa 607,885/769,786 MB (live/picco) a 622,056/783,958 MB: circa +14,172 MB. La variante da 48 byte mantiene il footprint precedente, entro poche centinaia di byte di variazione. Sono byte contabilizzati dalla sessione, non una misura della residenza fisica VRAM.

Validazione sperimentale 64 byte: suite nativa 25/25 e sessione con PHASE_HELPERS/TRIM_UNUSED/ARENA_GROWTH attivi 1/1. Variante 48 byte: matematica, raster e sessione con ottimizzazioni di memoria 3/3. Aggiunto nel prototipo un controllo GPU con 0, 3 e 257 righe, scritture alternate e confronto byte completo: righe non scritte, padding e buffer confinanti devono restare intatti. Test ripetuto nei due layout.

Qualita: immagini iniziali PPM a 8 bit identiche tra i quattro run di ogni serie sulle due viste. Dopo 500 step, differenza media assoluta riferimento-variante 0,750/0,682 livelli (64 byte) e 0,757/0,745 (48 byte), confrontabile con le differenze fra ripetizioni dello stesso modulo. Ispezionati montaggi di entrambe le viste e ritagli centrali a risoluzione nativa: nessun peggioramento evidente. Questi controlli non dimostrano equivalenza numerica universale; due viste SH0 e topologia fissa.

Esito: nessun nuovo guadagno di prestazioni. Entrambi i prototipi rimossi dal codice del branch ripristinando esattamente i quattro file modificati rispetto agli snapshot di inizio esperimento; conservati sorgenti sperimentali, DLL, test e risultati nella cartella locale rocm-tensor-study. La build ripristina il layout precedente e tutte le ottimizzazioni gia accettate, senza introdurre un nuovo flag nel prodotto.

Artefatti locali: `sdk-sources/audit/payload-final-analysis.json`, `payload48-final-analysis.json`, `payload-quality-comparisons.json`, `payload-quality.png`, `payload48-quality.png`; `payload-native-suite.log`, `payload-optimized-session.log`, `payload48-tests.log`; sorgenti in `payload48-source/`, snapshot `payload64-prepare_fast.py`/`payload64-fast_raster.hip`, DLL `fast-payload64.dll`/`fast-payload48.dll`. Prossimi tentativi dovranno riguardare altri colli di bottiglia: questo accorpamento non e una ottimizzazione utile nel carico misurato.

Verifica finale del ripristino: quattro file di codice byte-identici agli snapshot iniziali, 22 file generati idempotenti (byte e timestamp), cache CMake priva del flag sperimentale e matematica/raster/sessione 3/3 passati con ottimizzazioni di memoria attive.


## 2026-10-19 — Pressione sui registri del backward: quattro esperimenti scartati

Profilo preliminare della versione accettata: 40 step, primi 10 esclusi, 1M gaussiane, resize 1920, SH0, viste 0/41. Con sincronizzazioni diagnostiche attive: projection 3,497 ms, count/scan 0,592 ms, binning/sort 19,948 ms, forward 17,884 ms, gradient clear 0,673 ms, loss/gradient 6,544 ms, backward 50,768 ms, Adam 1,465 ms. Sottofasi backward misurate separatamente: blend 43,140 ms, projection backward 6,143 ms. Questi numeri includono il costo del profiling e non sono una scomposizione dei tempi non profilati da 80-86 ms.

Quattro prototipi confinati alla generazione privata HIP; nessun cambiamento ai sorgenti originali CUDA/Metal/Vulkan, al modello o al media ingest. Tutte le opzioni sperimentali erano opt-in. Nessuna modifica al compilatore o SDK installato.

1. Gradienti RGB immutabili degli otto pixel in LDS, righe per canale/lane senza conflitti di banco e accessi esclusivamente della lane proprietaria: nessuna sincronizzazione hardware richiesta. Il compilatore porta i VGPR da 154 a 178, senza spill, con 3072 byte LDS per blocco.

2. Stesso storage LDS ma letture dopo ammissione del contributo e controllo alpha: 179 VGPR, nessuno spill, 3072 byte LDS. Entrambi i tentativi peggiorano il risultato e sono rimossi.

3. Elaborazione sequenziale dei quattro vecchi domini di clipping: una coppia di pixel per volta, un accumulatore temporaneo e un aggregato dei soli gruppi dimostrati sicuri. Il test conservativo per gruppo usa la somma dei moduli dei campi < clamp/256 in tutte le 32 lane: ogni vecchia riduzione resta sotto clamp/8 e la somma dei quattro gruppi sotto clamp/2. NaN/Inf falliscono il confronto stretto. I gruppi non sicuri sono ridotti e clippati separatamente. Questo cambia il raggruppamento di alcune somme FP32; i test numerici sono quindi indispensabili. I VGPR scendono a 142, senza spill o LDS, ma il tempo peggiora.

4. Hint `__builtin_amdgcn_sched_barrier(0)` fra coppie di pixel, senza cambiare le formule o la riduzione del riferimento. Si tratta di regioni per lo scheduling del compilatore, non di una barriera di sincronizzazione fra thread. Fonte primaria: https://llvm.org/docs/AMDGPUUsage.html e https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td . Rimangono 154 VGPR, nessuno spill o LDS. Il risultato temporale e entro le oscillazioni dei riferimenti, quindi scartato.

Protocollo temporale per ogni prototipo: riscaldamento separato da 60 step, quattro run da 200 step A-B-B-A, primi 10 esclusi, stesso seed e impostazioni del profilo, profiling disattivato. Ogni confronto e seriale, senza compilazioni durante le misure. I valori non vanno confrontati direttamente con il benchmark storico da 500 step, perche lo stato del modello evolve.

| Variante | VGPR | LDS/blocco | Riferimento | Variante | Variazione |
|---|---:|---:|---:|---:|---:|
| LDS eager | 178 | 3072 B | 85.776 ms | 106.352 ms | +23.989% |
| LDS admitted only | 179 | 3072 B | 85.592 ms | 99.241 ms | +15.947% |
| Streamed quads | 142 | 0 B | 85.849 ms | 87.002 ms | +1.343% |
| Scheduler regions | 154 | 0 B | 85.915 ms | 85.738 ms | -0.206% |

Lettura: valori positivi significano piu lenti. Il -0,206% dello scheduling non e un miglioramento convincente: i due riferimenti variano da 85,777 a 86,053 ms, mentre le varianti sono 85,611 e 85,866 ms. Nessun prototipo adottato. Allocazioni globali della sessione sostanzialmente invariate, circa 607,885 MB live e 769,786 MB di picco; il conteggio non misura la residenza fisica VRAM.

Controlli eseguiti su ciascuno dei quattro prototipi: matematica, raster/clipping e FastSession con convergenza e recupero/cancellazione, 3/3 passati, con PHASE_HELPERS/TRIM_UNUSED/ARENA_GROWTH abilitati. I run da 200 step producono immagini e MSE per entrambe le viste; sono esperimenti di screening delle prestazioni, non una prova universale di qualita/convergenza su altri dataset. Nessun confronto lungo aggiuntivo da 500 step, perche non e emerso un vantaggio da validare.

Aggiunto e conservato un test mirato al clipping misto: ciascuno dei quattro domini 8x8 a turno riceve un gradiente enorme, gli altri gradienti piccoli, due larghezze non multiple del tile (17/35), altezza 29, entrambi i segni. Il riferimento senza culling e confrontato con il percorso ottimizzato su tutti i campi di gradiente. Il test controlla anche che sia effettivamente raggiunta la saturazione. Sono 16 nuove coppie, 32 invocazioni; insieme ai casi precedenti il probe raster esegue 200 invocazioni.

Stato finale: helper fast_four_pixels.py e HipFastKernels.cmake ripristinati byte per byte dagli snapshot di inizio esperimento. Tutti gli esperimenti conservati solo come sorgenti/DLL/risultati nella cartella locale rocm-tensor-study. Rimangono nel branch soltanto il nuovo test sul clipping e questo rapporto. Nessuna nuova ottimizzazione del runtime accettata.

Artefatti: `sdk-sources/audit/register-baseline-profile.json`, `register-baseline-backward-subphases.json`, `register-final-comparisons.json`, metadati `register-{lds,lds-lazy,streamed,schedule}-backward.json`; run e log `paired-register-*-screen-*`, `register-*-tests.log`; snapshot `register-*-fast_four_pixels.py` e `register-*-HipFastKernels.cmake`, DLL `fast-register-{lds-eager,lds-lazy,streamed,schedule}.dll`. Il profilo identifica il blend backward come priorita per un prossimo intervento algoritmico; la riduzione dei soli registri non ha dato benefici nel carico misurato.

Conferma finale: build precedente ripristinata; matematica, raster con saturazione mista verificata e sessione 3/3 passati con ottimizzazioni di memoria attive. Due file di codice byte-identici agli snapshot, 22 file generati idempotenti (byte e timestamp), nessun flag sperimentale residuo nella cache CMake. `git diff --check` passa.


## 2026-10-19 — Riduzione distribuita dei gradienti RGB: guadagno confermato

Intervento nella sola generazione privata del backward HIP RGB a otto pixel per thread. I singoli contributi dei pixel, il loro ordine per primitiva, il criterio conservativo di fusione dei quattro vecchi domini e i clamp dei domini non fondibili restano invariati. Con piu di una lane contribuente, una riduzione a farfalla distribuisce fra le lane i 16 campi, riducendo progressivamente i campi mantenuti. La lane finale riceve il campo (lane & 15). Il caso con una sola lane contribuente conserva il codice precedente. Cambia il raggruppamento di somme FP32: non si rivendica identita numerica universale. Nessun passaggio a FP16, nessun nuovo fast-math globale.

Opzione `LFS_HIP_FAST_BUTTERFLY_FIELDS`, default OFF, richiede il backward a otto pixel e la distribuzione delle atomiche gia presenti. Solo la sessione nativa sperimentale a topologia fissa; questa modifica non completa l'integrazione nel Trainer principale. Nessun cambiamento ai backend CUDA, Metal o Vulkan.

Protocollo principale: dataset/seed invariati, 1M gaussiane, resize massimo 1920, SH0, viste 0/41, L1/SSIM 0,2, cache target CPU 66.355.200 byte e budget sessione 2 GiB. Riscaldamento separato da 300 step, quattro run da 500 step A-B-B-A, primi 10 esclusi da ogni run, nessuna compilazione durante le misure, profiling disattivato.

| Misura | Riferimento | Nuova riduzione |
|---|---:|---:|
| Step nativo medio | 80.972107 ms | 75.352106 ms |
| Binding del frame medio | 4.851958 ms | 4.865571 ms |
| Step + binding medio | 85.824065 ms | 80.217677 ms |

Riduzione del tempo dello step: **6,9407%**. Il totale step+binding scende di circa **6,53%**. Il tempo di preparazione iniziale delle immagini e separato da queste misure.

Run da 500 step A-B-B-A: 81.165838, 75.355593, 75.348619, 80.778375 ms. Entrambi i candidati sono vicini a 75,35 ms; nessuno dei 1960 step inclusi supera 150 ms. Lo screening preliminare da quattro run di 200 step (60 di warmup separato) aveva dato 85.995380 -> 79.733520 ms (-7.2816%). I due protocolli non sono confrontabili direttamente come tempi assoluti: cambiano durata ed evoluzione del modello.

Memoria globale allocata invariata: circa 607,885 MB live e 769,786 MB di picco; nessuna nuova allocazione o duplicazione del modello, nessun nuovo LDS. Sono conteggi della sessione, non misure della residenza fisica VRAM. Shader RGB: 154 -> 160 VGPR, 71 -> 75 SGPR, zero spill e zero memoria privata. Il guadagno viene dal lavoro della riduzione; il numero di registri isolato non predice il tempo.

Validazione: suite nativa **25/25** con PHASE_HELPERS/TRIM_UNUSED/ARENA_GROWTH attivi, inclusi convergenza, budget, rollback, cancellazione, SH, alias, modello HIP-Vulkan condiviso e convivenza con CUDA. A opzione disattivata, matematica/raster/FastSession e i due test di alias SH **5/5**. Dopo il ripristino della build migliore, matematica/raster/FastSession senza quelle opzioni ambientali **3/3**.

Nuovo oracolo GPU delle riduzioni: 32 configurazioni (blocchi 32/64/128/512, 8 famiglie di input), 5888 risultati di lane confrontati con somme CPU indipendenti. Include campi distinti, contribuenti sparsi, segni opposti, cancellazioni, decimali, NaN, Inf, subnormali e grandi valori finiti. Verifica tutti i 16 campi e le wave multiple. Il probe raster mantiene i test analitici e di clipping misto e aggiunge due casi 1025x769 con gaussiana isotropa estesa: forzano l'emissione cooperativa con una sola sorgente attiva. Totale finale 202 invocazioni raster, passate sia ON sia OFF. I due casi estesi sono stati aggiunti dopo la suite completa, e poi verificati nuovamente nei controlli finali pertinenti.

Qualita: PPM iniziali a 8 bit byte-identici fra tutti e quattro i run su entrambe le viste. MSE finali A-B-B-A: 0.0104758583845, 0.0104642206253, 0.0104637116595, 0.0104611014709. Media riferimento 0.0104684799277; media candidato 0.0104639661424. Nessuna rivendicazione di miglioramento della qualita.

Differenze medie assolute riferimento-candidato: 0,720865/0,734279 livelli su 255 nelle due viste; riferimento-ripetizione 0,730587/0,729528; candidato-ripetizione 0,706780/0,687503. Ispezionati montaggio delle due viste e ritagli centrali a risoluzione nativa: nessun peggioramento evidente. Limite: due viste SH0, topologia fissa, 500 step, non una verifica holdout o di tutti i dataset.

Prova del perimetro: rispetto al generatore di partenza cambiano solo `hip_fast_wave.hpp` e `kernels_backward.cuh` fra i 22 file generati. Tutti i file forward e i corpi dei backward originali a due/quattro pixel sono byte-identici. A flag OFF anche i 23.284 byte di codice macchina del kernel a otto pixel coincidono con il riferimento (SHA-256 `f2d10c1b35f07cb098a6dcf6024223496f6bcc081ee971bcdc6829dc34922124`).

Altri esperimenti di questo giro:

| Esperimento | Esito sullo step | Decisione |
|---|---:|---|
| Taglio dei test geometrici oltre il prefisso di ogni sottotile | -0.0225% | Scartato |
| Scambi DS_SWIZZLE espliciti fra lane | +4.5350% | Scartato |
| Maschera delle sorgenti attive e soglia collettiva nel binning | +0.5481% | Scartato |

Gli scambi DS_SWIZZLE erano bit-identici nei test rispetto al percorso generico e riducevano i VGPR a 155, ma peggioravano il tempo. Maschere e indici del prototipo cooperativo preservavano ordine delle sorgenti/scritture; anche questo prototipo passa i test, ma non porta un vantaggio nel dataset. Codice e opzioni di questi prototipi rimossi dal branch; rimangono snapshot/DLL/risultati locali e il caso di test esteso.

Verifica separata del lavoro CPU: microbenchmark locale host-only su 33.177.600 byte, stessa classificazione IEEE-754 intera del controllo portabile, quattro ripetizioni in ordine alternato e un prototipo AVX2 con verifica delle feature CPU/OS. Risultato medio 0.566670 ms portabile e 0.481441 ms AVX2; 32896 controlli di offset, dimensioni e rifiuto di Inf/NaN anche nel tail. Non e una misura del binding completo. Il risparmio isolato e circa 0,085 ms e non giustifica ora un nuovo percorso nel runtime: nessuna modifica a training.hip o alle opzioni host. Il resto del binding richiede una misurazione diretta di copia e sincronizzazioni; non si attribuiscono tutti i suoi 4,86 ms alla scansione CPU.

Profilo diagnostico finale: riscaldamento separato e quattro run da 80 step A-B-B-A. La serie mostra deriva anche in fasi byte-identiche: ad esempio il count/scan varia da 0,600 a 1,289 ms. Conservata integralmente, esclusa dall'attribuzione quantitativa dei guadagni per fase. I tempi di profiling includono sincronizzazioni e intervalli host; non sostituiscono il confronto pulito da 500 step. Il riferimento non profilato e stabile e nessuna parte della serie principale e stata scartata selettivamente.

Artefatti: `sdk-sources/audit/butterfly-fields-clean-analysis.json`, `butterfly-fields-screen-analysis.json`, `butterfly-fields-quality.json/.png`, `butterfly-generated-scope.json`, `butterfly-option-off-code-parity.json`, `butterfly-phase-profile-analysis.json`, `host-validation-study.json`; log `butterfly-fields-native-suite.log`, `butterfly-fields-off-tests.log`, `butterfly-fields-final-default-tests.log`; build migliore `build-probes-butterfly-fields.cmd`, DLL di riferimento accettata `fast-butterfly-fields-accepted.dll`. I vecchi script di studio ora forzano esplicitamente OFF il nuovo flag per evitare contaminazione della cache CMake.

Prossime possibilita da studiare, non implementate: costo reale di upload/sincronizzazione dei frame; riduzione esatta dei bit delle chiavi di profondita tramite intervallo intero (offset dal minimo, senza quantizzare), solo se il costo della misura min/max e inferiore al passaggio radix risparmiato e preservando ordinamento stabile, fallback e budget. Projection, scan, loss e Adam mantengono il codice precedente: non si introducono cambiamenti senza un vantaggio verificato.

Verifica finale: quattro sorgenti della variante accettata preservati byte per byte; training.hip invariato; 22 file generati idempotenti per byte/timestamp; flag accettato ON nella build locale e default OFF nel CMake; flag dei prototipi rimossi dalla cache. Suite 25/25, OFF 5/5, configurazione finale senza opzioni ambientali 3/3 confermate dai log.


## 2026-10-19 — Screening successivo e memoria host del frame

Tre prototipi isolati scartati: compressione esatta delle chiavi di profondita con conversioni separate (-0,0574%, rumore), stessa compressione fusa in emissione/estrazione (+1,8284% tempo), riduzione dei soli nove campi RGB (nessun vantaggio iniziale, deriva del riferimento finale; non quantificabile come piccolo guadagno). Ogni build supera matematica/raster/FastSession 3/3; oracoli aggiuntivi per ordine stabile delle chiavi e somme dei nove campi. Nessuna quantizzazione. Codice/opzioni dei prototipi rimossi; sorgenti e runtime ripristinati byte-identici alla variante accettata, artefatti conservati.

Misura host isolata: 33.177.600 byte; RAM ordinaria + validazione + copia/attesa 3,0161 ms, HIP pinned 1,7014 ms, staging esplicito incluso 2,8547 ms. L'allocatore CPU usa pinned CUDA solo se CUDA e disponibile e selezionato o gia attivo; altrimenti usa malloc. Caso di convivenza: sullo stesso buffer temporaneo cudaHostAlloc, quattro blocchi A-B-B-A senza/con/con/senza registrazione HIP mostrano copia+attesa 2,701764 -> 1,135422 ms, dati verificati esatti. Registrazione 1,5422 ms e rilascio 0,9741 ms una tantum. Sono probe di trasferimento; non una misura del guadagno di training. La validazione del probe incrociato scansiona una copia shadow, quindi si confrontano solo copia e attesa.

Nessuna registrazione introdotta nel runtime o negli allocatori. Possibile seguito: registrazione legata al proprietario dell'allocazione CPU, con budget, alias/pagine sovrapposte/riuso del pool, concorrenza, piu sessioni/device, rilascio prima del proprietario CUDA e fallback nel log; non registrare per ogni frame, non rimuovere registrazioni altrui, non memorizzare la validazione dei frame C ABI per solo puntatore. Prima di adottarla servono confronti con i veri owner del runner e controlli dei contratti. Nessun cambiamento ai percorsi CUDA, Metal e Vulkan.

Rapporto locale completo: `.local/rocm-tensor-study/optimization-followup-report.it.md`. Risultati: `sdk-sources/audit/{range-keys-screen-analysis,range-keys-fused-screen-analysis,nine-fields-analysis,frame-transfer-study,frame-cross-vendor-abba,optimization-cycle-final-scope}.json`. Il risultato accettato rimane quello della riduzione a 16 campi, 80,9721 -> 75,3521 ms nel confronto da 500 step precedentemente confermato.


Verifica indipendenza CUDA (2026-10-19): probe autonomo compilato esclusivamente con HIP, allocazione hipHostMalloc/rilascio hipHostFree, trasferimenti e letture di verifica esatti. Con CUDA_VISIBLE_DEVICES=-1 e HIP_VISIBLE_DEVICES=0, il controllo finale GetModuleHandle conferma assenza di nvcuda.dll e cudart64_12.dll. Senza HIP_VISIBLE_DEVICES, HIP interpreta anche CUDA_VISIBLE_DEVICES e nasconde la stessa GPU AMD: il primo tentativo e quindi fallito per visibilita, non per dipendenza CUDA. Nessuna modifica permanente alle variabili di sistema. La serie dei tempi di questo ultimo probe mostra deriva ed e usata solo come prova di funzionamento/correttezza e assenza di moduli NVIDIA, non per un nuovo guadagno. Non equivale a un test dell'intera app su un sistema senza CUDA installato.

Direzione dell'integrazione: allocazione host HIP propria (hipHostMalloc), budget e proprietario espliciti, fallback a RAM ordinaria con log; nessuna dipendenza CUDA. La registrazione di memoria gia esistente e solo un'ottimizzazione facoltativa di interoperabilita, non il percorso richiesto per ROCm. Il pool pinned CUDA e i backend gia esistenti mantengono il comportamento corrente.

Validazione finale dopo rimozione dei tre prototipi: suite nativa 25/25, 122,13 secondi, inclusi budget/rollback, SH, alias, modello HIP-Vulkan condiviso e convivenza CUDA. Quattro sorgenti della variante accettata e training.hip byte-identici alle ancore; opzioni RANGE_KEYS e NINE_FIELDS eliminate dalla cache. Nessun processo di test o compilazione di questo giro rimane attivo.


## 2026-10-19 — Probe diretti di driver, SDK e matematica

Solo eseguibili di prova locali; nessuna nuova modifica a kernel, API, allocatori o backend. Runtime ordinario di sistema 70260201 e runtime nel pacchetto SDK 71460850: entrambi riconoscono gfx1150, host mapping supportato e managed/pageable access non supportato. Confronto SDK isolato senza installazioni o cambi permanenti. Binari identici, ma serie A-B-B-A con deriva dei riferimenti: nessun guadagno quantitativo attribuito al runtime nuovo. Test di sessione gia compilato passato anche nel confronto SDK (SH0..3, FP32/joint-fused, budget/rollback/cancellazione e scratch), senza adottarlo nel progetto.

Matematica: 262.144 campioni per famiglia, sei famiglie, oracoli CPU e casi speciali. Exp, reciproco e sigmoid hanno codice macchina byte-identico fra percorso corrente e builtin espliciti; nessuna accelerazione possibile riscrivendoli cosi. Rsqrt protetto migliora il loop sintetico ma riguarda soprattutto SH superiori, non e un guadagno di training SH0. Log protetto non migliora; sqrt esplicito piu lento. Oggetti reali forward/backward senza istruzioni FP64 accidentali, atomiche gradienti FP32 hardware e dual issue gia emesso.

Fattorizzazione della covarianza: meno operazioni ma arrotondamenti diversi, culling differente nei casi anisotropi estremi anche con Jacobiani pinhole; non adottata. Atomiche sintetiche: stride 9 migliore su indici contigui, stride 16 corrente migliore sugli indici dispersi; serve la distribuzione reale prima di modificare il layout.

Kernel L1/SSIM reale non modificato, target 1440x1920: hipHostMallocNonCoherent mappato 6,290693 ms contro copia+loss 6,864074 ms nello screening; tutti i 24 confronti finali di gradienti/mappa bit-identici nel probe. Memoria mappata default/coherent invece peggiora. Non e una misura dell'intera iterazione o della qualita del training. Il contratto C ABI borrowed non permette di introdurre alias persistenti impliciti: serve proprieta esplicita. Primo passo conservativo: target host HIP proprio, budget e fallback a RAM ordinaria con log, mantenendo la copia attuale e senza dipendere da CUDA.

Rapporto completo locale: `.local/rocm-tensor-study/driver-sdk-math-report.it.md`, con log, limiti, confronto fra runtime e prova bytewise del perimetro. Nessuna patch SDK/driver giustificata per ora. Nessuna ricompilazione del progetto richiesta da questi esperimenti.


## 2026-10-19 — Test rsqrt sulle funzioni SH reali

SH0 di controllo e SH1..3, 32.768 primitive, coefficienti rest nonnulli, layout swizzled: colore, gradienti posizione/SH0 e dei coefficienti contro riferimento e oracoli CPU double/differenze finite. Helper generico protetto: piccoli arrotondamenti nel calcolo composto, non adottato. Specializzazione limitata ai due punti con soglie gia presenti: tutti i 7.471.104 componenti del probe coincidono bitwise nei casi ordinari/speciali/vicini alle soglie, ripetuto in due processi. Errori massimi SH3 vs CPU: colore 2,277e-7, posizione 1,070e-7, coefficienti 2,673e-7.

Nessun guadagno robusto: throughput wall-clock del solo gradiente posizione SH1 +1,79%, SH2 -0,05%, SH3 +0,98%; non sono tempi del training. Forward e gradienti coefficienti hanno codice macchina byte-identico, il gradiente posizione elimina solo 28/32 byte. Il -32% scalare non si trasferisce al carico SH reale. Prototipo conservato solo negli artefatti, nessuna nuova opzione o modifica ai kernel del branch.

Le serie a eventi GPU interleaved sono escluse: osservati tempi negativi. Riproduttore minimo indipendente con stream esplicito: 2 negativi su 2048 coppie sia runtime 70260201 sia 71460850, query hipSuccess, non riparati da sincronizzazione esplicita. Causa nel percorso runtime/driver ancora da identificare; nessuna patch o issue pubblicata. Le misure step_ms del runner sono chrono host e non dipendono da questi timestamp; anche il confronto finale SH usa timer host e sincronizzazione.

Rapporto locale: `.local/rocm-tensor-study/sh-rsqrt-validation.it.md`; dati e SHA-256 in `sdk-sources/audit/sh-rsqrt-*.json`, riproduttore `hip-event-order-probe.hip`. Sorgenti accettati e training.hip byte-preservati; CUDA/Metal/Vulkan invariati.


## 2026-10-19 — Immagini CPU con proprietario HIP (opt-in)

Il runner sperimentale a topologia fissa supporta `LFS_HIP_HOST_TARGETS=1`: CameraFrame conserva la sua copia immutabile in un'allocazione propria hipHostMallocDefault, tramite proprietario condiviso e modulo mantenuto caricato fino al rilascio. Nessuna dipendenza CUDA aggiunta. Budget pinned separato, massimo 512 MiB, riducibile con `LFS_HIP_HOST_TARGET_BUDGET` (byte); nessuna seconda copia persistente della cache CPU. Set_frame conserva la copia sincrona al buffer GPU e la validazione corrente: nessun alias host persistente implicito.

Fallback a RAM ordinaria con motivo CLI per budget insufficiente, allocazione fallita o modulo precedente senza simboli opzionali. Budget malformato gestito come errore dell'opzione facoltativa. Il test di errore nativo ha rilevato e fatto correggere lo slot last-error HIP residuo che poteva contaminare un kernel successivo; ora verificato nello stesso processo prima di una loss valida.

Suite nativa definitiva 26/26 (121,03 s), inclusi proprieta/copie/lifetime del modulo, budget/riuso, quattro thread, frame NaN rifiutato senza danneggiare il frame preparato, loss valida dopo il rilascio del buffer host. Verificati anche training reali di fallback: budget di un byte, vecchio modulo butterfly e budget malformato. Kernel, matematica, set_frame e optimizer byte-preservati rispetto allo snapshot prima della prova; CUDA/Metal/Vulkan non modificati da questo intervento.

Screening ABBA a 200 passi: caricamento frame medio 4,7572 -> 3,1898 ms; tempo totale step+frame 90,2249 -> 87,1663 ms. Questo screening precede il monitor continuo delle compilazioni e rimane preliminare. Due successivi riscaldamenti contaminati da compilazioni esterne sono stati esclusi; la conferma richiede 30 secondi iniziali senza compilatori e monitor a 250 ms durante ogni esecuzione.

Rapporto e prove locali: `.local/rocm-tensor-study/host-target-training-report.it.md`, `host-target-final-suite.log`, `host-target-scope-audit.json`, `host-target-fallback-results.json`. Non e ancora il trainer applicativo completo; il mapping diretto nella loss resta separato e non implementato.

Conferma conclusa, serie `host-target-confirm-clean2`: quattro training da 500 passi ABBA, due viste 1920, 1M/SH0, tutti senza compilatori rilevati. Frame 5,2052 -> 3,3226 ms (-36,17%); step 75,4096 -> 75,9514 ms (+0,72%, oscillazioni); totale 80,6148 -> 79,2740 ms (-1,66%). MSE medio finale 0,0104681250 -> 0,0104699783 (+0,0177% relativo), differenze dei render dello stesso ordine delle ripetizioni e nessun peggioramento visivo strutturale osservato nelle due viste e nei dettagli; nessuna identita bitwise del training dichiarata. Cache CPU 66.355.200 byte in entrambi i casi; picco GPU contabilizzato identico 769.785.936 byte, non una misura completa del driver/processo. Percorso conservato opt-in, modulo locale `fast-butterfly-host-targets-accepted.dll`. Dettagli e limiti in `host-target-training-report.it.md`, analisi `host-target-confirm-analysis.json`, immagini `host-target-confirm-quality-{full,detail}.png`.


## Ripetizioni delle nuove organizzazioni globali dei tile, con controllo delle interferenze

Esperimenti nuovi rispetto al registro precedente: cambiano l'assegnazione globale dei tile, non le formule per pixel o le precedenti riduzioni intrawave. Prima prova: quattro classi di parita, per distanziare i tile contemporanei. Seconda: macroblocchi 4x4, per favorire localita. Terza: quattro tile indipendenti in un workgroup di 128 thread, conservando una wave32 e otto pixel per lane per ciascun tile. Nessuna LDS/barriera fra le wave nel kernel privato; guard delle wave finali prima di leggere i range. I tre prototipi sono soltanto negli artefatti locali e **non sono adottati**.

Copertura delle permutazioni: stesso corpo intero verificato su CPU (70.480.909 visite) e GPU (121 griglie, 112.896 visite, identita CPU/GPU); griglie dispari, strette, bordi. Packing: 4.101 dimensioni, 8.608.071 visite e 256 pixel unici per tile. Ogni variante ha passato matematica, raster/clipping e FastSession SH0..3 (3/3).

La prima serie pack4 aveva riferimenti 77,97 e 86,11 ms, deriva 10,43%: non e una misura affidabile. La prima serie macro4 mostrava -0,65%, troppo piccolo rispetto alla variabilita. L'utente ha richiesto la ripetizione per possibili lavori concorrenti. Il controllo ha rilevato realmente build FFmpeg/AV1 Windows (cl/CMake/Ninja/link), build Linux in Docker/WSL, ctest, script Python, compressione 7z e carico Defender. Non si attribuisce retroattivamente ogni vecchia oscillazione a una causa specifica.

Ripetizione finale: `tile-repeat-background-v3`, stessi quattro binari archiviati verificati SHA-256, due viste 0/41 del dataset statua_budda_villaggio, 1M gaussiane, SH0/fused, massimo 1920, L1/SSIM 0,2; RAM host HIP in entrambe le varianti. Riscaldamento separato di 60 passi, come nello screening originale; ciascun confronto usa quattro esecuzioni da 200 passi in ordine A-B-B-A, primi dieci esclusi. I 300 passi di warmup tentati in v2 sono stati esclusi per sovrapposizioni reali e non entrano nei risultati.

Monitor a circa un secondo: nomi di compilatori/test, script Python, CPU per processo (inclusa WSL) e motori GPU per PID tramite contatori Windows PDH. Prima dell'avvio si richiedono 30 secondi senza i carichi rilevati. Le esecuzioni con conflitti sono conservate, escluse e ripetute; il monitor deve vedere la nostra GPU in ogni campione della finestra misurata, senza errori o lacune oltre tre secondi. Dati GPU temporaneamente non disponibili fuori dalla finestra sono espliciti nella telemetria. Limite dichiarato prima dei confronti: deriva massima 3% fra i riferimenti, altrimenti ripetizione dell'intero gruppo. Il monitor e passivo ma non gratuito: costo CPU registrato per ogni campione; stesso controllo in tutte le varianti. Non equivale a isolamento completo del sistema o a una prova statistica per differenze sotto l'1%.

Sono state accettate 12 esecuzioni, scartate 6 nella serie finale; tutte le finestre misurate accettate hanno copertura GPU completa e nessun conflitto rilevato. Le esecuzioni valide seguono l'ordine ABBA; fra esse possono esserci tentativi scartati e attese di quiete, documentati nei log, quindi non si descrive la serie come ininterrotta.

| Variante | Riferimento step ms | Candidato step ms | Variazione | Deriva riferimenti |
|---|---:|---:|---:|---:|
| Ordine globale a quattro parita | 77.815206 | 78.652347 | +1.076% | 1.791% |
| Ordine globale in macroblocchi 4x4 | 77.220376 | 77.824554 | +0.782% | 0.173% |
| Quattro tile indipendenti per workgroup | 77.521736 | 83.167962 | +7.283% | 0.470% |

Valori positivi indicano maggiore lentezza. I risultati piccoli di phase2/macro4 non dimostrano una regressione universale, ma non offrono un beneficio da adottare; il -0,65% precedente di macro4 non si conferma. Pack4 e sfavorevole anche nella ripetizione: circa +7,28%. La query del runtime sui kernel reali stima 24 wave32 attive per multiprocessore nel riferimento e 20 in pack4; metadati 160 -> 167 VGPR, 75 -> 68 SGPR, nessuno spill o scratch privato. Sono stime e risorse statiche, non contatori dell'occupancy effettivamente osservata: non provano da sole la causa del rallentamento.

Target PPM identici per SHA-256 all'interno di ogni confronto. MSE finali e dati originali conservati integralmente nel JSON; niente garanzia di identita bitwise del training o su holdout. Non e stato necessario un nuovo confronto lungo da 500 passi per adottare una variante: nessuna ha mostrato un guadagno credibile. I tempi sono chrono host del runner, non eventi HIP/PDH. Sono tempi di step; non includono preparazione iniziale, binding immagini o esportazione e non vanno confrontati direttamente con i precedenti 500 passi.

Entrambi i generatori sono ripristinati byte per byte agli snapshot accettati. Nessun nuovo flag o modifica runtime adottata, nessuna patch SDK/driver, nessuna modifica CUDA/Metal/Vulkan in questo giro. La selezione di immagini host HIP precedentemente accettata rimane opt-in. Modello/trainer applicativo completo e refinement restano fuori dal runner sperimentale.

Prove locali: `.local/rocm-tensor-study/tile-repeat-final-analysis.json`, `tile-repeat-background-v3-{manifest,results,comparisons}.json`, telemetrie `*.telemetry.json`, `tile-repeat-restoration-and-binaries.json`, `tile-pack4-{coverage,safety-audit,backward}.json`, `tile-pack-occupancy.log`, log di test `tile-order-tests.log`, `tile-order-macro4-tests.log`, `tile-pack4-tests.log`. I confronti iniziali e i tentativi v1/v2 rimangono disponibili e non sono mischiati alla ripetizione finale. API di riferimento per i campioni wildcard: https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw .

Verifica finale del ripristino: matematica, raster/clipping, FastSession SH0..3/convergenza/cancellazione e OwnedHostTargets **4/4 passati**. I due generatori coincidono byte per byte con gli snapshot accettati; anche l'ELF GPU backward ricostruito e byte-identico all'immagine accettata prima degli esperimenti. Prove: `tile-repeat-restored-validation.json`, `tile-repeat-restored-build.log`, `tile-repeat-restored-tests.log`. Nessuna nuova ottimizzazione adottata.


## Verifica a 2.000 passi e ripresa integrazione — 20 ottobre 2026

Nuova esecuzione HIP valida, 1M fisso, SH0, 1440x1920, due viste 0/41,
seed dal checkpoint funzionale `final-adam-training-cuda/project.licht`, Adam
azzerato. Finestra 201–1900: step 65,764643 ms, frame 2,646883 ms, somma
68,411526 ms; mediana step 65,0292, p95 73,0267. Totali delle 2.000
iterazioni: 134,198585 s di step e 5,312534 s di bind, esclusi preparazione,
render di controllo ed export. Nessun conflitto rilevato nella prova accettata;
primo preriscaldamento contaminato conservato ed escluso.

MSE training 0,0736427444693 → 0,00655564343391. Controllo visivo di entrambe
le viste: statua e basamento ricostruiti nettamente meglio del seed, vegetazione
ancora sfumata e striata. Sono due viste usate per ottimizzare: non è prova di
qualità multi-vista, generalizzazione o parità con il trainer applicativo.
Picco sessione 769.785.168 byte, target host 66.355.200 byte; non confondere
contabilità sessione con residenza fisica complessiva. Nessun fallback ingest.

Confronti storici, finestre e metriche originali sono conservati in
`.local/rocm-tensor-study/backend-2000-comparison-oct20.it.md` e nel JSON
omonimo. CUDA e Mac non sono stati rieseguiti per performance. Il controllo
AMD Vulkan successivo da checkpoint misura 114,591 ms di step + 9,936 ms di
loader: non usare il riferimento precedente 229,98 ms per rappresentare lo
stato più ottimizzato. Protocolli e seed differiscono, quindi nessun rapporto
HIP/Vulkan viene dichiarato un confronto equivalente di training completo.

Primo incremento per integrazione: `HipBackendOps::dot_product`, `diag` e
`eye` sono ora nativi tramite i primitivi HIP esistenti. Dot usa temporaneo
GPU con ciclo di vita protetto e riduzione sulla stessa coda; è un percorso
corretto di base, non una nuova ottimizzazione fused. Nessun trasferimento CPU,
nessun fallback ad altri backend. Diag/eye controllano estensioni complete
prima delle scritture; diag rifiuta overlap. Contratti reali passati per offset,
padding, matrice rettangolare, identità vuota, overflow, memoria di altro
backend, errore senza modifica output. Il controllo di operazioni ancora
non implementate usa ora SGEMM, invece del dot appena portato.

Verifiche: build MSVC lfs_core riuscita, Tensor smoke con CUDA nascosta e
con CUDA visibile riusciti (inclusi controlli CUDA di correttezza prima/dopo
scope HIP); contratto backend isolato passato; ponte modello/camera/sessione
400 step passato; admission del trainer incompleto ancora respinta senza
inizializzare altri backend. Log `integration-matrix-*.log`. Nessun kernel
CUDA/Metal/Vulkan modificato da questo incremento. Il kernel HIP del benchmark
precede queste aggiunte e non ne dipende.

### Lavoro applicativo ancora necessario, in ordine di dipendenza

1. Completare i contratti Tensor richiesti dal trainer e dalle strategie:
   scritture indicizzate/maschere, multinomial, riduzioni fuse, sincronizzazione
   e readback; aggiungere test su capacità oltre il conteggio vivo e owner.
2. Esporre vere famiglie `TrainingOps` HIP (oggi tabella vuota intenzionale):
   Fast/Gsplat, photometric, Adam, SH, TrainingImage, SharedImage; per MCMC
   anche Mcmc e Refine. Non aggirare `required_training_families`.
3. Collegare crescita/prune/relocation con conservazione di momenti e codec,
   generazioni del modello, capacity e budget; poi restore completo checkpoint
   con stato Adam e scheduling. Oggi un PLY non è un checkpoint riprendibile.
4. Collegare ingest/cache asincroni applicativi e tutte le immagini con budget
   RAM, invalidazione, cancellazione e fallback visibile nel log. Il limite
   sperimentale 512 MiB/due viste non costituisce ingest applicativo completo.
5. Collegare storage condiviso al viewer Vulkan sulla stessa scheda e copie
   esplicite su schede diverse, con owner/sync, resize/growth e recovery.
6. Preferenze opt-in, rilevamento device e indicatore riavvio coerenti con il
   backend realmente attivo. Ammissione solo per capacità effettivamente coperte.
7. Ripetere infine 2.000 step con stesso seed, split, MCMC/refinement, stato
   iniziale, cache, schedule e valutatore del trainer; controllare qualità e
   viewer, CUDA e Mac. Soltanto allora formulare una classifica diretta.

Questa lista resta aperta: le tre operazioni aggiunte non equivalgono a piena
integrazione, e non autorizzano a promuovere HIP nelle preferenze o nei default.


## Integrazione applicativa in corso — blocco I/O verificato (20 ottobre)

Richiesta esplicita: completare il collegamento al trainer, non promuovere il
runner fisso come training applicativo. Il benchmark a due viste resta separato.

Collegati e verificati nelle API reali `lfs_core`:
- `TensorUpload`: copia CPU immediata in staging HIP pinned, trasferimento
  asincrono sulla coda HIP, owner fino al completamento, riuso/crescita e guardia
  dello slot occupato. La modifica della sorgente CPU/span dopo enqueue non
  altera il trasferimento.
- `TensorReadback`: snapshot, offset e sorgenti strided, riuso dello staging,
  destinazione preparata e durata del contesto/modulo oltre il facade.
- `TensorCompletion`: marker HIP di completamento, ready ripetibile e obblighi
  backend-wide. Prima il capture di un Tensor HIP non registrava il lavoro:
  questa omissione applicativa è ora corretta.
- `TensorFence`/`TensorWorkQueue`: marker riutilizzabili, attese sulla stessa
  coda senza attesa CPU al record, callback host native HIP e ticket execute.
  Le code logiche HIP sono serializzate su una coda nativa: non dichiarare
  parallelismo fra code. Gli handle CUDA/target estranei sono rifiutati.
- `TensorReadbackRing`: slot host pinned HIP, append di più range con offset,
  padding, seal, riuso e staging GPU solo se richiesto esplicitamente; nessuna
  seconda copia del modello GPU per il normale readback.
- Shutdown rifiutato con allocazioni, queue lease, trasferimenti o fence vivi.
  Gli errori di teardown mantengono caricati codice e risorse ancora in uso.

Prove passate, CUDA nascosta e HIP device 0: contratto backend nativo completo
(1,52 s), smoke applicativo con tutte le API precedenti, contratto ABI di
readback, training core/private 400 step (tutte le alias byte-exact, righe
riservate intatte), admission incompleta ancora rifiutata. Ripetuto anche il
controllo di correttezza CUDA prima/dopo lo scope HIP, con CUDA visibile: passa.
Nessun benchmark CUDA ripetuto. Log `integration-pipeline-*.log`; snapshot
verificato dei 12 file I/O in `integration-io-accepted-sources-oct20` e manifest
SHA256 omonimo. Le successive aggiunte FastGS non fanno parte di questo snapshot.

In lavorazione: C ABI FastGS applicativa con queue lease del Tensor runtime,
parametri/momenti del trainer presi in prestito e workspace nativo bounded.
`LFS_HIP_APPLICATION_FAST` registra solo questa famiglia nei build che abilitano
esplicitamente Tensor HIP, training runtime e Fast runtime. La tabella resta
incompleta: non abilita ancora un training principale né preferenze/default.
La compilazione/confronto del nuovo FastGS applicativo è ancora da verificare;
non confondere il codice aggiunto con un risultato di porting già superato.


### Application TrainingOps progress, 20 October: five families verified

The real HIP application table now has opt-in implementations of FastGS,
MCMC, refinement, photometric loss and standalone Adam. This does not admit
the complete application trainer: SH storage, image/ingest families, session
and viewer/shared storage still need integration and acceptance. No complete
2,000-step comparison is claimed from these family tests.

All five application contracts pass sequentially on HIP 0 with
HIP_VISIBLE_DEVICES=0 and CUDA_VISIBLE_DEVICES=-1. Fast forward is identical
to the validated private renderer; one-step fused Adam max error 1.1920929e-7.
Standalone Adam CPU update max error over three updates is 6.15e-8; frozen/crop
rows, FP32/F16/Q16 SH and 8/16-bit row/swizzle moment resets preserve unrelated
blocks and padding. All six photometric loss paths agree with an independent
double-precision 2-D CPU reference (max loss delta 5.77e-8); sampled finite-
difference gradient max absolute delta 1.2e-6, with unchanged test tolerances.
Byte targets/Bool masks, fixed/automatic normalization, empty masks, metric/
error maps and workspace reuse/reset pass. MCMC sampling is deterministic
for fixed HIP seed, first-row frequency .24925 vs .25 expected; copy/update,
noise statistics, frozen rows and fold/reset pass. Refinement's eight ops pass,
including exact positive median vs sorted CPU reference over 10,007 values,
counts, clipping, scores, split, slot reuse, pruning and teardown.

Core native code modules are retained until the Tensor context has drained
its queue; the module cache is weak and cannot retain the context cyclically.
GPU I/O and the new typed fills/clamp stay HIP-specific. Existing CUDA,
Metal and Vulkan operation implementations have not been edited. This is
source isolation evidence, not a new full macOS regression run.

Evidence: application-{fast,mcmc,refine,photo,adam}-contract.log,
application-families-host-build.log, application-adam-native-build.log.
SHA256 manifest: application-five-families-accepted-oct20.json.


### Integrazione applicativa, 20 ottobre: sette famiglie verificate

Sono ora collegati e verificati FastGS, MCMC, refinement, loss fotometrica,
Adam, tutte le operazioni SH e sessione. I contratti girano sequenzialmente
su HIP 0 con CUDA nascosta; non costituiscono ancora un training applicativo
completo. Il gate di ammissione resta chiuso alle famiglie mancanti.

SH passa per rest=3/8/15: swizzle, gather float/byte, scatter, azzeramento,
chunk Q16 con sentinelle, decode FP32/F16/Q16 a intervalli, rounding host
identico per serializzazione Q16, compattazione di tre run, aggiornamento
dei soli blocchi toccati con ordine canonico indiretto, copia diretta in
arena senza duplicare la concatenazione. Le viste raw HIP accettano solo
intervalli di allocazioni Tensor vive e registrate; il proprietario dell'arena
deve vivere almeno quanto la vista, come per i raw-view esistenti.

La sessione usa workspace FastSaved privati: non presta memoria di un frame
in uso. Gli scratch opzionali mantengono il percorso Tensor, analogamente
a Metal. Verificati discovery senza inizializzazione, memoria HIP reale,
controllo StreamQuery senza sincronizzazione della coda sana, rifiuto target
estranei, timeout, trim con Tensor vivi e shutdown. Il profiling hook non
avvia una cattura SDK; i benchmark applicativi usano i normali timer host.

Prossimi passaggi necessari: TrainingImage e SharedImage/ingest, funzioni
ausiliarie richieste dalla configurazione, storage condiviso/importazioni
HIP-Vulkan, preferenze/discovery, build trainer completa e validazione
checkpoint/crescita/pruning/resume. Solo dopo: confronto a 2.000 step con
le impostazioni comuni e verifica qualitativa, senza riciclare il benchmark
privato a topologia fissa come confronto equivalente.

Snapshot completo dei file verificati: application-seven-families-accepted-
sources-oct20; manifest SHA256: application-seven-families-accepted-oct20.json.
Log aggiuntivi: application-sh-contract.log e application-session-contract.log.


### 20 ottobre: ottava famiglia applicativa verificata

TrainingImage ora usa la coda e le allocazioni Tensor HIP per EMA delle camere,
ROI, sfondi, Canny, normalizzazione e quantizzazione. Contratto GPU con CUDA
nascosto: EMA/hash/quantizzazione esatti; resize contro riferimento CPU double
con errore massimo 2.41964841e-7; Canny contro convoluzione/NMS CPU indipendente
con errore massimo 4.66818609e-7 su dimensioni 35x37, oltre ai controlli UInt8,
maschera nulla, crop traslato/inverso e rifiuto di input CPU. Tutti gli otto
contratti applicativi ripetuti in sequenza passano con lo stesso runtime.

La famiglia SharedImage del core rimane non registrata: sono in implementazione
le conversioni native, prima di completare resize Lanczos e undistortion. Il
trainer completo rimane correttamente non ammesso. Nessun nuovo dato di velocità
end-to-end deriva da questi contratti; il confronto equivalente a 2000 step resta
da eseguire dopo il collegamento effettivo del trainer.


### Primitive native per ingest: verifica separata

Nuovo seam nel runtime Tensor del core: sedici conversioni ImageConversion,
sentinelle di decodifica, split RGBA float/byte e tre trasformazioni maschere.
I kernel di formato derivano dai sorgenti esistenti, senza modificarli. Primo
contratto nativo: sedici conversioni contro riferimenti CPU, tutti i 65536 alpha
UInt16 con divisione arrotondata esatta, sentinelle mutate/non mutate, split
RGB/alpha, normali sRGB ruotate e rifiuto di overlap/contesto estraneo/disallineamento.
La prima verifica precede l'aggiunta della terza trasformazione maschere; il
contratto esteso e il ponte Core Tensor vengono verificati a parte. La tabella
SharedImage pubblica non viene dichiarata disponibile finche' resize e lens
non sono completi: questi risultati non equivalgono ancora al loader integrato.


Verifica estesa completata: anche quantizzazione maschere a byte-grid e ponte
tra il seam di conversione e veri Tensor Core HIP passano. Ripetuti con il nuovo
core tutti gli otto contratti applicativi e il contratto runtime di ownership,
queue/readback e warp: tutti passano, CUDA nascosto. Questo non registra ancora
SharedImage e non rende disponibile il trainer completo. Risultati in
application-image-contract.log e shared-image-runtime-contract.log.


### Resize condiviso: cinque percorsi nativi e ponte Tensor verificati

Portate le funzioni device Lanczos RGB HWC (1-4 canali, byte/float), grayscale,
Float32 CHW e resize validity-aware per depth/normal. Workspace coefficienti
espliciti nelle allocazioni Tensor Core, senza allocator GPU esterno. Verificati
up/down, canali/layout, prior non validi, alias e stride insufficienti. I valori
CPU dei coefficienti seguono la specifica FP32 esistente; la convoluzione CPU
rimane indipendente in double. Un primo confronto interamente double aveva
esposto la sensibilita' numerica della normalizzazione in ingrandimento ~5x:
corretto il riferimento al contratto FP32, non aumentata la tolleranza 2e-5.
Errore massimo 1.51839911e-6. Verificati anche HWC/CHW e depth sui Tensor reali.
Tutti gli otto contratti applicativi ripetuti dopo il nuovo core passano.

SharedImage resta non registrata: manca ancora la correzione ottica completa,
oltre alle successive famiglie/configurazioni, interop/viewer e trainer finale.
Nessun confronto end-to-end nuovo a 2000 step e nessuna dichiarazione di
integrazione completa deriva da questi test. Log: shared-resize-runtime-contract.log.


### Collegamento applicativo: SharedImage ed ExtraLoss verificati

Registrata la tabella HIP SharedImage completa. Contratto reale: tutte le sedici
conversioni, sentinelle, cinque modelli ottici, immagini UInt8/Float32 a 1/3/4
canali, identita' esatta, maschere/depth/normal su piu' batch di righe, cinque
resize, rifiuto di input estranei e teardown passano. Nessun fallback CPU nel
percorso HIP; i confronti CPU sono esclusivamente nel test.

Registrate anche regolarizzazioni scale/opacity e backward ADMM. Riferimenti CPU
su 263, 10007 e 300001 elementi, riduzione grid-stride, accumulo/overwrite,
value-only, peso nullo e scratch insufficiente passano. Massimo errore loss
3.9929419e-8; gradiente 7.44256656e-9. Sono dieci famiglie applicative verificate,
non un'ammissione del trainer completo. Avviata build del trainer reale in
build-application con CUDA disabilitato, runtime HIP opt-in e due compilazioni.
Morton e codec SH Core sono in preparazione e non ancora verificati/registrati.

### Integrazione applicativa: undici famiglie verificate, trainer end-to-end ancora in verifica

La tabella HIP collega ora Fast, MCMC, Refine, Photometric, Adam, SH, Session, TrainingImage, SharedImage, ExtraLoss e Morton. Tutti gli undici contratti sono stati ricompilati ed eseguiti con lo stesso runtime nativo e CUDA nascosta. Anche il codec SH del core passa i confronti CPU per SH1/SH2/SH3, FP32/F16/Q16, gather/scatter, offset, alias, padding ed espansione SH0.

Morton verifica chiavi intere a 30 bit, stabilita dei pareggi e bit bassi, momenti compressi a 8/16 bit, uguaglianza bit per bit fra gruppi e buffer completo, permutazione del vero SplatData e copie da intervalli nativi posseduti dal core. L'errore massimo osservato nel riordino SH Q16 e 2.2649765e-6 (entro il limite della quantizzazione); FP32/F16 seguono il riferimento CPU esatto. Il test ha inoltre individuato un dispatch globale errato negli helper condivisi: nel solo percorso HIP ora viene rispettato il contesto di esecuzione HIP esplicito.

Il rilevamento hardware usa le entry point di inventario del modulo, senza creare allocatore o contesto Tensor; disponibilita e conteggio ripetuti passano il controllo di assenza di effetti sul backend configurato. La configurazione HIP esplicita e consentita nelle build opt-in; i candidati automatici restano invariati. Il controllo delle famiglie mancanti resta obbligatorio.

La prima build completa della libreria lfs_training con LFS_ENABLE_CUDA=OFF e riuscita. E stato aggiunto lfs_hip_application_trainer, che usa loadTrainingDataIntoScene, initializeTrainingModel e il vero Trainer::train: il vecchio runner a topologia fissa non viene usato per dichiarare completata l'integrazione. Le prove end-to-end di questo nuovo eseguibile sono ancora in corso: non sono ancora dimostrati checkpoint/ripresa, interop viewer, tutti i moduli opzionali o il confronto equivalente a 2000 step. Nessun nuovo tempo comparativo viene dichiarato su questa base.
### Pipeline reale del trainer: prime prove superate

Il nuovo eseguibile di verifica lfs_hip_application_trainer usa le API effettive dell'applicazione: importazione delle 82 camere del dataset, initializeTrainingModel, Trainer::initialize/train, salvataggio headless, riapertura del ProjectDocument e verifica del CKPT associato (iterazione e numero di gaussiani). La build ha LFS_ENABLE_CUDA=OFF e le prove nascondono CUDA. Non usa il vecchio runner HIP a topologia fissa.

Prove riuscite a larghezza massima 320, CPU offload, MCMC e correzione ottica attiva: 2 step SH0; 620 step SH0 con crescita da 19555 a 20532 gaussiani; ripresa 620->720 con ulteriore crescita a 21558; ripresa 2->4; prova accelerata SH0->SH3, crescita e Morton ogni due step fino a 8, seguita da ripresa 8->10. Tutti i casi riusciti salvano e riaprono il checkpoint. La prova accelerata e esplicitamente marcata NOT A BENCHMARK e non fornisce tempi confrontabili.

Queste prove hanno individuato e corretto tre collegamenti effettivamente mancanti: riduzioni Tensor fuse/segmentate necessarie all'inizializzazione, calibrazione dei trasferimenti checkpoint senza timestamp GPU HIP, e scrittura SH per snapshot in scratch UInt8 anziche solo Tensor Float32. Le riduzioni restano fuse sul dispositivo (nessun intermedio grande quanto il modello), con al massimo il workspace di riduzione gia previsto dal core. Il contratto numerico copre quattro riduzioni, segmenti brevi/lunghi/divisi, lease rhs, guardie e facade Tensor pubblica. Il contratto SH aggiunge destinazioni byte allineate per FP32/F16/Q16, equivalenza esatta con destinazioni tipizzate, guardie e rifiuto dei disallineamenti.

La calibrazione HIP usa tempo monotono sincronizzato dopo il completamento dei trasferimenti e lo segnala nel log CLI; non viene presentata come tempo del dispositivo. Le annotazioni dei range del profiler sono opzionali e disabilitate esplicitamente per HIP, senza dipendenza NVTX/CUDA. I percorsi degli altri backend per calibrazione e annotazioni restano quelli esistenti.

Rimangono da completare: tutti i moduli opzionali ancora non registrati, selezione e identita delle GPU, interop del viewer e gestione delle configurazioni miste, verifiche estese di qualita e confronto equivalente a 2000 step con i parametri storici. Non e ancora una dichiarazione di integrazione completa o di assenza globale di regressioni.
### Integrazione applicativa: LPIPS e pubblicazione viewer (21 ottobre)

La famiglia LPIPS dispone ora di dispatch HIP nativo dei programmi del repository: il testo CUDA prodotto da Slang è solo un intermedio, privo di prelude, compilatore o runtime CUDA. Il controllo del modello VGG reale a 32x40 usa i pesi locali `lpips-vgg16-v0.1.lfw`: identità, simmetria e ripetizione warm passano. Punteggio HIP 0.218880534; riferimento Vulkan 0.218885884 (differenza assoluta 0.00000535). È una verifica di correttezza, non un benchmark di training. Il pooling per la piramide delle maschere è oggetto di verifica separata; non dedurre copertura delle maschere dal solo confronto non mascherato.

L'inventario HIP interroga nome, architettura, identità e memoria dal runtime senza creare il contesto Tensor. La scelta UUID/indice è verificata sulla Radeon 890M e si congela all'avvio del contesto; una selezione non valida conserva le opzioni precedenti. I circa 63 GB riportati dal runtime UMA non costituiscono un budget di VRAM dedicata. Non sono disponibili due schede HIP fisiche per una prova multi-HIP.

Il collegamento iniziale Core HIP -> consumer Vulkan usa il fallback host con sincronizzazione e messaggio CLI esplicito. `viewer_interop_contract.cpp` passa con Radeon 890M e GeForce RTX 5060 Ti come consumer separati: vecchio lease immutabile dopo modifica del produttore, lettura e scrittura mediante comandi Vulkan, attesa timeline, ritorno dei dati al Tensor HIP, coda consumer e teardown. Questi test non equivalgono a una prova del viewer GUI completo. La condivisione del modello sulla stessa GPU senza duplicazione resta da integrare: il fallback usa memoria consumer distinta e non deve essere presentato come zero-copy.

Restano aperti: famiglie opzionali non registrate, primitive Core ulteriori, interop condiviso, selettori e persistenza nell'interfaccia, build GUI completa e confronto reale equivalente di 2.000 iterazioni. Il ramo non è ancora una integrazione completa e i tempi del runner privato non sostituiscono tale confronto.
### 21 ottobre: app GUI nativa, preferenze e Structure

La build completa LichtFeld-Studio + modulo Python e compilata con CUDA disabilitato. Verificati nell'app via MCP: Tensor HIP e identita ROCm DLPack (device type 10), rifiuto degli import estranei prima di consumare capsule, inventario lazy, cambi backend/GPU con indicatore di riavvio e sua rimozione tornando alla configurazione attiva. Screenshot della sezione Hardware & GPUs e del badge arancione Restart needed conservati nello studio locale. Non e ancora implementato l'import di allocazioni ROCm estranee.

Il training GUI ha individuato due connessioni mancanti: selezione dell'allocatore generico dello storage splat e sostituzione della timeline del consumer. Correzioni verificate da contratti su crescita in-place, dati precedenti, slack opacity/quaternion, overflow rifiutato, sostituzione di semafori e consumer senza device. Ripetuti con successo tutti i 20 contratti precedenti. La GUI ha poi completato 12 iterazioni MCMC a max-width 320, cap 30000, SH0, CPU offload e undistort, con thin_structure_weight=0.5 e densify_structure_weight=1, salvando project.licht senza nuovi errori di rendering. La loss sui gradienti non viene attivata da 12 iterazioni; e verificata separatamente dal contratto numerico.

Registrata la tredicesima famiglia Structure: ridge a quattro scale con workspace per bande, pesi fotometrici/densificazione e loss Sobel con backward accumulato. Shader del progetto abbassato a HIP senza SDK CUDA, parametri per dispatch e scratch condiviso per blocco. Il generatore ordina le letture XOR dei conteggi prima della barriera per evitare race fra wave. Contratto PASS: riferimenti CPU per pesi e loss/gradiente analitico, tile dispari, accumulo, equivalenza delle bande, normalizzazione ridge, input UInt8 rispetto a Float32 normalizzato, maschere tutte valide e tutte nulle. Non sono state modificate le implementazioni numeriche CUDA/Metal/Vulkan.

Verifiche UI aggiuntive: 152 test Python, policy GPU HIP on/off (16+13), viewer GPU memory (3 PASS e un caso CUDA non compilato SKIP). La telemetria HIP non usa i valori NVML della GeForce; memoria condivisa riportata dal driver non e descritta come VRAM dedicata.

Il controllo CLI preliminare con seed storico e evaluation ha scartato tutte le viste per mancanza del warp inverso HIP. Quel run e FALLITO e non vale come confronto di qualita. Il ricampionamento inverso e ora in implementazione/verifica prima di eseguire il benchmark da 2000 step. Nessun nuovo tempo comparativo dichiarato. Rimangono otto famiglie applicative non registrate (Gsplat, Geometry, Masks, Mrnf, Bilateral, PPISP, Controller, Blob), zero-copy HIP/Vulkan sulla stessa scheda e completamento delle verifiche di compatibilita. Il viewer usa ancora fallback host sincronizzato, esplicitamente segnalato nella CLI, anche sulla stessa scheda.
### 21 ottobre: evaluation reale e NN Core

Il warp inverso HIP e ora collegato per colore con maschera di validita, maschere, depth HW/CHW a un canale e normal. Il contratto confronta raggi analitici e copertura CPU, bordi/tile dispari, cinque modelli ottici e tutti i modi con Vulkan. Passa. La prima evaluation completa del trainer reale valuta tutte le 11 viste e salva le immagini, PSNR/SSIM/LPIPS e progetto; e uno smoke test a 2 step, non una misura di qualita finale.

Completati i cinque kernel NN Core prima mancanti: im2col, col2im, resize, attivazioni e griglia; il pooling nativo precedente resta invariato. GEMM FP32 nativa copre prodotto normale, peso trasposto, batch e bias/ReLU: tile condivisi per matrici e un kernel per poche righe, senza librerie CUDA. Il contratto verifica riferimenti CPU, batch, dimensioni dispari, K nullo, guardie, padding, canali di ingresso/uscita diversi nel fold, cinque attivazioni, nove combinazioni di resize e descrittori/alias rifiutati senza scritture. Tutti i 23 contratti applicativi passano con la stessa build.

L'uscita headless HIP senza script ora lascia distruggere trainer, scena e generazioni ritirate prima del teardown in main. Il controllo reale a 3 step con crescita e checkpoint termina con successo e senza l'avviso sulle allocazioni vive. I percorsi di uscita CUDA/macOS e quelli con script mantengono la scelta preesistente. Gli eventi temporali HIP restano volutamente non esposti: i riproduttori precedenti in sh-rsqrt-validation.it.md avevano gia dimostrato tempi negativi del driver; non vanno presentati come GPU time attendibile.

Il primo tentativo da 2000 step e stato fermato ed escluso: applicare come override di ripresa il training_config.json esportato, che contiene step non scalati, cambia il calendario effettivo. Un preparatore CPU C++ usa OptimizationParameters::apply_step_scaling e rende espliciti i valori risolti con steps_scaler=1: start_refine=33, refine_every=7, stop_refine=1667, Morton=333. Il nuovo run real-hip-historical-2000-oct21-r2 verifica questi valori anche nel file effettivamente scritto dall'app, usa seed SHA256 4273c94a3c29daffc84c490651510d91ac71ba9cfb9eba8aa3abc0fced3f63f7, cap 1000000, SH0, max-width 1920, CPU offload e evaluation sulle stesse 11 viste. E ancora in corso; nessun tempo finale o giudizio di qualita finale e dichiarato. I primi campioni del trainer reale sono piu lenti del runner isolato precedente e richiedono profiling/ottimizzazione dell'integrazione, non una dichiarazione di equivalenza con quel runner.
### Verifica integrata del 21 ottobre: 2.000 step reali

Eseguito il trainer headless reale (CUDA disabilitato in compilazione), seed comune verificato SHA256, resize 1920, cap 1.000.000, SH0, MCMC, schedule C++ risolta: start_refine 33, refine_every 7, stop_refine 1667. Completati training, checkpoint finale e tutte le 11 viste di evaluation, uscita 0 senza avviso di allocazioni HIP trattenute allo shutdown.

Qualità: PSNR 18.208645, SSIM 0.437029, LPIPS 0.577670. Riferimento CUDA storico: 18.189753 / 0.426636 / 0.602192. Prime viste confrontate visivamente: nessun collasso evidente. Le immagini GT dei due percorsi non sono bit-identiche; non si dichiara equivalenza numerica esatta.

Prestazioni CPU step, finestra 201–1900, 1700 campioni: media 328.7188 ms, mediana 297.3130, p90 597.0236. Il numero PerfBench 520.87 ms include evaluation terminale e non rappresenta il training ordinario. Ripresa del checkpoint finale, warmup loader 142 step: 186.45 ms/iter, attesa loader 0.06 ms. I precedenti 68.41 ms del runner privato NON rappresentano l'integrazione: topologia fissa, due viste e Adam nuovo. Obiettivo prestazionale del backend reale ancora non raggiunto.

Memoria: picco Core pool usato 577335508 byte, riservato 586527264; esclusi workspace nativi raster/LPIPS. Il picco hipMemGetInfo usato 6249742336 è contabilità globale del driver con memoria condivisa: non VRAM dedicata né memoria isolata del processo.

Artefatti locali: real-hip-historical-2000-oct21-r2, real-hip-resume-phase-oct21-r1, real-hip-integrated-comparison-oct21.json. Nessun nuovo test CUDA/macOS per questa tabella. Le tracce di fase iniziali sono confini CPU accodati; il costo asincrono del backward può essere addebitato al forward successivo. Aggiunta diagnostica HIP opzionale con sincronizzazione esplicita, ancora da misurare. Non usare gli eventi HIP per tempi: il driver ha già restituito durate negative nei riproduttori indipendenti.

### Porting integrato: Blob, Controller, MRNF, Bilateral e PPISP (21 ottobre)

Aggiunte e registrate cinque famiglie complete, arrivando a 18/21: Blob, Controller, MRNF, Bilateral e PPISP. Le tre famiglie ancora assenti sono Gsplat/3DGUT, Masks e Geometry; la condivisione HIP/Vulkan dello stesso modello senza duplicazione resta da realizzare. Non dichiarare completo il backend.

Blob: downsample CPU indipendente, immagine uniforme senza falsi picchi, sweep analitico con nove viste di supporto, picchi/bitmap confrontati con Vulkan. Controller: contratto dello slot esposizione, accumulo dei gradienti ripetuto, ReLU e gradiente opzionale. MRNF: tutti gli otto callback verificati, bounds ordinati sulla GPU, Gumbel denso/sparso deterministico, noise congelato/invisibile, decay, pruning e compattazione confrontati con CPU. Training MRNF reale da dataset fresco: 80 step, crescita 19.555 -> 30.000 gaussiane, checkpoint finale e uscita 0. Un checkpoint MCMC non puo essere ripreso come MRNF: il rifiuto strategy mismatch e corretto.

Bilateral: kernel del repository preparati in copie HIP isolate; affini CHW/HWC contro forward e gradienti analitici CPU, exposure-chroma contro Vulkan, TV/Adam/momenti/proiezione/offset contro CPU, percorsi in-place e trasferimenti limitati. PPISP: inizializzazione neutra, forward/backward contro Vulkan, differenze finite dei parametri, coordinate crop, backward in-place, regolarizzazione/Adam/project-mean CPU e gruppi/output opzionali. Correzione esposizione reale con Bilateral+PPISP: 24 step, crescita a 23.766 gaussiane, progetto salvato e uscita 0. La rete controller completa ha rivelato ulteriori ingressi NN legacy del Core: bias_add/bias_relu/relu sono ora nativi e testati; max_pool2d/adaptive_avg_pool2d sono in fase di collegamento/verifica. Non confondere il test del callback Controller con la rete completa nel trainer.

Core: Int32 -> Bool usa il kernel di conversione HIP, evitando un broadcast intero non supportato. Riduzioni native Bool e intere con accumuli interi, assi arbitrari e viste trasposte; verificati Int64 oltre 2^53 tramite descrittore nativo, identita vuote tramite facade e ingressi pubblici Bool sum_scalar/any e Int32 sum. Le limitazioni pubbliche condivise delle riduzioni Int64/any-all su Float32 non sono state cambiate.

Ottimizzazione MCMC sperimentale: la variante RGB multi-pixel prima copriva solo DensificationType::None. Estesa a MCMC sotto LFS_HIP_FAST_DENSIFICATION_PIXELS, senza cambiare i kernel CUDA/Metal/Vulkan originali. Tre step, 263 gaussiane, viste 19x17/64x48/1x1, gradienti piccoli e molto grandi: parametri identici e statistiche densificazione entro 2e-5 assoluto+relativo rispetto al modulo precedente. ABBA sul trainer reale, checkpoint finale da un milione, 220 step per processo e 142 di warmup: A1 187.802576 ms, B1 149.579704 ms, B2 149.563194 ms, A2 187.839890 ms. Media 187.821233 -> 149.571449 ms, -20.365%, speedup 1.256x. Attesa loader circa 0.06 ms in tutti i casi. La nuova variante non ha ancora la verifica completa di qualita a 2.000 step e resta esplicita/sperimentale; non dichiarare raggiunto il tempo Vulkan ottimizzato.

Diagnostica per fasi: 78 campioni di CPU steady_clock con sincronizzazione della coda, non eventi temporali HIP: pre-forward 1.678 ms, forward 64.715 ms, loss 11.519 ms, backward 106.181 ms, optimizer 0.640 ms. Nel backward, blend ~95.857 ms e proiezione ~7.840 ms su tutti i 220 step. La coda HIP condivisa puo includere lavoro del loader nelle finestre diagnostiche.

Un eseguibile di test terminato dopo un errore iniziale resta nello stato Windows di terminazione e mantiene mapping di DLL. GPU verificata sana tramite contratto Core indipendente. File mappati conservati con nomi archivio, build normale ripristinata; nessun riavvio e nessun processo di altre chat toccato. Il temporaneo CMAKE_PROJECT_INCLUDE per il nome alternativo Core e stato rimosso dalla configurazione locale.

### Oct 21 - real controller, geometry, masks and long MCMC verification

The complete PPISP controller network now finishes its real 24-step activation test (activation at step 5), saves its project and exits successfully. The missing legacy Core bias/ReLU and pooling entry points use native HIP operations and pass CPU references.

Geometry: all five callbacks are connected. Four loss operations pass weighted/unweighted and inactive-coverage cases against Vulkan, including gradients and diagnostic prefixes; independent CPU cosine and anchor projection references pass. Anchor scratch is bounded and released only after confirmed queue completion. No CUDA SDK/runtime is used. Masks: all five callbacks are connected; photometric, opacity and alpha penalties pass independent CPU references, while small/large/near-clipped/invalid mesh faces and point coverage match Vulkan, including inverse sample maps. Distorted mesh preparation exposed Core masked_fill; its HIP implementation preserves unselected raw storage and supports the seven facade dtypes. Additional dtype/broadcast tests are queued for the current rebuild.

A complete real MCMC candidate run (same seed and resolved schedule, 2,000 steps, max_width 1920, cap 1M, SH0, 11 evaluation views) finished with exit 0. CPU step window 201-1900, 1,700 samples: previous mean 328.718765 ms, candidate 213.357566 ms (-35.094%); medians 297.312950 -> 204.939900 ms. This whole-run result includes the evolving topology; the earlier resumed ABBA experiment isolated a 20.365% improvement. The two measurements are not interchangeable.

Quality averages: PSNR 18.208645 -> 18.290804, SSIM 0.437029 -> 0.439301, LPIPS 0.577670 -> 0.576331. All 11 views were inspected in side-by-side sheets including the historical CUDA render; no overall visual degradation was observed. This is evidence for this dataset/configuration, not a guarantee for every scene. Core pool peaks are unchanged (used 577,335,508 bytes; reserved 586,527,264 bytes). Driver global shared-memory accounting is not isolated process VRAM. Full wall time, including evaluation, was 1009.409 -> 815.698 seconds and must not be presented as ordinary training-step time.

The validated MCMC multi-pixel path is now the HIP default when eligible; LFS_HIP_FAST_DENSIFICATION_PIXELS=0 explicitly selects the former diagnostic path. Original CUDA, Metal and Vulkan kernels are unchanged. The candidate did not yet beat the historical optimized Vulkan training baseline.

3DGUT: the complete application adapter and native entry points have been written for RGB, D, ED, RGB_D, RGB_ED and RGB_D_N, SH0-3, camera rays, geometry gradients, density/error/edge/share accumulation, resource errors and cache lifetime. One/three/four/eight channel kernels and geometry features are being compiled and tested; do not count this family as validated until those tests pass. Checkpoint-resume and the consolidated serial suite remain pending. Same-device HIP/Vulkan shared storage, physical GPU inventory unification and broader platform validation also remain open.

Per-view qualification of the long comparison: PSNR improves on 6/11 views, SSIM on 7/11, LPIPS on 5/11; the respective averages improve. Individual views are not identical: worst PSNR delta -0.111305 dB, worst SSIM delta -0.00305748, worst LPIPS delta +0.00511187. These small view-level fluctuations and the absence of an overall visual degradation must not be described as bitwise parity or improvement on every image. Full per-view deltas and binary hashes are retained in the local mcmc-pixels-full-quality-summary-oct21.json artifact.


### Verifica di integrazione, 21 ottobre: inventario, SH3 e memoria esterna

- `full-integration-oct21-r7-results.json`: 33 contratti seriali superati sui binari ricompilati. Gsplat include ora quattro confronti aggiuntivi su ritagli SH2 (pinhole, fisheye, equirettangolare e thin-prism), con output e gradienti confrontati con Vulkan.
- `fast-sh3-application-smoke-oct21-r4` e `gsplat-application-smoke-oct21-r3`: trainer reali, 64 step, SH0-3, refinement, cap 30.000 e checkpoint; entrambi exit 0 e senza l'avviso di allocazioni HIP trattenute alla chiusura. La causa era lo scratch statico della ricodifica Q16: lo slot HIP viene ora svuotato attraverso il pre-shutdown hook; il ramo preesistente degli altri backend resta invariato.
- L'inventario C++ verifica la Radeon HIP/Vulkan come `pci:0000:c8:00.0` e mantiene distinta la GeForce `pci:0000:c6:00.0`. Le scelte conservano i propri UUID API. Superati 13 test Python di inventario, comprese collisioni, identificatori ambigui e memoria condivisa.
- `external-memory-contract-oct21-r1.log`: nuova API del runtime per importare memoria esterna, 128 cicli bidirezionali reali HIP/Vulkan sulla stessa allocazione, ownership Vulkan EXTERNAL con completamenti host, controlli di range, piattaforma, contesto, handle rilasciati e allocazioni normali successive. Questo prova il seam nativo, non ancora la pubblicazione condivisa del modello nel viewer.
- Il collegamento Core degli import e la contabilita senza duplicazione del pool sono in compilazione/verifica. La pubblicazione del modello nel viewer resta da collegare con le barriere di ownership e i reader; non va descritta come completata sulla base del solo test nativo.

Riferimento della sincronizzazione e import: https://rocmdocs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/external_interop.html . I completamenti host sono un percorso supportato; non si assume che i semafori esterni funzionino su un driver che li ha rifiutati nelle prove precedenti.


### Integrazione HIP/Vulkan, verifica locale del 21 ottobre

La suite `full-integration-oct21-r12-results.json` supera 36 contratti seriali sugli stessi binari: include le 21 famiglie del trainer, import esterno Core, coordinamento concorrente e pubblicazione condivisa del viewer. Il confronto MCMC ottimizzato/non ottimizzato usa due politiche dello stesso modulo corrente; il prodotto non richiede moduli HIP precedenti.

L'import Core conserva il proprietario Vulkan fino all'ultima vista Tensor ed esclude i byte importati dal pool HIP. Il contratto del viewer verifica scritture bidirezionali GPU, offset, capienza, sostituzione del modello con lease ancora vivo, 32 alternanze HIP/Vulkan e il blocco di una chiamata nativa del trainer durante la registrazione del consumer. Il coordinatore protegge registrazione e invio dei comandi e alterna i ruoli in contesa. Il renderer usa questi scope solo nelle build HIP; le implementazioni degli altri backend non sono sostituite.

Il caricamento nell'app reale conferma `HIP/Vulkan shared storage active`. La prova GUI da 2.000 step usa il CKPT estratto senza modifiche dal progetto iniziale comune (SHA-256 del CKPT `55485e15b62870b8a32d960800693f299a49d58233b174b0a9a6db7135c6cd2e`), dataset locale, undistort, larghezza massima 1920, SH0, cap 1M, cache CPU e steps_scaler=1. Il percorso GUI dei checkpoint richiede il CKPT separato; il progetto originale e il suo checkpoint restano immutati. Il test ha superato 1.200 step e la crescita a 1M, con due cambi di camera e un salvataggio concorrente; il completamento e la chiusura non sono ancora verificati a questo punto. I tempi GUI con viewport attivo non sostituiscono la baseline headless.

Restano aperti l'import DLPack di allocazioni ROCm esterne, le operazioni Tensor generali ancora esplicitamente non supportate, le verifiche delle altre piattaforme e il superamento prestazionale della baseline Vulkan. I contratti superati non dimostrano assenza universale di regressioni.


Aggiornamento della prova GUI: completati 2.000 step, crescita a 1M, valutazione e checkpoint finale (generation 4); due cambi camera e salvataggio concorrente riusciti. Questa prova e' esclusivamente di integrazione: il caricamento GUI del CKPT riimporta il dataset e produce una suddivisione train/test e una preparazione delle camere differenti dal progetto conservato nel protocollo headless. Le metriche GUI non entrano nella tabella comparativa storica.

Il controllo reale successivo ha rilevato due lacune ancora aperte: la selezione richiedeva `filter_points`, non ancora portato, e la chiusura GUI segnalava allocazioni HIP vive. Aggiunti adattatore nativo del filtro, generazione dai predicati Core condivisi e contratto per etichette, offset/stride, tabelle vuote/alias, forme invertite, tre proiezioni e input temporanei accodati. Compilazione e ripetizione GUI in corso; lo shutdown e' in diagnosi. Non descrivere l'integrazione complessiva come conclusa prima di questi controlli.


Ulteriore verifica GUI: la cache globale dell'overlay dei frustum tratteneva 82 camere e 164 piccole allocazioni HIP (6.232 byte). Il rilascio e' ora esplicito alla chiusura solo quando il backend attivo e' HIP; con diagnostica abilitata, le sessioni GUI successive non segnalano piu' risorse HIP vive. CUDA e gli altri backend non entrano nel nuovo ramo.

`point-filter-contract-oct21-r2.log` verifica anche update_labels e and_live: modi Add/Remove/Replace, lock, categorie, indici ripetuti, alias, stride e offset. `point-filter-contract-oct21-r3.log` aggiunge l'istogramma di gruppi UInt8/Bool con bin Int32, bin zero escluso, input vuoto e guardie sui bin adiacenti. Questi contratti passano su CPU e HIP; la suite r14 resta il riferimento di 37 casi antecedente alle ultime aggiunte. Il successivo test dell'app ha isolato un confronto Int32 NotEqualTensor mancante (op 30); aggiunta implementazione nativa dei predicati Int32/Int64/UInt32/Float16, attualmente in compilazione con test di precisione intera e speciali floating point. Non considerare ancora passata la selezione GUI finale.


Verifica conclusiva della prima integrazione GUI: `full-integration-oct21-r18-results.json` supera 37 contratti seriali con gli ultimi binari. I predicati tipizzati sono confrontati con CPU: Int32 oltre 2^24, Int64 oltre 2^53, UInt32 e FP16 con NaN/infinito. Le promozioni scalari pubbliche restano quelle della libreria; non sono state ampliate le API comuni per adattarle al test.

Nella sessione GUI r8 sul modello da 1M: selezione rettangolare 718.183 elementi; rimozione 336.521; undo 718.183; redo 336.521; intersezione con la regione rimossa zero; aggiunta di quella regione 381.662; clear zero. Ogni modifica e' verificata con selection_get, inclusi indici e conteggi. Il nodo camera correttamente esclude il modello dal filtro; il nodo Model lo include. Controllato il viewport attraverso catture della finestra. L'uscita normale non riporta errori HIP o allocazioni trattenute. Il modulo nativo del trainer mantiene SHA-256 `be44344559a43b4019ecb9efa3de954872e4c3532fbffa0c5401b32f33d4f936`: i kernel del trainer non sono stati modificati per queste correzioni del viewer.

Il lavoro prosegue su project_points e mark_points_2d, necessari alle selezioni riferite alle camere del dataset e alle regioni 2D. Adattatori scritti riutilizzando PointProjection/pointInRegion/rounded math condivisi; contratto aggiunto per tre modelli di camera, trasformazioni, visibilita, indici fuori range, code, stride, regioni disco/rettangolo/poligono/dischi e valori non finiti. Questi due nuovi port sono ancora in compilazione: non sono inclusi nei 37 casi r18.

Verifica successiva del 21 ottobre: la suite r20 supera 38 contratti seriali, inclusi project_points e mark_points_2d. Nella sessione GUI r9 sulla camera 0 del dataset, rettangolo e poligono equivalente selezionano gli stessi 7.906 indici; il pennello seleziona 2.563 indici, tutti contenuti nel rettangolo. Clear e chiusura normale passano senza risorse HIP trattenute. Questi risultati riguardano integrazione e correttezza, non nuovi guadagni di training.

Esperimento separato, non abilitato di default: LFS_HIP_FAST_SPLIT_MCMC_STATS=1 divide il backward RGB dalle statistiche di densita/errore MCMC. L'obiettivo e' ridurre i registri vivi; due passaggi implicano letture aggiuntive. Occorrono contratto numerico e ABBA del trainer reale prima di qualsiasi accettazione prestazionale. La baseline primaria resta 213,358 ms/iterazione nel confronto headless da 2.000 step.

Esito dell'esperimento split MCMC: scartato e rimosso dal generatore di produzione. Sei casi numerici passano con due livelli di gradiente, ma ABBA reale dallo stesso checkpoint 1M (2.000 -> 2.220, 142 warmup, 78 steady) misura A1 145,970274 / B1 164,117023 / B2 164,904163 / A2 146,319896 ms. Medie: 146,145085 contro 164,510593 ms, peggioramento +12,5666%. Attesa loader steady 0,059-0,064 ms. Non e' una nuova misura del protocollo completo da 2.000 step. Pool picco A1/B1/B2 618.711.288 byte, A2 610.416.892 byte; conteggi allocazioni Core, non VRAM fisica per processo. Loss finale circa 0,14714 in tutte le prove; non sostituisce una valutazione qualitativa completa. Artefatti split-stats-abba-summary-oct21.json, split-stats-contract-oct21-results.json, snapshot del generatore e DLL sperimentale nella cartella locale rocm-tensor-study. Nessun backend esistente modificato da questo esperimento.

Controllo supplementare sullo stesso checkpoint finale (SHA-256 f0894923c0f26b35756ec9fd3f96b6d6907664be15b7481f8621dc7cfed5030b): V-H-H-V, resize 1920 SH0, cap 1M, 2000 -> 2220, 142 warmup/78 steady. HIP 144,170741 e 145,559773 ms; Vulkan di questo branch 256,554869 e 262,056609 ms. Questa versione Vulkan non e' il riferimento ottimizzato piu' recente gia' registrato, e la topologia e' piena fin dall'inizio: non sostituisce la baseline storica 114,591 ms + loader, non dimostra raggiunto l'obiettivo prestazionale, non e' il test completo con crescita da 2.000 step. I quattro run hanno lo stesso hash iniziale e completano senza errori; qualita' completa non rivalutata in questo controllo. Riepilogo same-checkpoint-hip-vulkan-oct21-summary.json.

Nuova ipotesi circoscritta al resume HIP: MCMC inizializza statistiche transitorie anche riprendendo dopo stop_refine, ma normalmente le libera solo allo step esatto stop_refine. Il prototipo LFS_HIP_STOPPED_MCMC_CLEANUP=1 esegue la stessa transizione scaduta su HIP alla prima post_backward utile, senza cambiare rumore, parametri o momenti e senza cambiare gli altri backend. ABBA reale in corso; nessun guadagno ancora accettato. Il ciclo fresh che attraversa stop_refine non e' oggetto di questa ottimizzazione.


Congelamento del 21 ottobre su richiesta dell'utente: prove fermate, valutazione qualitativa del candidato resume incompleta. Riferimento esaustivo: rocm-historical-handoff-2026-10-21.it.md nella documentazione development. Il prototipo resume e il suo contratto non sono accettati; il contratto non e' stato compilato. Si procede ora a studio e allineamento locale all'ultimo upstream/dev, preservando questo stato storico.
