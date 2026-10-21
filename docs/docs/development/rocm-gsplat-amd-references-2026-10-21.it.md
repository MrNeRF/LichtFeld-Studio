# Riferimenti AMD GSplat per il backend HIP nativo

Consultazione del 21 ottobre 2026, durante l'allineamento locale a upstream/dev.
Questa raccolta non aggiunge librerie, pacchetti Python o dipendenze Torch/PyTorch
a LichtFeld. Serve per studiare algoritmi, layout e verifiche numeriche dei kernel.
Nessuna delle modifiche elencate sotto e' stata applicata automaticamente.

## Fonti e limiti del confronto

- [Repository AMD-Ecosystem/gsplat-amd](https://github.com/AMD-Ecosystem/gsplat-amd), branch predefinito rilevato `release/1.5.3b2`.
- [Documentazione ufficiale AMD](https://rocm.docs.amd.com/projects/gsplat/en/latest/).
- [Profiling](https://rocm.docs.amd.com/projects/gsplat/en/latest/reference/profile-gsplat-library.html): confronta separatamente forward, backward e memoria degli intermedi, con e senza packed/sparse gradient.
- [API](https://rocm.docs.amd.com/projects/gsplat/en/latest/reference/gsplat-api-reference.html): descrive il compromesso packed/dense e il costo degli offset per il sort segmentato.
- [Benchmark di qualita'](https://rocm.docs.amd.com/projects/gsplat/en/latest/reference/benchmark-evaluation.html).

Il progetto AMD usa PyTorch e documenta principalmente Linux/Instinct MI300X.
I suoi FPS, tempi e consumi non sono una baseline per Windows/Radeon 890M.
Gli spunti utili vanno tradotti nel modello Tensor, nelle code e nell'ownership
gia' presenti in LichtFeld, senza importare ATen, autograd, allocator o SparseAdam.
In particolare GSplat EWA e il percorso 3DGUT non coincidono con FastGS: prima
di portare un'idea occorre identificare il kernel locale e il lavoro equivalente.

Snapshot delle risposte GitHub API, corpi PR e diff dei file conservati negli
artefatti privati `dev-alignment/gsplat-amd-*.json`. Stato osservato: 9 PR aperte;
sono stati recuperati i diff di #17, #14, #24, #32, #11, #10, #21 e #22.
Esaminati in particolare la scelta wave/riduzioni di #17, le guardie numeriche
di #24 e i cambi proiezione/intersezioni/forward/SH di #32. Il nuovo backward
multi-tile di #32 richiede ancora un confronto dettagliato con i nostri kernel;
la raccolta dei diff non equivale alla validazione completa delle PR.
I risultati dichiarati dagli autori non sono risultati misurati su LichtFeld.

## PR da consultare

| Fonte | Stato osservato e revisione studiata | Spunto e cautela |
| --- | --- | --- |
| [#17 RDNA3.5 / wave32](https://github.com/AMD-Ecosystem/gsplat-amd/pull/17) | Aperta; head `20f38d5a438a09723e7db79cef82ac6dc5614e34` | Individua assunzioni wave64 in riduzioni, partizioni e conteggi. Rischio di gradienti errati anche con build riuscita. Audit di maschere, lane inattive, wave parziali e numero di wave per blocco; gfx1151 citato dalla PR non e' il nostro gfx1150. |
| [#14 ROCm 7.2 / RDNA](https://github.com/AMD-Ecosystem/gsplat-amd/pull/14) | Aperta; head `202fa4dbd40bd5c028eae01486c6d6d37f5b69db` | Variante che combina due wave in LDS per tile 8x8. L'autore dichiara limiti di qualita' del codice e incompatibilita' wave64: riferimento sperimentale, non soluzione da adottare globalmente. |
| [#32 sincronizzazione e ottimizzazioni](https://github.com/AMD-Ecosystem/gsplat-amd/pull/32) | Aperta; head `a14c194cb8d22224cb7268d268cc8479c4d35d15` | Backward wave32 multi-tile, pressione VGPR/LDS, forward float4 con prefetch, enumerazione cooperativa delle intersezioni, eliminazione di riduzioni/atomici solo quando la proprieta' degli indirizzi e' dimostrata, riuso della base SH fra canali. Risultati dichiarati su gfx1250 non trasferibili direttamente a RDNA3.5. |
| [#24 reciprocali e SH degeneri](https://github.com/AMD-Ecosystem/gsplat-amd/pull/24) | Mergiata 30 settembre; head `389f152be501106fe85e92dc85eadea0bae040e4` | Scale collassate e direzione camera-Gaussian nulla: verificare output e gradienti finiti, con stessa convenzione nel forward e backward. Una modifica di clamp cambia la funzione: richiede casi numerici e controllo dei valori ordinari. |
| [#10 overflow intersezioni](https://github.com/AMD-Ecosystem/gsplat-amd/pull/10), [#11 alternativa](https://github.com/AMD-Ecosystem/gsplat-amd/pull/11) | #10 mergiata 26 febbraio; #11 aperta | Controllare offset/prodotti int32 prima del lancio. Una guardia controllata non equivale al supporto di scene con indici a 64 bit. Non importare TORCH_CHECK: usare errori e limiti del backend locale. |
| [#21 GLM e hipify](https://github.com/AMD-Ecosystem/gsplat-amd/pull/21) | Aperta | Evitare che la trasformazione dei sorgenti modifichi header di terze parti. I generatori locali devono produrre output privati senza riscrivere SDK/deps. |
| [#22 architettura esplicita](https://github.com/AMD-Ecosystem/gsplat-amd/pull/22) | Aperta | Una scelta esplicita di architettura deve precedere la discovery e non degradare silenziosamente a un'altra GPU. Studiare il principio; non aggiungere variabili o strumenti PyTorch alla build LichtFeld. |

## Priorita' da verificare dopo l'allineamento

1. **Correttezza wave e degenerazioni.** Confrontare prima le assunzioni locali
   gia' validate con i rischi di #17/#24: SH0-3, Gaussian al centro camera,
   scale estreme, tile incompleti e riduzioni con lane inattive. Non riaprire
   esperimenti gia' scartati senza una differenza concreta identificata.
2. **Intersezioni sbilanciate.** La #32 distribuisce il rettangolo di tile di un
   Gaussian sull'intera wave nel secondo passaggio. Valutare l'equivalente
   locale, preservando ordine, chiavi, offset e casi vuoti. Misurare numero di
   intersezioni e distribuzione del lavoro, oltre al tempo del solo kernel.
3. **Ownership e riduzioni.** La #32 evita riduzioni quando le lane hanno indici
   distinti e usa store solo con un unico proprietario globale. L'ottimizzatore
   locale accumula anche gradienti preesistenti: uno store al posto di += non e'
   automaticamente corretto. Verificare capture, camere multiple e regularizer.
4. **LDS/VGPR e batching.** Confrontare layout float4/prefetch e tile multipli
   con le varianti gia' provate. Richiedere disassembly/risorse del compilatore
   e misure sulla 890M. La stessa PR descrive regressioni del percorso float4
   su wave64: selezione per capacita' e fallback verificato sono essenziali.
5. **SH.** Verificare se il calcolo della base e' gia' condiviso tra RGB nel
   nostro percorso. L'idea della #32 non offre vantaggio se il lavoro e' gia'
   fuso; SH0 resta il confronto principale, SH1-3 la copertura aggiuntiva.
6. **Memoria.** Confrontare intermedi attivi/packed e gradienti densi soltanto
   nei punti che dominano memoria o traffico. Non cambiare semantica Adam,
   condivisione HIP/Vulkan o cache immagini per imitare un allocator PyTorch.

Accettazione: prima contratti numerici e ownership, poi benchmark seriali
ABBA; infine protocollo comune fresh 2.000 step, undistort/1920/SH0/cap1M,
qualita' per vista e controllo visivo. Non modificare CUDA, Metal o Vulkan per
adattare una variante HIP. Nessun fast-math globale basato soltanto su una PR.

## Aggiornamento Adrenalin durante il lavoro

L'utente ha segnalato l'aggiornamento Adrenalin il 21 ottobre. Lettura effettiva:

- Radeon 890M: driver Windows `32.0.32015.7029`, data riportata 18 ottobre 2026.
- NVIDIA RTX 5060 Ti: driver Windows `32.0.16.1742`, data 30 settembre 2026.
- `C:/Windows/System32/amdhip64_7.dll`: FileVersion `10.0.3683.0`, ultima
  modifica riportata 18 ottobre 2026. Il nome del file non determina la sua
  versione effettiva.

La build in corso continua a usare lo SDK privato precedente; aggiornare
Adrenalin non significa aver aggiornato il compilatore o lo SDK scelto da CMake.
Salvare hash/versioni dei moduli effettivamente caricati prima dei prossimi
test. Il primo run con il nuovo driver e' una nuova condizione sperimentale:
non attribuire differenze rispetto allo storico al solo codice. Ripetere i
contratti e una baseline di controllo HIP/Vulkan con lo stesso driver prima
di quantificare il guadagno di nuove ottimizzazioni.
