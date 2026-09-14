# Where the fly brain comes from

Everything here is public and openly licensed. Nobody is training on a secret
dataset — the "fly plays Doom" projects all start from the same two or three
releases, and the differences between them are almost entirely in what they do
*after* loading the graph.

---

## 1. MaleCNS v1.0 — the complete male *Drosophila* CNS

The dataset behind the September 2026 announcement: whole male central nervous
system (brain **and** ventral nerve cord), including the descending and ascending
neurons that connect the two.

| | |
|---|---|
| Scale | ~166,000 neurons, ~125 million chemical synapses |
| Contents | brain, VNC, sensory afferents, descending neurons, motor neurons |
| Also | 262 sex-specific and 114 sexually dimorphic cell types |
| License | CC-BY 4.0 |
| Paper | *Cell*, doi:10.1016/j.cell.2026.08.015 (preprint bioRxiv 2025.10.09.680999) |
| Announcement | research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain |

### 1a. Flat connectome (what this repo wants)

A pre-computed, "flat" edge list — no segmentation volumes, no skeletons, just
neurons and their synapses. ~1.1 GB total:

```
gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/
    body-annotations-male-cns-v1.0-minconf-0.5.feather       13 MB   cell types, sides, ROIs
    body-neurotransmitters-male-cns-v1.0.feather             42 MB   transmitter identity
    connectome-weights-male-cns-v1.0-minconf-0.5.feather    1.1 GB   the edges
```

The same objects are served over HTTPS from
`https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/<name>`,
which is what `tools/fetch_connectome.py` uses. `gsutil cp -r gs://…` works too.

```bash
python -m flybfm.tools.fetch_connectome --list          # show the files and sizes
python -m flybfm.tools.fetch_connectome --download      # ~1.2 GB into data/malecns/
```

Then point the brain at them:

```bash
export FLYBFM_CONNECTOME=data/malecns/connectome-weights-...feather
export FLYBFM_ANNOTATIONS=data/malecns/body-annotations-...feather
python -m flybfm probe --brain malecns
```

`brain/connectome.py::load_malecns_flat(path, annotations_path)` does the rest:
it sniffs the column names (they move between releases), joins the annotations,
reads transmitter identity to sign each edge, and maps synapse counts to
synaptic weights.

### 1b. neuPrint (what you probably actually want)

Downloading 1.1 GB to look at six neurons is silly. neuPrint gives you the graph
as a queryable database.

```python
from neuprint import Client, fetch_neurons, fetch_adjacencies

client = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0",
                token=os.environ["NEUPRINT_TOKEN"])     # free account
neurons, synapses = fetch_neurons("DNge104")            # a real descending neuron
out, info = fetch_adjacencies("DNge104")                # what it talks to
```

* Web UI: https://neuprint.janelia.org — dataset `male-cns:v1.0`
* Python: `pip install neuprint-python`
* R: `neuprintr`
* Skeletons + transforms: `navis` with `flybrains`
* Support: the neuPrint Google group

`brain/connectome.py::load_neuprint(...)` wraps this for a cell-type list, so you
can pull exactly the visual → descending → (optionally) motor subgraph you need.

### 1c. Everything else in the release

| Asset | Where | Use |
|---|---|---|
| Cell Type Explorer (Reiser lab) | reiserlab.github.io/celltype-explorer-drosophila-male-cns | find cell types by name/hemisphere |
| Neuroglancer scene | `gs://flyem-male-cns/v1.0/male-cns-v1.0.json` | browse the EM, check your wiring by eye |
| EM volumes | N5 `gs://flyem_cns_z0720_07m_dvidcoords_n5` (8 nm) | raw data if you want to re-annotate |
| Proofread segmentation | `gs://flyem-male-cns/v1.0/segmentation` | skeletons, meshes |
| ROIs (brain + VNC neuropil) | `gs://flyem-male-cns/v1.0/rois/fullbrain-roi-v4`, `…/malecns-vnc-neuropil-roi-v0` | restricting edges to a neuropil |
| Browsing images | `gs://flyem-male-cns/v1.0/em/em-clahe-jpeg` | quick look without a volume client |
| Clio | Janelia's in-house analysis tool | lineage/type exploration |

---

## 2. Female FlyWire — the dataset most published models use

139,255 neurons of the adult **female** brain, no VNC. This is what almost every
2024–2025 whole-brain simulation actually runs on, including the ones widely
reported as "the fly brain".

* Codex (interactive): https://codex.flywire.ai
* Annotations (CC-BY 4.0): https://github.com/flyconnectome/flywire_annotations
* Bulk dump used by the modelling papers: Zenodo record `10.5281/zenodo.10676865`
* Cell types: ~8,400, with hemibrain-consistent naming

Practical difference for this project: **no VNC and no motor neurons**, so the
brain→wing interface has to be invented. MaleCNS v1.0 removes that problem, which
is the single best reason to use it for a flight-control project.

## 3. Body and vision (if you want the fly to have a *body*)

| Project | What it gives you | License |
|---|---|---|
| **FlyGym / NeuroMechFly v2** — `NeLy-EPFL/flygym` | biomechanical fly: wings, halteres, legs, adhesion; 2.x is a rewrite, ~2× real time on CPU | Apache-2.0 |
| **FlyVision** — `TuragaLab/flyvis` | connectome-constrained vision (lamina → medulla → lobula plate), trained to reproduce fly optomotor behaviour | MIT-ish, check repo |
| **MANC** (male VNC) | the motor side: descending neuron → wing/haltere/leg motor neurons | via MaleCNS or the MANC release |

A useful architecture note from NeuroMechFly v2: it deliberately splits
brain and VNC into two simulators with an explicit descending/ascending
interface. That is exactly the interface this repo reads out, so the two are
compatible by construction.

## 4. Prior whole-brain *simulations* (models, not data)

| Project | What it is |
|---|---|
| Shiu et al., *Nature* 2024 — `philshiu/Drosophila_brain_model` | whole-brain LIF model of FlyWire; the reference implementation of "the brain as a connectome-constrained RNN" |
| `nftechie/doomfly` | 166.7k neurons, 25.6M connections, 124M synapses; R1-R6 (3,335) and R8 (811) driven by game state; reward injected into **two** PPL101 dopamine neurons. The source of the widely quoted "3,000 runs, no learning" |
| `dohun1214/flybrain` | DN bridge between a whole-brain model and FlyGym |
| `erojasoficial-byte/fly-brain` | FlyWire LIF (138,639 neurons, 15M synapses, 5 kHz) on a MuJoCo NeuroMechFly v2 body, with Hebbian plasticity |

Reading their source is worthwhile and sobering: the connectome supplies a
*topology*, every dynamical parameter is guessed, and the part that plays the
game is a decoder trained by ordinary machine learning.

---

## 5. What the data does *not* contain

* **Synaptic weights.** The connectome gives synapse *counts* per connection.
  Count is a weak proxy for efficacy; gating, receptor type, distance from the
  soma and neuromodulatory state all matter and none of them is in the file.
  Every whole-brain model therefore guesses a count→weight mapping. In this repo
  the mapping is `w = count**0.5` scaled to [0.05, 3.0] (`weight_fn` argument of
  `load_malecns_flat`) and it is *the* unidentifiable parameter of the exercise.
* **Cell-type-specific dynamics.** Membrane time constants, thresholds and
  adaptation are known for a handful of neurons and assumed for the rest.
* **Neuromodulatory state.** Dopamine, octopamine and serotonin change the
  circuit's behaviour; the connectome is one snapshot of the wiring.
* **Anything behavioural.** The graph does not say what a neuron *means*.
  Meaning comes from anatomy (which neuropil it targets) and from experiments,
  which is why the readout below is anchored to named, experimentally studied
  descending neurons rather than to a clustering algorithm.
