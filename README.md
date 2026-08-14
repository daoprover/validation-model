# DAOprover Bitcoin Graph Classifier

This project builds graph neural-network classifiers over Bitcoin address and
transaction activity. The current supervised task maps each graph to either
`white` or `anomaly`.

## Scope and safety boundary

This model is an anomaly detector, **not a Bitcoin consensus validator**. Its
prediction must never replace validation performed by Bitcoin Core. It does not
verify UTXOs, scripts, signatures, amounts, locktimes, sequence rules, witnesses,
or double-spend rules.

## Graph representation

Each transaction is represented explicitly:

```text
input address -> transaction -> output address
```

This preserves repeated transactions and avoids inventing an input-to-output
mapping that does not exist in Bitcoin. Node features are always ordered as:

1. `total_sent`
2. `total_received`
3. `num_transactions`
4. `total_fees`
5. `avg_transaction_value`

Edge features are always ordered as `amount`, `fee`, `size`, and `timestamp`.
Missing values are zero-filled; random fallback features are not used. Count and
value features receive a deterministic `log1p` scale before entering the model,
while Unix timestamps are divided by one billion so raw magnitudes cannot
dominate attention scores.

## Setup

Python 3.10 or later is required.

```bash
cd /path/to/daoprover/validation-model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run every command below from the `validation-model` directory. Paths in
`models/gnn/gat/hyperparams.json` are resolved relative to the current working
directory.

## Preprocessing

Preprocessing has two stages:

1. fetch address histories and build labeled GEXF graphs in
   `data/assets/graphs`;
2. split those graphs into leakage-safe train, validation, and test directories.

The generated graph data is intentionally not committed to Git.

### 1. Configure preprocessing

Edit the `dataset` section of `models/gnn/gat/hyperparams.json`:

```json
{
  "dataset": {
    "raw_dataset": "./data/assets/graphs",
    "raw_dataset_csv": "./data/BitcoinHeistData.csv",
    "train": "./data/assets/train",
    "valid": "./data/assets/valid",
    "test": "./data/assets/test",
    "start_block": 400000,
    "end_block": 400010,
    "step": 2,
    "txs_per_block": 2
  }
}
```

`start_block` is inclusive and `end_block` is exclusive. The indexer visits
blocks using `range(start_block, end_block, step)` and selects at most
`txs_per_block` unique addresses from each block.

Start with the small range above. The repository's wider `400000..500000`
range can attempt roughly 100,000 address downloads and may take several days.
The indexer intentionally waits four seconds between addresses.

Indexing uses the public `blockchain.info` API, so an internet connection is
required. API failures are logged and the affected address is skipped.

### 2. Index labeled anomaly addresses

Place the labeled CSV at the configured `raw_dataset_csv` path. The preferred
format is:

```csv
address,label
1ExampleAddress,ransomware
1AnotherAddress,white
```

If the CSV contains columns named `address` and `label`, their order does not
matter. For compatibility with BitcoinHeistData-style files, the fallback is
column 0 for the address and column 9 for the label.

Run:

```bash
python cli.py index-marked-addresses --verbose 1
```

The original label is stored in each GEXF file. During binary training, every
label other than the exact string `white` is mapped to `anomaly`.

### 3. Index normal addresses from Bitcoin blocks

Run:

```bash
python cli.py index-white-addresses --verbose 1
```

These graphs are saved with the `white` label. If an address already exists and
a later labeled CSV assigns it an anomaly label, rerunning
`index-marked-addresses` updates the stored label without downloading the graph
again.

You may run either indexing command first. To train a binary classifier, the
final dataset must contain both `white` and non-`white` graphs.

### 4. Inspect the raw graph output

```bash
find data/assets/graphs -type f -name '*.gexf' | wc -l
python - <<'PY'
from collections import Counter
from pathlib import Path
import networkx as nx

counts = Counter()
for path in Path("data/assets/graphs").glob("*.gexf"):
    graph = nx.read_gexf(path)
    label = graph.graph.get("name")
    counts["white" if label == "white" else "anomaly"] += 1
print(dict(counts))
PY
```

Do not continue until both classes have usable samples. A heavily imbalanced
dataset is supported through class-weighted loss, but it still needs enough
independent anomaly groups for validation and testing.

### 5. Create train, validation, and test splits

The splitter copies files by default, preserving the raw graph directory:


```bash
python dataset/split.py data/assets/graphs data/assets --seed 42
```

Expected output resembles:

```text
train=700 valid=150 test=150
```

The exact counts can differ because graphs sharing any transaction are kept in
the same partition. This prevents transaction leakage between train,
validation, and test data. Legacy address-to-address graphs without explicit
`transaction:*` nodes are rejected and must be regenerated with the current
indexer.

Use `--move` only if you intentionally want to remove files from the raw graph
directory:

```bash
python dataset/split.py data/assets/graphs data/assets --seed 42 --move
```

### 6. Verify the prepared directory layout

```text
data/
├── BitcoinHeistData.csv
└── assets/
    ├── graphs/       # raw generated graphs
    ├── train/        # training graphs
    ├── valid/        # checkpoint-selection graphs
    └── test/         # final evaluation graphs
```

Check the split counts:

```bash
for split in train valid test; do
  printf '%s: ' "$split"
  find "data/assets/$split" -type f -name '*.gexf' | wc -l
done
```

If preprocessing is rerun with a different seed or source dataset, empty the
old split directories first or write to a new destination. The splitter does
not delete stale files already present in the destination.

## Training and evaluation

After preprocessing succeeds:

```bash
python cli.py train-gat
python cli.py test-gat
```

Training uses a fixed seed, class-weighted loss, and a held-out validation
subset. Graphs connected by any shared transaction are kept in the same split;
training fails if a leakage-free validation set cannot be formed. The `best`
model is selected by sample-weighted validation loss. The test directory remains
separate and evaluation rejects shared transactions between train and test.

Evaluation treats `anomaly` as the positive class and reports accuracy, balanced
accuracy, anomaly precision/recall/F1, ROC AUC, average precision, and a confusion
matrix. Interactive plots are disabled by default and can be enabled with
`testing.show_plots`.

Checkpoints produced by the old random-feature/address-to-address pipeline are
not compatible with this corrected representation and must be retrained.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover deterministic feature extraction, transaction accounting,
preservation of repeated relationships, labels, and GEXF round trips.
