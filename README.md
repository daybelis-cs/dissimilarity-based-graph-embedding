# Dissimilarity-based-graph-embedding
This project implements a framework in Python for processing mined graph patterns and generating dissimilarity-based embeddings for supervised classification. The framework loads previously mined frequent graph patterns, extracts pattern subsets (closed and maximal graph patterns), and generates training and evaluation matrices with dissimilarity-based graph embedding across 5 folds.

The input data is organized according to the mining algorithm and dataset category.

```dissimilarity-based graph embedding/
│
├── run.py
├── utils.py
│
├── FOLDS/
│   ├── Coenen/
│   │   ├── 1/
│   │   ├── 2/
│   │   └── 3/
│   │   └── 4/
│   │   └── 5/
│   │
│   └── PTC/
│       ├── 1/
│       ├── 2/
│       ├── 3/
│       └── 4/
│       └── 5/
│
├── GAGP/
│   ├── Coenen/
│   │   ├── 1/
│   │   ├── 2/
│   │   └── 3/
│   │   └── 4/
│   │   └── 5/
│   │
│   └── PTC/
│       ├── 1/
│       ├── 2/
│       ├── 3/
│       └── 4/
│       └── 5/
│
├── LAGP/
│   ├── Coenen/
│   │   ├── 1/
│   │   ├── 2/
│   │   └── 3/
│   │   └── 4/
│   │   └── 5/
│   │
│   └── PTC/
│       ├── 1/
│       ├── 2/
│       ├── 3/
│       └── 4/
│       └── 5/

```

## Datasets
The input data wo collections of graph datasets:
- **PTC**: The PTC (FM, FR, MM, and MR) datasets were obtained from the Network Repository, a collection of network datasets available for research purposes.
- **Coenen**: These are synthetic datasets generated using the Coenen image generator and subsequently converted into graph representations. 

The original datasets are not included in this repository. Instead, the five computed folds are provided. Each fold follows an 80%-20% split (80% for training and 20% for evaluation).

The folds are used as input data, and graph miners were used to obtain the graph patterns. The frequent graph patterns obtained are provided in the corresponding pattern files in the repository. The approximate graph patterns mined locally are stored in the LAGP (Local Approximate Graph Patterns) folder. These frequent approximate graph patterns are mined for each individual graph in the collection and then merged to obtain a set of unique patterns. As each graph is considered individually, a frequent approximate graph pattern must be present a certain number of times within that graph. The approximate graph patterns mined globally are stored in the GAGP (Global Approximate Graph Patterns) folder. In this case, the graph patterns are mined throughout the collection; a frequent approximate graph pattern must be present in a number of graphs within the collection, usually considered as a percentage. For each dataset and fold, the corresponding sub-bases are organized according to the experimental configurations.

## Configuration

All experimental settings are defined directly in `run.py`.

Before executing the program, the user must modify the corresponding variables to to select the dataset, sub-base, embedding approach (LAGP, GAGP) , fold, support threshold, and dissimilarity threshold.

```python
DB        = "Coenen"  # "PTC" or "Coenen"
SUB_BASE  = "200"     # Coenen: 100, 200, 300
                      # PTC: FM, FR, MM, MR
APPROACH  = "GAGP"    # "LAGP" or "GAGP"
FOLD      = 2         # 1 to 5
SUPPORT   = "90"      # GAGP: 80, 90
                      # LAGP: 1, 2
DISSIM    = 1         # 1 or 2

OUTPUT_DIR = "Output"
```

### Configuration parameters

| Parameter | Description | Available values |
|---|---|---|
| `DB` | Selects the dataset collection. | Coenen, PTC |
| `SUB_BASE` | Specifies the sub-base to process. | **Coenen:** 100, 200, 300; **PTC:** FM, FR, MM, MR |
| `APPROACH` | Selects the graph pattern mining approach. | LAGP, GAGP |
| `FOLD` | Specifies the fold to process. | 1–5 |
| `SUPPORT` | Defines the support threshold associated with the selected approach. | **GAGP:** 80, 90; **LAGP:** 1, 2 |
| `DISSIM` | Selects the dissimilarity threshold. | 1, 2 |
| `OUTPUT_DIR` | Specifies the directory where the resulting matrices are stored. | Output |
| `LABELS_BY_DB` | The class labels are automatically assigned according to the selected dataset collection. | **Coenen:** labels 0 and 1; **PTC:** labels -1 and 1. |

The program processes the configuration specified in `run.py`. To execute a different experimental configuration, the corresponding variables must be modified before running the program.

To evaluate different configurations, the user must update these values and execute the program again.

## Execution

The main entry point is `run.py`.

### 1. Configure the experiment

Open `run.py` and modify the configuration parameters.

### 2. Execute the program

Run the following command from the project directory:

```bash
python run.py
```

The program uses the functions implemented in `utils.py`.

To process another experimental configuration, modify the corresponding values in `run.py` and execute the program again.

## Output

The program generates an `output/` directory containing the embeddings associated with the selected experimental configuration specified in `run.py`. It does not automatically generate results for all parameter configurations.

The output includes the dissimilarity-based graph embeddings required for training and evaluation, as well as the dissimilarity-based graph embeddings generated from the closed and maximal graph pattern subsets.

The resulting dissimilarity-based graph embeddings can be used as input for subsequent supervised classification experiments.

## Requirements

- Python 3.9
- Additional dependencies required by `run.py` and `utils.py`: networkx and numpy.

## Important Notes

- Graph pattern mining is performed beforehand and this program processes previously mined graph patterns.
- The user must manually set the parameter configurations in `run.py` before each execution.
- The program processes five folds for the selected dataset.
- closed and maximal are extracted and used to generate additional embeddings.
- One training and evaluation matrix containing the dissimilarity-based graph embedding is generated per fold.
