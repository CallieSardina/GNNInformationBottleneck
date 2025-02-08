## Datasets

### dataset Statistics 

| Dataset           | Mutagenicity |  |
|-------------------|--------------|--|
| # Graphs          | 4308         |  |
| # Nodes           | 130719       |  |
| # Edges           | 132707       |  |
| # Node labels     | 10           |  |

### Mutagenicity

- Classification as mutagenic/ non-mutagenic (binary classificiation task).

## Models

### GCN 

- From https://doi.org/10.1145/3698108. 

    #### Classifier Accuacy

    | # Layers             | 3         | 4         | 5         | 6         | 7         |
    |----------------------|-----------|-----------|-----------|-----------|-----------|
    | Training             | 0.8843    | 0.8631    | 0.8933    | 0.8675    |           |
    | Validation           | 0.8372    | 0.8047    | 0.8279    | 0.8023    |           |
    | Testing              | 0.7977    | 0.7953    | 0.8070    | 0.7844    |           |

## Training & MI Calculation 

run_gnn.py --train=1 --num_layers=3 --model=gcn

- Creates ./MI_logs/{args.model}/mi_log_{args.num_layers}_layers.txt file for logging MI values for each I(X;Z), I(Z;Y).
- Mutual Information Estimator from https://doi.org/10.48550/arXiv.1801.09125. 

## Visualizing Information Plane

python3 utils/mi_plots.py --file_name=MI_logs/gcn/run_results_3_layer --model=gcn

- Note that the file_name passed as cmd line arg should not have the '.txt' ending.
- Visualizes information plane plots and saves them to ./plots/{model}/ folder.

