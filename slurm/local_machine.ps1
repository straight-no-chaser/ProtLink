python scripts/run_graphsage.py `
  --fasta "C:\Users\ThinkPad\Downloads\sage\seqs.fasta" `
  --edges "C:\Users\ThinkPad\Downloads\sage\9606.protein.physical.links.detailed.v12.0.txt" `
  --embeddings "C:\Users\ThinkPad\Downloads\sage\esm2_35m.npz" `
  --device cpu `
  --output-dir "C:\Users\ThinkPad\Downloads\sage\outputs\graphsage"

python scripts/run_mlp.py `
  --fasta "C:\Users\ThinkPad\Downloads\sage\seqs.fasta" `
  --edges "C:\Users\ThinkPad\Downloads\sage\9606.protein.physical.links.detailed.v12.0.txt" `
  --embeddings "C:\Users\ThinkPad\Downloads\sage\esm2_35m.npz" `
  --device cpu `
  --output-dir "C:\Users\ThinkPad\Downloads\sage\outputs\mlp"

python scripts/run_cosine.py `
  --fasta "C:\Users\ThinkPad\Downloads\sage\seqs.fasta" `
  --edges "C:\Users\ThinkPad\Downloads\sage\9606.protein.physical.links.detailed.v12.0.txt" `
  --embeddings "C:\Users\ThinkPad\Downloads\sage\esm2_35m.npz" `
  --output-dir "C:\Users\ThinkPad\Downloads\sage\outputs\cosine"

python scripts/run_graphsage.py `
  --fasta "C:\Users\ThinkPad\Downloads\sage\seqs.fasta" `
  --edges "C:\Users\ThinkPad\Downloads\sage\9606.protein.physical.links.detailed.v12.0.txt" `
  --embeddings "C:\Users\ThinkPad\Downloads\sage\esm2_35m.npz" `
  --device cpu `
  --decoder mlp `
  --residual concat `
  --split-mode edge_random `
  --negative-mode random `
  --output-dir "C:\Users\ThinkPad\Downloads\sage\outputs\fair_decoder"

python scripts/run_cosine.py `
  --fasta "C:\Users\ThinkPad\Downloads\sage\seqs.fasta" `
  --edges "C:\Users\ThinkPad\Downloads\sage\9606.protein.physical.links.detailed.v12.0.txt" `
  --embeddings "C:\Users\ThinkPad\Downloads\sage\esm2_35m.npz" `
  --split-mode edge_random `
  --negative-mode two_hop_hard `
  --output-dir "C:\Users\ThinkPad\Downloads\sage\outputs\hard_negatives\cosine"

python scripts/run_mlp.py `
  --fasta "C:\Users\ThinkPad\Downloads\sage\seqs.fasta" `
  --edges "C:\Users\ThinkPad\Downloads\sage\9606.protein.physical.links.detailed.v12.0.txt" `
  --embeddings "C:\Users\ThinkPad\Downloads\sage\esm2_35m.npz" `
  --device cpu `
  --split-mode edge_random `
  --negative-mode two_hop_hard `
  --output-dir "C:\Users\ThinkPad\Downloads\sage\outputs\hard_negatives\mlp"

python scripts/run_graphsage.py `
  --fasta "C:\Users\ThinkPad\Downloads\sage\seqs.fasta" `
  --edges "C:\Users\ThinkPad\Downloads\sage\9606.protein.physical.links.detailed.v12.0.txt" `
  --embeddings "C:\Users\ThinkPad\Downloads\sage\esm2_35m.npz" `
  --device cpu `
  --decoder mlp `
  --residual concat `
  --split-mode edge_random `
  --negative-mode two_hop_hard `
  --output-dir "C:\Users\ThinkPad\Downloads\sage\outputs\hard_negatives\graphsage_mlp"

python scripts/run_cosine.py `
  --fasta "C:\Users\ThinkPad\Downloads\sage\seqs.fasta" `
  --edges "C:\Users\ThinkPad\Downloads\sage\9606.protein.physical.links.detailed.v12.0.txt" `
  --embeddings "C:\Users\ThinkPad\Downloads\sage\esm2_35m.npz" `
  --split-mode node_disjoint `
  --negative-mode random `
  --output-dir "C:\Users\ThinkPad\Downloads\sage\outputs\node_disjoint\cosine"

python scripts/run_mlp.py `
  --fasta "C:\Users\ThinkPad\Downloads\sage\seqs.fasta" `
  --edges "C:\Users\ThinkPad\Downloads\sage\9606.protein.physical.links.detailed.v12.0.txt" `
  --embeddings "C:\Users\ThinkPad\Downloads\sage\esm2_35m.npz" `
  --device cpu `
  --split-mode node_disjoint `
  --negative-mode random `
  --output-dir "C:\Users\ThinkPad\Downloads\sage\outputs\node_disjoint\mlp"

python scripts/run_graphsage.py `
  --fasta "C:\Users\ThinkPad\Downloads\sage\seqs.fasta" `
  --edges "C:\Users\ThinkPad\Downloads\sage\9606.protein.physical.links.detailed.v12.0.txt" `
  --embeddings "C:\Users\ThinkPad\Downloads\sage\esm2_35m.npz" `
  --device cpu `
  --decoder mlp `
  --residual concat `
  --split-mode node_disjoint `
  --negative-mode random `
  --output-dir "C:\Users\ThinkPad\Downloads\sage\outputs\node_disjoint\graphsage_mlp"

python scripts/run_graphsage.py `
  --fasta "C:\Users\ThinkPad\Downloads\sage\seqs.fasta" `
  --edges "C:\Users\ThinkPad\Downloads\sage\9606.protein.physical.links.detailed.v12.0.txt" `
  --embeddings "C:\Users\ThinkPad\Downloads\sage\esm2_35m.npz" `
  --graph-type hetero `
  --encoder hgt `
  --device cpu `
  --decoder mlp `
  --hgt-heads 4 `
  --hgt-dropout 0.2 `
  --hgt-layers 2 `
  --negative-mode two_hop_hard `
  --output-dir "C:\Users\ThinkPad\Downloads\sage\outputs"

python scripts/run_graphsage.py `
  --experiment multispecies_hetero `
  --graph-type hetero `
  --encoder hgt `
  --decoder mlp `
  --fasta "C:\Users\ThinkPad\Downloads\sage\data\proteins.fasta" `
  --protein-metadata "C:\Users\ThinkPad\Downloads\sage\data\protein_metadata.tsv" `
  --ppi-edges "C:\Users\ThinkPad\Downloads\sage\data\ppi_edges.tsv" `
  --protein-to-orthogroup "C:\Users\ThinkPad\Downloads\sage\data\protein_to_orthogroup.tsv" `
  --embeddings "C:\Users\ThinkPad\Downloads\sage\data\esm2_t33_650m.npz" `
  --target-species "9606" `
  --device cpu `
  --hgt-heads 4 `
  --hgt-dropout 0.2 `
  --hgt-layers 2 `
  --output-dir "C:\Users\ThinkPad\Downloads\sage\outputs\default_650m"

python scripts/run_graphsage.py `
  --experiment multispecies_hetero `
  --graph-type hetero `
  --encoder hgt `
  --decoder mlp `
  --fasta "C:\Users\ThinkPad\Downloads\sage\data\proteins.fasta" `
  --protein-metadata "C:\Users\ThinkPad\Downloads\sage\data\protein_metadata.tsv" `
  --ppi-edges "C:\Users\ThinkPad\Downloads\sage\data\ppi_edges.tsv" `
  --protein-to-orthogroup "C:\Users\ThinkPad\Downloads\sage\data\protein_to_orthogroup.tsv" `
  --embeddings "C:\Users\ThinkPad\Downloads\sage\data\esm2_t33_650m.npz" `
  --target-species "9606" `
  --device cpu `
  --hgt-heads 4 `
  --hgt-dropout 0.2 `
  --hgt-layers 2 `
  --negative-mode two_hop_hard `
  --output-dir "C:\Users\ThinkPad\Downloads\sage\outputs\hardneg_650m"

python scripts/run_graphsage.py `
  --experiment multispecies_hetero `
  --graph-type hetero `
  --encoder hgt `
  --decoder mlp `
  --fasta "C:\Users\ThinkPad\Downloads\sage\data\proteins.fasta" `
  --protein-metadata "C:\Users\ThinkPad\Downloads\sage\data\protein_metadata.tsv" `
  --ppi-edges "C:\Users\ThinkPad\Downloads\sage\data\ppi_edges.tsv" `
  --protein-to-orthogroup "C:\Users\ThinkPad\Downloads\sage\data\protein_to_orthogroup.tsv" `
  --embeddings "C:\Users\ThinkPad\Downloads\sage\data\esm2_t33_650m.npz" `
  --target-species "9606" `
  --device cpu `
  --hgt-heads 4 `
  --hgt-dropout 0.2 `
  --hgt-layers 2 `
  --split-mode node_disjoint `
  --output-dir "C:\Users\ThinkPad\Downloads\sage\outputs\node_disjoint_650m"
