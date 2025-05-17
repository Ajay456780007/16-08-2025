import math
import matplotlib.pyplot as plt
import gzip

import seaborn as sns
from Bio import SeqIO
from urllib.parse import unquote
import numpy as np
from keras import Sequential,layers
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, \
    mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import mixed_precision
import os


# === Paths ===
dna_dir = "dataset/dataset1/dna_chromosomes"
gff3_dir = "dataset/dataset1/gff3_files"
#
# # === Collect all FASTA and GFF3 files ===
fasta_files = sorted([
     os.path.join(dna_dir, f) for f in os.listdir(dna_dir)
     if f.lower().endswith(".fa.gz")
 ])
gff3_files = sorted([
     os.path.join(gff3_dir, f) for f in os.listdir(gff3_dir)
     if f.lower().endswith(".gff3")
])
#
#
# === Parse GFF3 attributes ===
def parse_attributes(attr_str):
    attr_dict = {}
    for pair in attr_str.strip().split(";"):
        if "=" in pair:
            key, value = pair.split("=", 1)
            attr_dict[key.strip()] = unquote(value.strip())
    return attr_dict
#
#
# # === Function to parse GFF3 and extract gene entries for a given chromosome ===
def parse_gff3(gff3_file, chrom_id):
    genes = []
    with open(gff3_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            if parts[2] != "gene":
                continue
            if parts[0] != chrom_id:
                continue  # skip if this line refers to a different chromosome

            start = int(parts[3]) - 1  # Convert to 0-based index
            end = int(parts[4])
            strand = parts[6]
            attrs = parse_attributes(parts[8])
            gene_id = attrs.get("ID", "NA")

            genes.append((start, end, strand, gene_id))
    return genes


# === Extract gene sequences ===
gene_sequences = []

for fasta_path, gff_path in zip(fasta_files, gff3_files):
    print(f"Processing: {os.path.basename(fasta_path)} with {os.path.basename(gff_path)}")

    # Read the chromosome sequence (support gzipped or uncompressed)
    if fasta_path.endswith(".gz"):
         with gzip.open(fasta_path, "rt", encoding="utf-8") as f:
             record = next(SeqIO.parse(f, "fasta"))
    else:
         with open(fasta_path, "r", encoding="utf-8") as f:
            record = next(SeqIO.parse(f, "fasta"))

    chrom_seq = record.seq
    chrom_id = record.id

    # Parse gene entries from the GFF3 file
    genes = parse_gff3(gff_path, chrom_id)

    for start, end, strand, gene_id in genes:
        # Boundary check to avoid errors
        if start < 0 or end > len(chrom_seq):
            continue
#
        gene_seq = chrom_seq[start:end]
        if strand == "-":
            gene_seq = gene_seq.reverse_complement()

        gene_sequences.append({
            "gene_id": gene_id,
            "chrom": chrom_id,
            "start": start,
            "end": end,
            "strand": strand,
            "sequence": str(gene_seq)
        })
#
# === Convert to DataFrame ===
df_genes = pd.DataFrame(gene_sequences)

# === Preview first few entries ===
print(df_genes.head())
geo_path = "dataset/dataset1/alias/genes_to_alias_ids.tsv"
df = pd.read_csv(geo_path, sep='\t')
alias_path = "dataset/dataset1/geo_file/abundance.tsv"
df_alias = pd.read_csv(alias_path, sep='\t')
df.rename(columns={"Zm00001eb000010": "id1", "B73 Zm00001eb.1": "id2", "Zm00001d027230": "gene_alias_id"})

import pandas as pd
#
# # Fix column names in 'df'
df.columns = ['id1', 'id2', 'gene_alias_id', 'AGPv4_Zm00001d.2']
#
# # Step 1: Clean the 'gene_id' column in df_genes
df_genes['gene_id_clean'] = df_genes['gene_id'].str.replace('gene:', '', regex=False)
#
# # Step 2: Create a mapping from 'id1' to 'gene_alias_id'
id_to_alias = df.set_index('id1')['gene_alias_id'].to_dict()
#
# # Step 3: Map gene_alias_id to df_genes
df_genes['alias_id'] = df_genes['gene_id_clean'].map(id_to_alias)
#
# # Step 4: Clean up the temporary column
df_genes.drop(columns=['gene_id_clean'], inplace=True)
#
# # Check the results
print(df_genes[['gene_id', 'alias_id']].head(50))
#
# # Drop rows where alias_id is NaN
df_genes = df_genes.dropna(subset=['alias_id'])
#
# # Reset index if needed
df_genes = df_genes.reset_index(drop=True)
#
# # Optional: check that it's gone
print(df_genes['alias_id'].isna().sum())
#

#
# # Step 1: Clean up df_alias to remove transcript suffix
df_alias['clean_id'] = df_alias['target_id'].str.replace(r'_T\d+$', '', regex=True)
#
# # Step 2: Group by clean_id and sum or average TPM if needed (in case multiple transcripts per gene)
# # Here we'll sum TPMs for all isoforms of a gene
tpm_by_gene = df_alias.groupby('clean_id')['tpm'].sum().reset_index()
#
# # Step 3: Merge df_genes with this TPM info using alias_id == clean_id
df_genes = df_genes.merge(tpm_by_gene, how='left', left_on='alias_id', right_on='clean_id')
#
# # Step 4: Rename the column to tpm_value and drop clean_id
df_genes = df_genes.rename(columns={'tpm': 'tpm_value'}).drop(columns=['clean_id'])
#
# # Done
print(df_genes.head())
#
# # Mapping dictionary
base_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
#
#
# # Function to encode a DNA sequence string
def encode_sequence(seq):
     return [base_map.get(base, -1) for base in seq.upper()]  # -1 for unknown bases like N
#
#
# # Apply to each row in df_genes['sequence']
df_genes['encoded_sequence'] = df_genes['sequence'].apply(encode_sequence)
#
# # Done
print(df_genes[['sequence', 'encoded_sequence']].head())
#
df_genes.drop("sequence", axis=1, inplace=True)
#
max_len = df_genes['encoded_sequence'].apply(len).max()
print("Maximum encoded sequence length:", max_len)
#
# # Step 1: Store lengths of all encoded sequences
sequence_lengths = df_genes['encoded_sequence'].apply(len)
#
# # Step 2: Count how many are greater than 50,000
num_greater_than_40000 = (sequence_lengths > 40000).sum()
#
# # Output results
print("Total sequences:", len(sequence_lengths))
print("Sequences > 50,000 bases:", num_greater_than_40000)
#
# Fixed length
FIXED_LEN = 5000
PAD_VALUE = 0  # A = 0

#
def pad_or_truncate(seq):
    if len(seq) > FIXED_LEN:
        return seq[:FIXED_LEN]
    else:
        return seq + [PAD_VALUE] * (FIXED_LEN - len(seq))  # pad
#
#
# # Apply to each sequence
df_genes['encoded_50k'] = df_genes['encoded_sequence'].apply(pad_or_truncate)

# Check shape of one example
print(len(df_genes['encoded_50k'].iloc[0]))
#
print(df_genes.head())

df_genes['strand'] = df_genes['strand'].map({'+': 1, '-': 0})

df_genes.rename(columns={"encoded_50k": "sequence"}, inplace=True)

df_genes["sequence"]
#

#
#
def fix_sequence(seq):
    return [4 if val == -1 else val for val in seq]
#

df_genes['sequence'] = df_genes['sequence'].apply(fix_sequence)
seq = np.array(df_genes['sequence'].tolist(), dtype=np.uint8)


tpm_val = df_genes["tpm_value"]

print("this is x shape:", seq.shape)
print("this is y shape:", tpm_val.shape)
#
import numpy as np

tpm_val = np.array(tpm_val)

np.save("x.npy",seq)
np.save('y.npy',tpm_val)


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

# === Load encoded DNA sequences and TPM ===
x = np.load("x.npy", mmap_mode='r')[:20, :10000]  # Shape: (2000, 1000)
y = np.load("y.npy", mmap_mode='r')[:20]         # Shape: (2000,)

# === Convert memory-mapped arrays to regular numpy arrays ===
x = np.array(x)
y = np.array(y)

# === Compute labels from TPM ===
def compute_labels(tpm_array):
    mean_tpm = np.mean(tpm_array)
    low = mean_tpm / 2
    high = mean_tpm * 1.5
    return np.array([
        0 if t < low else 1 if t < high else 2 for t in tpm_array
    ], dtype=np.int32)

labels = compute_labels(y)

# === Node features from df_genes (make sure this matches x/y) ===
# Assuming df_genes is already defined and has 100 matching entries
df_genes = df_genes.iloc[:20].reset_index(drop=True)
df_genes['length'] = df_genes['end'] - df_genes['start']
df_genes['mean_seq'] = df_genes['sequence'].apply(lambda s: np.mean(s))
node_features = df_genes[['strand', 'length', 'tpm_value', 'mean_seq']].values
node_features = StandardScaler().fit_transform(node_features)

# === Hypergraph adjacency matrix ===
tpm_vector = df_genes['tpm_value'].values
num_genes = len(tpm_vector)
hg_adj = np.zeros((num_genes, num_genes), dtype=np.float32)
threshold = 5.0
for i in range(num_genes):
    for j in range(num_genes):
        if abs(tpm_vector[i] - tpm_vector[j]) < threshold:
            hg_adj[i, j] = 1.0
np.fill_diagonal(hg_adj, 1.0)

# === Stratified Split with Consistent Indexing ===
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(x, labels):
    # DNA sequence data
    x_train = x[train_idx]
    x_test = x[test_idx]

    # Class labels
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    # TPM values (if needed separately)
    tpm_train = y[train_idx]
    tpm_test = y[test_idx]

    # Node features
    # Just expand the dimensions once
    x_hg_train = np.expand_dims(node_features, axis=0)   # shape (1, N, F)
    x_hg_test = np.expand_dims(node_features, axis=0)

    hg_adj_train = np.expand_dims(hg_adj, axis=0)        # shape (1, N, N)
    hg_adj_test = np.expand_dims(hg_adj, axis=0)


# === Final Shape Summary ===
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_hg_train:", x_hg_train.shape)
print("hg_adj_train:", hg_adj_train.shape)
print("Train class distribution:", Counter(y_train))


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import math

# === Ensure CPU is used ===
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forces TensorFlow to use CPU

#=== Remove mixed precision if used ===
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import math

# === Positional Encoding ===
def get_positional_encoding(seq_len, model_dim):
    angle_rads = np.arange(seq_len)[:, np.newaxis] / np.power(
        10000, (2 * (np.arange(model_dim)[np.newaxis, :] // 2)) / model_dim
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

# === Sinusoidal Time Embedding ===
class SinusoidalEmbedding(layers.Layer):
    def __init__(self, model_dim):
        super().__init__()
        self.model_dim = model_dim

    def call(self, x):
        half_dim = self.model_dim // 2
        freqs = tf.exp(tf.linspace(tf.math.log(1.0), tf.math.log(1000.0), half_dim))
        angles = 2.0 * math.pi * tf.cast(x, tf.float32) * freqs
        return tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)

# === Transformer Block ===
class TransformerBlock(layers.Layer):
    def __init__(self, model_dim, heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=heads, key_dim=model_dim, dropout=rate)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dense(model_dim)
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training=False):
        attn_output = self.att(x, x)
        x = self.norm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout2(ffn_output, training=training))

# === Diffusion Transformer Model ===
class DiffusionTransformer(tf.keras.Model):
    def __init__(self, seq_len=5000, model_dim=128, num_heads=4, ff_dim=256, depth=4):
        super().__init__()
        self.embedding = layers.Embedding(input_dim=5, output_dim=model_dim)
        self.pos_encoding = get_positional_encoding(seq_len, model_dim)
        self.time_emb = SinusoidalEmbedding(model_dim)
        self.blocks = [TransformerBlock(model_dim, num_heads, ff_dim) for _ in range(depth)]
        self.global_pool = layers.GlobalAveragePooling1D()

    def call(self, x, training=False):
        seq_embed = self.embedding(x)
        seq_embed += self.pos_encoding[:, :tf.shape(x)[1], :]
        time = tf.zeros((tf.shape(x)[0], 1))
        time_embed = self.time_emb(time)
        time_embed = tf.expand_dims(time_embed, 1)
        time_embed = tf.repeat(time_embed, tf.shape(x)[1], axis=1)
        x = seq_embed + time_embed

        for block in self.blocks:
            x = block(x, training=training)

        return self.global_pool(x)

# === HGNN Layer ===
class HGNNConv(layers.Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = layers.Dense(output_dim)

    def call(self, x, G):
        x = self.linear(x)
        return tf.nn.relu(tf.matmul(G, x))

# === HGNN Embedding ===
class HGNNEmbedding(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.hgc1 = HGNNConv(input_dim, hidden_dim)
        self.hgc2 = HGNNConv(hidden_dim, hidden_dim)

    def call(self, x, G, index):
        x = self.hgc1(x, G)
        x = self.hgc2(x, G)
        return tf.gather(x, index, axis=1, batch_dims=1)  # Extract per-sample node

# === Fusion Module ===
class HybridAttentionFusion(tf.keras.Model):
    def __init__(self, fusion_dim=64, num_classes=3):
        super().__init__()
        self.dense1 = layers.Dense(fusion_dim, activation='relu')
        self.dense2 = layers.Dense(fusion_dim // 2, activation='relu')
        self.out = layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.out(x)

# === Combined Model ===
class CombinedModel(tf.keras.Model):
    def __init__(self, model_dim=128, hgnn_dim=64, fusion_dim=64, num_classes=3):
        super().__init__()
        self.dna_encoder = DiffusionTransformer(model_dim=model_dim)
        self.hgnn = HGNNEmbedding(input_dim=4, hidden_dim=hgnn_dim)
        self.fusion = HybridAttentionFusion(fusion_dim=fusion_dim, num_classes=num_classes)

    def call(self, inputs, training=False):
        dna_seq, node_features, hg_adj, gene_idx = inputs
        batch_size = tf.shape(dna_seq)[0]
        x_hg_batch = tf.repeat(node_features, repeats=batch_size, axis=0)
        hg_adj_batch = tf.repeat(hg_adj, repeats=batch_size, axis=0)
        dna_embed = self.dna_encoder(dna_seq, training=training)
        graph_embed = self.hgnn(x_hg_batch, hg_adj_batch, gene_idx)
        combined = tf.concat([dna_embed, graph_embed], axis=1)
        return self.fusion(combined)

# Proposed Model Building
# Instantiate your pretrained or newly created DiffusionTransformer and HGNNEmbedding
diffusion_transformer = DiffusionTransformer(seq_len=5000, model_dim=64, num_heads=2, ff_dim=128, depth=2)
hgnn_embedding = HGNNEmbedding(input_dim=x_hg_train.shape[2], hidden_dim=64)

import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

class KNN:
    def __init__(self, k):
        self.k = k
        print(f"KNN initialized with k = {self.k}")

    def fit(self, X_train, y_train):
        if self.k > len(X_train):
            raise ValueError("k cannot be greater than the number of training samples")
        self.x_train = np.array(X_train)
        self.y_train = np.array(y_train).flatten()

    def calculate_euclidean(self, sample1, sample2):
        return np.linalg.norm(sample1.astype(np.float32) - sample2.astype(np.float32))

    def nearest_neighbors(self, test_sample):
        distances = [
            (self.y_train[i], self.calculate_euclidean(self.x_train[i], test_sample))
            for i in range(len(self.x_train))
        ]
        distances.sort(key=lambda x: x[1])  # Sort by distance
        neighbors = [distances[i][0] for i in range(self.k)]
        return neighbors

    def majority_vote(self, neighbors):
        count = Counter(neighbors)
        return sorted(count.items(), key=lambda x: (-x[1], x[0]))[0][0]

    def predict(self, test_set):
        predictions = []
        for test_sample in test_set:
            neighbors = self.nearest_neighbors(test_sample)
            prediction = self.majority_vote(neighbors)
            predictions.append(prediction)
        return predictions


def create_BiLSTM(input_shape, num_classes):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=64,
                                 return_sequences=False,
                                 activation='tanh'),
                            input_shape=input_shape))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=["accuracy"])
    return model



def compute_metrics(y_true, y_pred, average='macro'):
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fn + fp)
    specificity = np.mean(tn / (tn + fp)) if np.all(tn + fp) else 0.0

    return {
        "confusion_matrix": cm,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        "specificity": specificity,
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred)
    }

# === Evaluation Loop ===
results = {"ProposedModel": [], "KNN": [], "BiLSTM": []}
metrics = {"ProposedModel": [], "KNN": [], "BiLSTM": []}
training_percentage = [40, 50, 60, 70, 80, 90]

x_all = np.concatenate([x_train, x_test], axis=0)
y_all = np.concatenate([y_train, y_test], axis=0)
num_nodes = node_features.shape[0]  # e.g., 100
gene_idx = np.random.randint(0, num_nodes, size=len(x_all))  # one gene index per sample

x_hg_all = np.repeat(node_features[np.newaxis, :, :], x_all.shape[0], axis=0)
hg_adj_all = np.repeat(hg_adj[np.newaxis, :, :], x_all.shape[0], axis=0)


for percent in training_percentage:
    print(f"\n=== Training with {percent}% of total data ===")
    indices = np.arange(len(x_all))
    np.random.shuffle(indices)
    num_train = int(len(x_all) * percent / 100)

    train_idx = indices[:num_train]
    test_idx = indices[num_train:]

    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    x_test = x_all[test_idx]
    y_test = y_all[test_idx]

    x_hg_train = x_hg_all[train_idx]
    hg_adj_train = hg_adj_all[train_idx]
    x_hg_test = x_hg_all[test_idx]
    hg_adj_test = hg_adj_all[test_idx]
    
    gene_idx_train = gene_idx[train_idx]
    gene_idx_test = gene_idx[test_idx]


    # --- Proposed Model ---
    combined_model = CombinedModel(model_dim=64, hgnn_dim=64, fusion_dim=64, num_classes=3)

    combined_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    combined_model.fit([x_train, x_hg_train, hg_adj_train, gene_idx_train], y_train,
                   validation_data=([x_test, x_hg_test, hg_adj_test, gene_idx_test], y_test),
                   batch_size=32, epochs=10, verbose=0)


    y_pred = np.argmax(combined_model.predict([x_test, x_hg_test, hg_adj_test, gene_idx_test]), axis=1)
    metric_vals = compute_metrics(y_test, y_pred)
    results["ProposedModel"].append(metric_vals["accuracy"])
    metrics["ProposedModel"].append(metric_vals)
    print(f"ProposedModel Accuracy: {metric_vals['accuracy']:.4f}")

    # --- BiLSTM ---
    x_train_bilstm = np.expand_dims(x_train, axis=-1).astype(np.float32)
    x_test_bilstm = np.expand_dims(x_test, axis=-1).astype(np.float32)
    model2 = create_BiLSTM(x_train_bilstm.shape[1:], num_classes=3)
    model2.fit(x_train_bilstm, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
    y_pred = np.argmax(model2.predict(x_test_bilstm), axis=-1)
    metric_vals = compute_metrics(y_test, y_pred)
    results["BiLSTM"].append(metric_vals["accuracy"])
    metrics["BiLSTM"].append(metric_vals)
    print(f"BiLSTM Accuracy: {metric_vals['accuracy']:.4f}")

    # --- KNN ---
    knn_model = KNN(k=5)
    knn_model.fit(x_train, y_train)
    y_pred = knn_model.predict(x_test)
    metric_vals = compute_metrics(y_test, y_pred)
    results["KNN"].append(metric_vals["accuracy"])
    metrics["KNN"].append(metric_vals)
    print(f"KNN Accuracy: {metric_vals['accuracy']:.4f}")

# === Save results ===
np.save("model_accuracy_results.npy", results)
np.save("model_detailed_metrics.npy", metrics)

# === Plot Accuracy Bar Graph ===
bar_width = 0.2
x_range = np.arange(len(training_percentage))
model_names = list(results.keys())
plt.figure(figsize=(12, 6))

for i, model_name in enumerate(model_names):
    plt.bar(x_range + i * bar_width, results[model_name], width=bar_width, label=model_name)

plt.xlabel("Training Percentage")
plt.ylabel("Accuracy")
plt.title("Model Accuracy vs Training Data Percentage")
plt.xticks(x_range + bar_width, training_percentage)
plt.legend()
plt.tight_layout()
plt.savefig("training_percentage_comparison_bar.png")
plt.show()

# === Save Confusion Matrices ===
for model_name in model_names:
    for i, percent in enumerate(training_percentage):
        cm = metrics[model_name][i]["confusion_matrix"]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{model_name} Confusion Matrix ({percent}%)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"conf_matrix_{model_name}_{percent}.png")
        plt.close()
