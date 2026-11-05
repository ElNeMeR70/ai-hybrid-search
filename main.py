"""Reciprocal Rank Fusion (RRF) Hybrid Search"""

def rrf_merge(dense_ranks, sparse_ranks, k=60):
    scores = {}
    for rank, doc in enumerate(dense_ranks):
        scores[doc] = scores.get(doc, 0.0) + (1.0 / (k + rank + 1))
    for rank, doc in enumerate(sparse_ranks):
        scores[doc] = scores.get(doc, 0.0) + (1.0 / (k + rank + 1))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    print("[RUN] Hybrid Search (RRF)...")
    dense = ["docA", "docB", "docC"]
    sparse = ["docB", "docA", "docD"]
    fused = rrf_merge(dense, sparse)
    print(f"[OK] Top Fused Document: {fused[0][0]} (Score: {fused[0][1]:.5f})")
