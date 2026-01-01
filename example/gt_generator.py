import numpy as np

from dataset import get_datasets_config, read_vecs, write_ivecs

def faiss_groundtruth(base: np.ndarray,
                      query: np.ndarray,
                      topR: int,
                      metric: str = "l2"):
    """
    Compute exact topR ground truth with Faiss.

    Args:
        base:  (N, D) float32 numpy array.
        query: (Q, D) float32 numpy array.
        topR:  number of nearest neighbors to return.
        metric: "l2" or "ip" (inner product). For cosine, normalize then use "ip".

    Returns:
        gt_I: (Q, topR) int64 numpy array of neighbor IDs.
        gt_D: (Q, topR) float32 numpy array of distances (L2^2 for l2, -ip? see notes below).
    """
    import faiss  # local import so caller sees error if faiss not installed

    if topR <= 0:
        raise ValueError("topR must be > 0")

    # Ensure float32 contiguous
    base = np.ascontiguousarray(base, dtype=np.float32)
    query = np.ascontiguousarray(query, dtype=np.float32)

    if base.ndim != 2 or query.ndim != 2:
        raise ValueError("base and query must be 2D arrays")
    if base.shape[1] != query.shape[1]:
        raise ValueError(f"Dim mismatch: base D={base.shape[1]} vs query D={query.shape[1]}")

    D = base.shape[1]
    if metric.lower() == "l2":
        index = faiss.IndexFlatL2(D)     # exact L2 (actually returns squared L2 distances)
    elif metric.lower() == "ip":
        index = faiss.IndexFlatIP(D)     # exact inner product (larger is better)
    else:
        raise ValueError("metric must be 'l2' or 'ip'")

    index.add(base)
    gt_D, gt_I = index.search(query, topR)

    # Faiss returns:
    # - L2: squared L2 distances (smaller is better)
    # - IP: inner products (larger is better)
    # Keep as-is; caller can post-process if needed.
    return gt_I, gt_D


confs = get_datasets_config(["tiny5m"], mod=None)

for ds, conf in confs.items():
    print(f"Processing {ds}")

    base = read_vecs(conf['base'])
    query = read_vecs(conf['query'])
    gt_I, gt_D = faiss_groundtruth(base, query, 1000)

    write_ivecs(f"{conf['path']}/gt_top1000.ivecs", gt_I)