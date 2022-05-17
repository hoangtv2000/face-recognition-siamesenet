import faiss 
import pandas as pd
import numpy as np
from source import config



def preprocess_index_anno_file(anno_path, gpu=False):
    pd_csv = pd.read_pickle(anno_path)
    known_faces_embedded = pd_csv['feature_vector'].values
    face_ids = pd_csv['ID'].values

    train_vecs = np.stack(known_faces_embedded).astype(np.float32)
    face_ids = np.stack(face_ids).astype(np.int64)
    faiss.normalize_L2(train_vecs)

    index = faiss.IndexFlatL2(config.EMBEDDING_DIM)
    index = faiss.IndexIDMap(index) # Mapping df index as id 
    index.add_with_ids(train_vecs, face_ids)
    
    if gpu == True:
        res = faiss.StandardGPUResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    return index, face_ids, pd_csv



def inference(index, embedd_queries):
    faiss.normalize_L2(embedd_queries)
    face_sim, face_sim_idx = index.search(embedd_queries, k=config.K_NEIGHBORS)
    face_sim = np.around(np.clip(face_sim, 0, 1), decimals=4)
    return face_sim, face_sim_idx