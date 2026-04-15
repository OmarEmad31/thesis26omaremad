"""Audio Emotion Classification — 60% Accuracy Strategy.

PHASES IMPLEMENTED:
  Phase 1-6 features from before: Colab dynamic paths, SMOTE, SWA, Augmentation.
  Step 1: Multi-Backbone Feature Fusion (emotion2vec + Wav2Vec2-Emotion + AST).
          Massive offline-extracted feature space (6144-dim for MLP, 2560-dim for SVM).
  Step 2: ArcFace Metric Learning in the MLP (improves class separation).
  Step 4: Meta-Learner Stacking (Logistic Regression over all model probabilities).

COLAB USAGE:
  !pip install funasr librosa transformers torchaudio -q
  !git pull
  !python -m src.audio_baseline.train
"""
import sys, os, logging, warnings, collections, math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from transformers import set_seed, AutoModel, AutoFeatureExtractor
from transformers import logging as hf_log
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import time
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")
hf_log.set_verbosity_error()
logging.getLogger("funasr").setLevel(logging.ERROR)

from src.audio_baseline import config

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
log_file = config.CHECKPOINT_DIR.parent / "audio_training_log.txt"
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[
    logging.FileHandler(log_file, mode="a", encoding="utf-8"),
    logging.StreamHandler(sys.stdout),
])
logger = logging.getLogger(__name__)


# ===========================================================================
# FEATURE AGGREGATION
# ===========================================================================
def aggregate_frames(feat: np.ndarray, method: str) -> np.ndarray:
    """
    method 'mean': returns [D]
    method 'stats': returns concat(mean, std, max) = [3*D]
    method 'cls': returns feat[0] = [D]
    """
    if feat.ndim == 1:
        if method == "stats":
            return np.concatenate([feat, np.zeros_like(feat), feat])
        return feat
    if method == "mean":
        return feat.mean(0)
    elif method == "cls":
        return feat[0]
    elif method == "stats":
        return np.concatenate([feat.mean(0), feat.std(0), feat.max(0)])
    return feat.mean(0)


def get_augmentations(audio: np.ndarray) -> list[tuple[str, np.ndarray]]:
    import librosa
    out = [("orig", audio.copy())]
    for rate in config.AUG_SPEED_RATES:
        try: out.append((f"spd{rate}", librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)))
        except: pass
    noise = config.AUG_NOISE_STD * float(np.std(audio) + 1e-8)
    out.append(("n", (audio + noise * np.random.randn(len(audio))).astype(np.float32)))
    return out


# ===========================================================================
# SMOTE
# ===========================================================================
def simple_smote(X, y, target, k=5):
    classes, Xs, ys = np.unique(y), [X], [y]
    for cls in classes:
        mask = y == cls
        X_cls = X[mask]
        needed = target - len(X_cls)
        if needed <= 0 or len(X_cls) < 2: continue
        nn = NearestNeighbors(n_neighbors=min(k, len(X_cls)-1) + 1).fit(X_cls)
        _, idx = nn.kneighbors(X_cls)
        for _ in range(needed):
            i = np.random.randint(len(X_cls))
            j = idx[i, 1 + np.random.randint(idx.shape[1]-1)]
            lam = np.random.random()
            Xs.append(((lam * X_cls[i] + (1 - lam) * X_cls[j]).astype(np.float32))[None])
            ys.append(np.array([cls]))
    return np.vstack(Xs), np.concatenate(ys)


# ===========================================================================
# FEATURE EXTRACTION (MULTI-BACKBONE)
# ===========================================================================
def extract_single_model(model_id, df, label2id, augment_mlp, device):
    """Memory-efficient extraction. Loads one model, extracts for all splits, deletes model."""
    import librosa
    logger.info(f"   📥 Loading {model_id}...")
    
    # Check model type
    is_funasr = "emotion2vec" in model_id.lower()
    is_ast    = "ast" in model_id.lower()
    is_w2v2   = "wav2vec" in model_id.lower() or "hubert" in model_id.lower()

    if is_funasr:
        from funasr import AutoModel as FAutoModel
        model = FAutoModel(model=model_id, hub="hf", trust_remote_code=True)
        model.model.eval()
        for p in model.model.parameters(): p.requires_grad = False
    else:
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()

    def process_split(split_df, is_train):
        do_aug = is_train and augment_mlp
        svm_embs, mlp_embs, svm_labels, mlp_labels = [], [], [], []
        
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"    {model_id[-10:]} ({'aug' if do_aug else 'cln'})"):
            fldr = str(row["folder"]).strip()
            rel = str(row["audio_relpath"]).replace("\\", "/").lstrip("/")
            cand = config.DATA_ROOT / fldr / rel
            if not cand.exists(): continue
            try:
                audio, _ = librosa.load(cand, sr=config.SAMPLING_RATE, mono=True)
                audio = audio[:config.MAX_AUDIO_SAMPLES].astype(np.float32)
            except: continue

            versions = get_augmentations(audio) if do_aug else [("orig", audio)]
            lbl = label2id[row["emotion_final"]]

            for name, wav in versions:
                wav = wav[:config.MAX_AUDIO_SAMPLES]
                try:
                    if is_funasr:
                        with open(os.devnull, "w") as dn:
                            sys.stdout, sys.stderr = dn, dn
                            try: res = model.generate(input=wav, granularity="frame", extract_embedding=True)
                            finally: sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
                        if not res: continue
                        feat = np.array(res[0]["feats"], dtype=np.float32)
                        
                        mlp_e = aggregate_frames(feat, "stats")
                        svm_e = aggregate_frames(feat, "mean")
                        
                    elif is_w2v2:
                        inp = processor(wav, sampling_rate=16000, return_tensors="pt").input_values.to(device)
                        with torch.no_grad(): out = model(inp).last_hidden_state[0].cpu().numpy()
                        mlp_e = aggregate_frames(out, "stats")
                        svm_e = aggregate_frames(out, "mean")
                        
                    elif is_ast:
                        inp = processor(wav, sampling_rate=16000, return_tensors="pt").input_values.to(device)
                        with torch.no_grad(): out = model(inp).last_hidden_state[0].cpu().numpy()
                        mlp_e = aggregate_frames(out, "cls")
                        svm_e = aggregate_frames(out, "cls")

                    mlp_embs.append(mlp_e)
                    mlp_labels.append(lbl)
                    if name == "orig":
                        svm_embs.append(svm_e)
                        svm_labels.append(lbl)
                        
                except Exception as e:
                    continue

        return np.array(svm_embs), np.array(mlp_embs), np.array(svm_labels), np.array(mlp_labels)

    tr_svm, tr_mlp, tr_s_lbl, tr_m_lbl = process_split(df["train"], is_train=True)
    va_svm, va_mlp, va_s_lbl, va_m_lbl = process_split(df["val"],   is_train=False)
    te_svm, te_mlp, te_s_lbl, te_m_lbl = process_split(df["test"],  is_train=False)

    del model; import gc; gc.collect(); torch.cuda.empty_cache()
    return (tr_svm, tr_mlp, tr_s_lbl, tr_m_lbl), (va_svm, va_mlp, va_s_lbl, va_m_lbl), (te_svm, te_mlp, te_s_lbl, te_m_lbl)


# ===========================================================================
# ARCFACE & SVM/MLP
# ===========================================================================
class ArcFaceLoss(nn.Module):
    """Additive Angular Margin Loss. STRICT margin between classes = better minority accuracy."""
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        # Cosine similarity
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-9, 1))
        # phi = cos(theta + m)
        phi = cosine * math.cos(self.m) - sine * math.sin(self.m)
        # Apply margin only to target class
        one_hot = torch.zeros(cosine.size(), device=features.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return F.cross_entropy(output, labels)


class AudioMLP(nn.Module):
    def __init__(self, in_dim, num_labels):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Dropout(0.4), nn.Linear(in_dim, 1024), nn.GELU(),
            nn.LayerNorm(1024), nn.Dropout(0.3), nn.Linear(1024, 512), nn.GELU(),
            nn.LayerNorm(512), nn.Dropout(0.2)
        )
        self.arcface = ArcFaceLoss(512, num_labels, s=30.0, m=0.50)
        self.eval_linear = nn.Linear(512, num_labels) # For standard eval outputs
        self.eval_linear.weight = self.arcface.weight # Tie weights

    def forward(self, x, labels=None):
        feat = self.net(x)
        if self.training and labels is not None:
            return self.arcface(feat, labels)
        return self.eval_linear(F.normalize(feat)) * self.arcface.s # Scale appropriately


from sklearn.decomposition import PCA

def run_sklearn(Xtr, ytr, Xva, yva, Xte, yte, id2label):
    # Scale Data
    scaler = StandardScaler()
    Xtr, Xva, Xte = scaler.fit_transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
    
    # Compress 2560 dims to 95% variance for ultra-fast Tree building
    pca = PCA(n_components=0.95, random_state=42)
    Xtr = pca.fit_transform(Xtr)
    Xva = pca.transform(Xva)
    Xte = pca.transform(Xte)

    res, probs_va, probs_te = {}, {}, {}
    
    logger.info("   [SVM] fitting...")
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced", random_state=42).fit(Xtr, ytr)
    probs_va["svm"], probs_te["svm"] = svm.predict_proba(Xva), svm.predict_proba(Xte)
    res["SVM_test_f1"] = f1_score(yte, svm.predict(Xte), average="macro")

    logger.info("   [GBM] fitting...")
    gbm = HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=42).fit(Xtr, ytr)
    probs_va["gbm"], probs_te["gbm"] = gbm.predict_proba(Xva), gbm.predict_proba(Xte)
    res["GBM_test_f1"] = f1_score(yte, gbm.predict(Xte), average="macro")
    
    logger.info("   [RF] fitting...")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42).fit(Xtr, ytr)
    probs_va["rf"], probs_te["rf"] = rf.predict_proba(Xva), rf.predict_proba(Xte)
    res["RF_test_f1"] = f1_score(yte, rf.predict(Xte), average="macro")

    return res, probs_va, probs_te, scaler


def train_mlp(Xtr, ytr, Xva, yva, num_lbl, device):
    tr_loader = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)), batch_size=config.BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(TensorDataset(torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.long)), batch_size=256)

    model = AudioMLP(Xtr.shape[1], num_lbl).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config.LEARNING_RATE*10, steps_per_epoch=len(tr_loader), epochs=config.EPOCHS, pct_start=0.1)

    swa_model = torch.optim.swa_utils.AveragedModel(model) if config.USE_SWA else None
    swa_start = int(config.EPOCHS * config.SWA_START_FRAC)
    swa_sched = torch.optim.swa_utils.SWALR(opt, swa_lr=5e-5) if config.USE_SWA else None
    swa_used = False

    best_f1, b_state, pat = 0, None, 0
    for ep in range(config.EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            loss = model(xb.to(device), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            if not config.USE_SWA or ep < swa_start: sched.step()
        
        if config.USE_SWA and ep >= swa_start:
            swa_model.update_parameters(model); swa_sched.step(); swa_used = True

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                vp.extend(model(xb.to(device)).argmax(1).cpu().numpy())
                vt.extend(yb.numpy())
        vf = f1_score(vt, vp, average="macro")
        if vf > best_f1:
            best_f1, b_state, pat = vf, {k: v.clone() for k,v in model.state_dict().items()}, 0
        else: pat += 1
        
        if pat >= config.MAX_PATIENCE: break

    model.load_state_dict(b_state)
    return swa_model if swa_used else model, model


def get_mlp_probs(model, X, device):
    model.eval()
    with torch.no_grad():
        lg = model(torch.tensor(X, dtype=torch.float32).to(device))
        return F.softmax(lg, -1).cpu().numpy()


# ===========================================================================
# MAIN RUNNER
# ===========================================================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n🚀 STEP 1: Multi-Backbone Feature Fusion ({device})")

    df_dict = {
        "train": pd.read_csv(config.SPLIT_CSV_DIR / "train.csv"),
        "val":   pd.read_csv(config.SPLIT_CSV_DIR / "val.csv"),
        "test":  pd.read_csv(config.SPLIT_CSV_DIR / "test.csv"),
    }
    label2id = {n: i for i, n in enumerate(sorted(df_dict["train"]["emotion_final"].unique()))}
    id2label = {i: n for n, i in label2id.items()}

    cache_file = config.EMBEDDING_CACHE
    if cache_file.exists():
        logger.info(f"⚡ Loading fusion cache: {cache_file}")
        d = np.load(cache_file)
        S_tr, M_tr, L_tr = d["S_tr"], d["M_tr"], d["L_tr"]
        S_va, M_va, L_va = d["S_va"], d["M_va"], d["L_va"]
        S_te, M_te, L_te = d["S_te"], d["M_te"], d["L_te"]
        
        # Backwards compatibility: if ML_tr isn't stored, M_tr must be exactly ratio-sized to L_tr
        if "ML_tr" in d:
            ML_tr = d["ML_tr"]
        else:
            ratio = len(M_tr) // len(L_tr)
            ML_tr = np.repeat(L_tr, ratio)
            
    else:
        logger.info("🧠 Commencing Multi-Backbone Sequential Extraction. This takes time but avoids OOM.")
        models = [config.MODEL_NAME, config.W2V2_NAME, config.AST_NAME]
        svm_parts_tr, mlp_parts_tr = [], []
        svm_parts_va, mlp_parts_va = [], []
        svm_parts_te, mlp_parts_te = [], []
        
        for mid in models:
            tr, va, te = extract_single_model(mid, df_dict, label2id, config.AUGMENT_MLP_TRAIN, device)
            svm_parts_tr.append(tr[0]); mlp_parts_tr.append(tr[1]); L_tr = tr[2]; ML_tr = tr[3]
            svm_parts_va.append(va[0]); mlp_parts_va.append(va[1]); L_va = va[2]
            svm_parts_te.append(te[0]); mlp_parts_te.append(te[1]); L_te = te[2]
            
        S_tr, M_tr = np.concatenate(svm_parts_tr, axis=1), np.concatenate(mlp_parts_tr, axis=1)
        S_va, M_va = np.concatenate(svm_parts_va, axis=1), np.concatenate(mlp_parts_va, axis=1)
        S_te, M_te = np.concatenate(svm_parts_te, axis=1), np.concatenate(mlp_parts_te, axis=1)
        
        logger.info(f"   💾 Fusion complete! Saving multi-backbone cache ({M_tr.shape[1]}-dim)...")
        np.savez_compressed(cache_file, S_tr=S_tr, M_tr=M_tr, L_tr=L_tr, ML_tr=ML_tr,
                            S_va=S_va, M_va=M_va, L_va=L_va, S_te=S_te, M_te=M_te, L_te=L_te)

    logger.info(f"   SVM Features (3 pooled backbones) : Dim = {S_tr.shape[1]}")
    logger.info(f"   MLP Features (3 pooled+aug backbones): Dim = {M_tr.shape[1]}")

    if config.USE_SMOTE:
        tgt = max(collections.Counter(L_tr.tolist()).values())
        logger.info(f"🧬 Phase 5 SMOTE: Target {tgt}/class")
        M_tr, ML_tr = simple_smote(M_tr, ML_tr, target=tgt) # Expand MLP Data Only
        
    sk_res, probs_va, probs_te, scaler = run_sklearn(S_tr, L_tr, S_va, L_va, S_te, L_te, id2label)
    
    logger.info("   [MLP] training with ARCFACE loss...")
    final_mlp, _ = train_mlp(M_tr, ML_tr, M_va, L_va, len(id2label), device)
    
    # MLP outputs
    probs_te["mlp"] = get_mlp_probs(final_mlp, M_te, device)
    probs_va["mlp"] = get_mlp_probs(final_mlp, M_va, device)
    mlp_f1 = f1_score(L_te, probs_te["mlp"].argmax(1), average="macro")

    # =======================================================================
    # META-LEARNER STACKING
    # =======================================================================
    # Fit meta learner purely on Validation Set (prevents overfit & aligns dimensions flawlessly)
    meta_X_va = np.concatenate([probs_va[m] for m in ["svm", "gbm", "rf", "mlp"]], axis=1)
    meta_X_te = np.concatenate([probs_te[m] for m in ["svm", "gbm", "rf", "mlp"]], axis=1)

    meta_clf = LogisticRegression(C=1.0, class_weight="balanced").fit(meta_X_va, L_va)
    meta_pred = meta_clf.predict(meta_X_te)
    meta_acc = accuracy_score(L_te, meta_pred)
    meta_f1  = f1_score(L_te, meta_pred, average="macro")

    logger.info("\n======================================================================")
    logger.info("🏆 FINAL AUDIO RESULTS (FUSION + ARCFACE STACK)")
    logger.info("======================================================================")
    logger.info(f"   GBM (Fusion 2560-dim)        Test F1: {sk_res['GBM_test_f1']:.4f}")
    logger.info(f"   RF  (Fusion 2560-dim)        Test F1: {sk_res['RF_test_f1']:.4f}")
    logger.info(f"   SVM (Fusion 2560-dim)        Test F1: {sk_res['SVM_test_f1']:.4f}")
    logger.info(f"   MLP-ArcFace (Fusion 6144-dim) Test F1: {mlp_f1:.4f}")
    logger.info("-" * 70)
    logger.info(f"   🏅 Meta-Learner Ensemble Test Accuracy : {meta_acc:.4f}")
    logger.info(f"   🏅 Meta-Learner Ensemble Test F1       : {meta_f1:.4f}")
    logger.info("======================================================================")

if __name__ == "__main__":
    main()
