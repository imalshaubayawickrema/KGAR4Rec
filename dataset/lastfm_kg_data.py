import os, pandas as pd, numpy as np
from types import SimpleNamespace as edict
import json

def _read_tsv(path):
    return pd.read_csv(path, sep="\t")

def load_user_artists(data_dir, test, pad_idx: int | None = None, to_raw_artist_ids: bool = True):

    if to_raw_artist_ids:
        bridge = pd.read_pickle(os.path.join(data_dir, "artist.df"))
        dense2raw = dict(zip(bridge["id"].astype(int), bridge["artistID"].astype(int)))

    pairs = []
    for row in test.itertuples(index=False):
        u = int(getattr(row, "userID"))
        seq = list(getattr(row, "seq"))
        L = int(getattr(row, "len_seq"))
        items = seq[:L]  # drop right padding
        if pad_idx is not None:
            items = [it for it in items if it != pad_idx]
        if to_raw_artist_ids:
            items = [dense2raw[it] for it in items if it in dense2raw]
        for a in set(items):
            pairs.append((u, a))

    ua = pd.DataFrame(pairs, columns=["userID","artistID"]).drop_duplicates()
    return ua

def load_user_taggedartists(data_dir):
    ts = os.path.join(data_dir, "user_taggedartists-timestamps.dat")
    dated = os.path.join(data_dir, "user_taggedartists.dat")
    if os.path.exists(ts):
        df = _read_tsv(ts)
        df.columns = ["userID","artistID","tagID","timestamp"][:df.shape[1]]
        for c in ["userID","artistID","tagID","timestamp"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["userID","artistID","tagID","timestamp"]).astype("int64")
    else:
        df = _read_tsv(dated)
        df.columns = ["userID","artistID","tagID","day","month","year"][:df.shape[1]]
        for c in ["userID","artistID","tagID","day","month","year"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["userID","artistID","tagID","day","month","year"]).astype("int64")
        ts_ms = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]), errors="coerce").astype("int64") // 10**6
        df["timestamp"] = ts_ms
        df = df[["userID","artistID","tagID","timestamp"]]
    # normalize: earliest tag per (user, artist, tag)
    df = df.sort_values(["userID","artistID","tagID","timestamp"])
    df = df.groupby(["userID","artistID","tagID"], as_index=False).first()
    return df

def load_tags(data_dir):
    path = os.path.join(data_dir, "tags.dat")
    for e in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, sep="\t", header=0, encoding=e, engine="python")
        except Exception:
            pass
    # Some dumps use "tagID\ttagValue", others have named cols; normalize:
    if df.shape[1] == 2:
        df.columns = ["tagID","tagValue"]
    if "tagID" not in df.columns:
        df.columns = ["tagID","tagValue"][:df.shape[1]]
    df["tagID"] = pd.to_numeric(df["tagID"], errors="coerce")
    df = df.dropna(subset=["tagID"]).astype({"tagID":"int64"})
    return df[["tagID","tagValue"]] if "tagValue" in df.columns else df.assign(tagValue="")

def load_user_friends(data_dir):
    path = os.path.join(data_dir, "user_friends.dat")
    df = _read_tsv(path)
    df.columns = ["userID","friendID"][:df.shape[1]]
    for c in ["userID","friendID"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["userID","friendID"]).astype("int64")
    # dedupe edges
    df = df.drop_duplicates(["userID","friendID"])
    return df


def load_artist_bridge(data_dir):
    path = os.path.join(data_dir, "artist.df")
    if not os.path.exists(path):
        return None
    df = pd.read_pickle(path)
    df = df.sort_values("id").reset_index(drop=True)
    amap = dict(zip(df["artistID"].astype(int), df["id"].astype(int)))
    artist_vocab = df["artistName"].tolist()  # index = id
    return df, amap, artist_vocab


def _load_artist_relation_edges(data_dir, filename, valid_artist_ids=None):
    path = os.path.join(data_dir, filename) 
    if not os.path.exists(path):
        return pd.DataFrame(columns=["artistID","objQID","objLabel","objDesc"])
    df = pd.read_csv(path)
    df["artistID"] = pd.to_numeric(df["artistID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["artistID","objQID"]).astype({"artistID":"int64"})
    df = df[df["objQID"].astype(str).str.startswith("Q")]
    # filter by artists present in artist_df
    if valid_artist_ids is not None:
        valid_artist_ids = set(int(a) for a in valid_artist_ids)
        df = df[df["artistID"].isin(valid_artist_ids)]
    return df[["artistID","objQID","objLabel"]].drop_duplicates()

def _build_obj_vocab(df):
    if df is None or df.empty:
        return {}, []
    qids = sorted(df["objQID"].astype(str).unique())
    omap = {q:i for i,q in enumerate(qids)}
    best = (df.dropna(subset=["objLabel"])
              .drop_duplicates(["objQID"])
              .set_index("objQID")["objLabel"]
              .to_dict())
    vocab = [best.get(q, q) for q in qids]
    return omap, vocab


def build_vocabs(data_dir,test):
    artist_df, amap, artist_vocab = load_artist_bridge(data_dir)
    pad_idx = len(artist_vocab)
    ua = load_user_artists(data_dir, test, pad_idx=pad_idx,to_raw_artist_ids=True)                 # userID, artistID, weight
    ut = load_user_taggedartists(data_dir)           # userID, artistID, tagID, timestamp
    tg = load_tags(data_dir)                         # tagID, tagValue
    fr = load_user_friends(data_dir)                 # userID, friendID


    users_raw = set(ua["userID"].unique())
    ut = ut[ut["userID"].isin(users_raw)]
    fr = fr[fr["userID"].isin(users_raw) & fr["friendID"].isin(users_raw)]

    user_ids = sorted(users_raw)
    tag_ids = sorted(ut["tagID"].unique())

    # rawID -> dense idx maps
    umap = {rid:i for i,rid in enumerate(user_ids)}
    tmap = {rid:i for i,rid in enumerate(tag_ids)}

    # vocabs
    user_vocab   = [str(rid) for rid in user_ids]
    tag_id2name = dict(zip(tg["tagID"], tg["tagValue"]))
    tag_vocab    = [tag_id2name.get(rid, str(rid)) for rid in tag_ids]

    valid_artist_ids = set(artist_df["artistID"].astype("int64").tolist())

    genre_e = _load_artist_relation_edges(data_dir, "artist_genre_edges.csv", valid_artist_ids=valid_artist_ids)
    gmap, genre_vocab = _build_obj_vocab(genre_e)

    return edict(
        # raw→idx maps
        umap=umap, amap=amap, tmap=tmap, gmap=gmap,
        # idx→string/name lists
        user=edict(vocab=user_vocab,   vocab_size=len(user_vocab)),
        artist=edict(vocab=artist_vocab, vocab_size=len(artist_vocab)),
        tag=edict(vocab=tag_vocab,     vocab_size=len(tag_vocab)),
        genre=edict(vocab=genre_vocab, vocab_size=len(genre_vocab)),
        # raw frames
        ua=ua, ut=ut, fr=fr, genre_e=genre_e,
    )


