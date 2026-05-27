import numpy as np
from utils.rw_process import *

class KnowledgeGraphLastFM:
    def __init__(self, ds):
        self.G = {}
        self._init_nodes(ds)
        self._load_listens(ds)     # USER ↔ ARTIST  (from ua)
        self._load_tags(ds)        # USER ↔ TAG, ARTIST ↔ TAG (from ut)
        self._load_friends(ds)     # USER ↔ USER   (from fr)
        self._load_genre(ds)  # ARTIST ↔ GENRE   (from genre_e)
        self._clean()
        self.degrees = None

    def _init_nodes(self, ds):
        sizes = {
            USER:   ds.user.vocab_size,
            ARTIST: ds.artist.vocab_size,
            TAG:    ds.tag.vocab_size,
            GENRE: ds.genre.vocab_size,
        }
        self.G = {etype: {} for etype in sizes}
        for etype, n in sizes.items():
            rels = get_relations(etype)
            for i in range(n):
                self.G[etype][i] = {r: [] for r in rels}

    def _add_edge(self, e1, id1, rel, e2, id2):
        self.G[e1][id1][rel].append(id2)
        self.G[e2][id2][rel].append(id1)

    def _load_listens(self, ds):
        # using user_taggedartists pairs as listens proxy (ua)
        for r in ds.ua.itertuples(index=False):
            u_raw, a_raw = int(r.userID), int(r.artistID)
            u = ds.umap.get(u_raw); a = ds.amap.get(a_raw)
            if u is None or a is None:
                continue
            self._add_edge(USER, u, LISTEN, ARTIST, a)

    def _load_tags(self, ds):
        for r in ds.ut.itertuples(index=False):
            u_raw, a_raw, t_raw = int(r.userID), int(r.artistID), int(r.tagID)
            u = ds.umap.get(u_raw); a = ds.amap.get(a_raw); t = ds.tmap.get(t_raw)
            if u is None or a is None or t is None:
                continue
            self._add_edge(USER,   u, TAGGED,       TAG,    t)
            self._add_edge(ARTIST, a, DESCRIBED_AS, TAG,    t)

    def _load_friends(self, ds):
        seen = set()
        for r in ds.fr.itertuples(index=False):
            u = ds.umap.get(int(r.userID))
            v = ds.umap.get(int(r.friendID))
            if u is None or v is None:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            self._add_edge(USER, a, FRIENDS_WITH, USER, b)

    def _load_genre(self, ds):
        # ARTIST --BELONG_TO--> GENRE
        if hasattr(ds, "genre_e") and ds.genre_e is not None and len(ds.genre_e):
            for r in ds.genre_e.itertuples(index=False):
                a_raw = int(r.artistID)
                a = ds.amap.get(a_raw)
                g = ds.gmap.get(str(r.objQID))
                if a is None or g is None:
                    continue
                self._add_edge(ARTIST, a, BELONG_TO, GENRE, g)

    def _clean(self):
        for etype in self.G:
            for eid in self.G[etype]:
                for rel in self.G[etype][eid]:
                    nbrs = sorted(set(self.G[etype][eid][rel]))
                    self.G[etype][eid][rel] = tuple(nbrs)

    def compute_degrees(self):
        deg = {etype: np.zeros(len(self.G[etype]), dtype=np.int64) for etype in self.G}
        for etype in self.G:
            for i, rels in self.G[etype].items():
                deg[etype][i] = sum(len(rels[r]) for r in rels)
        self.degrees = deg
        return deg

    # helpers
    def neighbors(self, etype, idx, rel):
        return self.G[etype][idx][rel]
