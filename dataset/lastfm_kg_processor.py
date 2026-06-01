from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set
import numpy as np
from tqdm import tqdm

from utils.rw_process import *


class KGPathExtractor:
    """
    Extracts multi-hop paths from Last.fm knowledge graph between user and artist entities.
    """
    
    def __init__(self, kg, d_max=3, k_top=50):
        """
        Args:
            kg: KnowledgeGraphLastFM object
            d_max: Maximum path depth (default: 3)
            k_top: Top-k paths to retain per depth (default: 50)
        """
        self.kg = kg
        self.d_max = d_max
        self.k_top = k_top
        self.user_pattern_cache = {}  # Cache for user's historical pattern frequencies
        
    def extract_paths(self, user_id, artist_id, user_history=None, use_pattern_scoring=True):
        """
        Extract 2-hop and 3-hop paths between user and artist.
        
        Args:
            user_id: User entity ID
            artist_id: Artist entity ID
            user_history: List of artist IDs in user's history (for frequency-based filtering)
            use_pattern_scoring: If True, score paths by user's historical patterns
            
        Returns:
            paths_2hop: List of 2-hop paths [(user, r1, e_mid, r2, artist), ...]
            paths_3hop: List of 3-hop paths [(user, r1, e1, r2, e2, r3, artist), ...]
        """
        
        # Extract raw paths
        paths_2hop_raw = self._find_all_paths(user_id, artist_id, depth=2)
        paths_3hop_raw = self._find_all_paths(user_id, artist_id, depth=3)
        
        # Apply pruning
        paths_2hop = self._prune_paths(
            paths_2hop_raw, 
            user_id, 
            user_history, 
            depth=2,
            use_pattern_scoring=use_pattern_scoring
        )
        paths_3hop = self._prune_paths(
            paths_3hop_raw, 
            user_id, 
            user_history, 
            depth=3,
            use_pattern_scoring=use_pattern_scoring
        )
        
        return paths_2hop, paths_3hop
    
    def extract_user_pattern_profile(self, user_id, user_history):
        """
        Analyze user's historical interaction patterns by extracting paths
        between user and their historical artists.
        
        FOR ANALYST AGENT - to understand user's music discovery patterns.
        
        Args:
            user_id: User entity ID
            user_history: List of artist IDs user has interacted with
            
        Returns:
            pattern_freq: Dict mapping relation patterns to frequencies
                         e.g., {('listen', 'belong_to'): 0.45, ...}
        """

        cache_key = f"{user_id}"
        if cache_key in self.user_pattern_cache:
            return self.user_pattern_cache[cache_key]
        
        all_historical_patterns = []
        
        # Extract paths between user and each historical artist
        for hist_artist_id in tqdm(user_history, desc="Analyzing historical artists", leave=False):
            # Get paths for this historical interaction
            paths_2hop_raw = self._find_all_paths(user_id, hist_artist_id, depth=2)
            paths_3hop_raw = self._find_all_paths(user_id, hist_artist_id, depth=3)
            
            # Only structural filtering (remove cycles)
            paths_2hop_clean = self._remove_cycles(paths_2hop_raw)
            paths_3hop_clean = self._remove_cycles(paths_3hop_raw)
            
            # Extract patterns from these paths
            for path in paths_2hop_clean + paths_3hop_clean:
                pattern = self._extract_relation_pattern(path)
                all_historical_patterns.append(pattern)
        
        # Compute frequency distribution
        pattern_counts = defaultdict(int)
        for pattern in all_historical_patterns:
            pattern_counts[pattern] += 1
        
        # Normalize to probabilities
        total = sum(pattern_counts.values())
        if total > 0:
            pattern_freq = {k: v/total for k, v in pattern_counts.items()}
        else:
            pattern_freq = {}
        
        # Cache for future use
        self.user_pattern_cache[cache_key] = pattern_freq
        
        return pattern_freq
    
    def _find_all_paths(self, source_id, target_id, depth):
        """
        Find all paths of specified depth using BFS.
        Path format: [(entity_type_1, entity_id_1, relation_1, entity_type_2, entity_id_2), ...]
        """
        if depth not in [2, 3]:
            raise ValueError("Only depths 2 and 3 are supported")
        
        paths = []
        
        # BFS with path tracking
        # Queue items: (current_entity_type, current_entity_id, current_depth, path_so_far)
        queue = deque([(USER, source_id, 0, [])])
        
        while queue:
            curr_type, curr_id, curr_depth, path = queue.popleft()
            
            if curr_depth == depth:
                if curr_type == ARTIST and curr_id == target_id:
                    paths.append(path)
                continue

            if curr_depth >= depth:
                continue
            if curr_type not in self.kg.G or curr_id not in self.kg.G[curr_type]:
                continue
                
            for relation, tail_ids in self.kg.G[curr_type][curr_id].items():
                if len(tail_ids) == 0:
                    continue

                tail_type = get_tail_type(curr_type, relation)
                if tail_type is None:
                    continue
                
                for tail_id in tail_ids:
                    if len(path) > 0 and tail_id == path[-1][1]:
                        continue
                    
                    new_path = path + [(curr_type, curr_id, relation, tail_type, tail_id)]
                    queue.append((tail_type, tail_id, curr_depth + 1, new_path))
        
        return paths
    
    def _prune_paths(self, paths, user_id, user_history, depth, use_pattern_scoring=True):
        """
        Apply path pruning based on:
        1. Structural filtering (remove cycles, redundant sub-paths)
        2. Frequency-based filtering (prioritize patterns common in user's historical behavior)
        3. Top-k selection per depth
        """
        if len(paths) == 0:
            return paths
        
        # Step 1: Structural filtering - remove paths with cycles
        paths_no_cycles = self._remove_cycles(paths)
        
        # Step 2: Frequency-based filtering
        if user_history and len(user_history) > 0 and use_pattern_scoring:
            # Get user's historical pattern profile
            historical_pattern_freq = self.extract_user_pattern_profile(user_id, user_history)
            
            # Score each path by historical pattern preference
            path_scores = []
            for path in paths_no_cycles:
                pattern = self._extract_relation_pattern(path)
                score = historical_pattern_freq.get(pattern, 0.0)
                path_scores.append((path, score))
            
            # Sort by score (descending)
            path_scores.sort(key=lambda x: x[1], reverse=True)
            paths_scored = [p for p, s in path_scores]
        else:
            # Fallback: score by frequency in current candidate paths
            pattern_scores = self._compute_pattern_frequencies_in_history(paths_no_cycles)
            
            path_scores = []
            for path in paths_no_cycles:
                pattern = self._extract_relation_pattern(path)
                score = pattern_scores.get(pattern, 0.0)
                path_scores.append((path, score))
            
            path_scores.sort(key=lambda x: x[1], reverse=True)
            paths_scored = [p for p, s in path_scores]
        
        # Step 3: Select top-k
        if len(paths_scored) > self.k_top:
            return paths_scored[:self.k_top]
        
        return paths_scored
    
    def _remove_cycles(self, paths):
        """
        Remove paths that contain cycles (any entity appearing more than once).
        """
        paths_no_cycles = []
        for path in paths:
            entity_ids = [step[1] for step in path] + [path[-1][4]]  # All entity IDs in path
            if len(entity_ids) == len(set(entity_ids)):  # No duplicates = no cycles
                paths_no_cycles.append(path)
        return paths_no_cycles
    
    def _extract_relation_pattern(self, path):
        """Extract relation pattern from path: (r1, r2) or (r1, r2, r3)"""
        return tuple(step[2] for step in path)
    
    def _compute_pattern_frequencies_in_history(self, paths):
        """
        Compute pattern frequencies within the candidate paths themselves.
        This is a fallback when we don't have user history.
        """
        pattern_scores = defaultdict(float)
        for path in paths:
            pattern = self._extract_relation_pattern(path)
            pattern_scores[pattern] += 1.0
        
        # Normalize
        total = sum(pattern_scores.values())
        if total > 0:
            pattern_scores = {k: v/total for k, v in pattern_scores.items()}
        
        return pattern_scores


class KGPathTranslator:
    """
    Translates KG paths into natural language descriptions for Last.fm domain.
    """
    
    def __init__(self, kg, dataset):
        """
        Args:
            kg: KnowledgeGraphLastFM object
            dataset: Dataset object with vocab lookups (from build_vocabs)
        """
        self.kg = kg
        self.dataset = dataset
        
    def translate_2hop(self, paths_2hop):
        """
        Translate 2-hop paths using grouping and consolidation.
        
        Groups paths by relation pattern and formats as natural language.
        
        Args:
            paths_2hop: List of 2-hop paths
            
        Returns:
            text: Natural language description
            grouped_patterns: Dictionary of grouped paths by pattern
        """
        if len(paths_2hop) == 0:
            return "", {}
        
        # Group paths by relation pattern: (r1, r2)
        grouped = defaultdict(list)
        for path in paths_2hop:
            pattern = self._extract_pattern(path)
            grouped[pattern].append(path)
        
        # Generate text for each pattern group
        text_parts = []
        for pattern, pattern_paths in grouped.items():
            r1, r2 = pattern
            
            # Extract intermediate entities for this pattern
            mid_entities = set()
            for path in pattern_paths:
                # path format: [(user, uid, r1, mid_type, mid_id), (mid_type, mid_id, r2, artist_type, artist_id)]
                mid_type, mid_id = path[0][3], path[0][4]
                mid_entities.add((mid_type, mid_id))
            
            # Format entities
            entity_names = []
            for mid_type, mid_id in mid_entities:
                if mid_type == TAG:
                    continue
                name = self._get_entity_name(mid_type, mid_id)
                if name:
                    entity_names.append(name)
            
            if len(entity_names) > 0:
                # Format relation for user subject
                r1_formatted = self._format_relation(r1, subject='user')
                r2_formatted = self._format_relation(r2, subject='they')
                
                # Create natural language sentence
                entities_str = ", ".join(entity_names[:10])  # Limit to top 10
                if len(entity_names) > 10:
                    entities_str += f" (and {len(entity_names) - 10} more)"
                
                text_parts.append(f"The user {r1_formatted} features such as {entities_str} which {r2_formatted} this artist.")
        
        text = " ".join(text_parts)
        return text, grouped
    
    def translate_3hop(self, paths_3hop, user_history_liked=None, user_history_disliked=None):
        """
        Translate 3-hop paths using discriminative entity extraction.
        
        Extracts discriminative entities that distinguish liked from disliked artists.
        
        Args:
            paths_3hop: List of 3-hop paths
            user_history_liked: List of liked artist IDs
            user_history_disliked: List of disliked artist IDs
            
        Returns:
            text: Natural language description
            discriminative_info: Dictionary with positive/negative discriminative entities
        """
        if len(paths_3hop) == 0:
            return "", {}
        
        # Extract discriminative entities from liked/disliked items
        if user_history_liked and len(user_history_liked) > 0:
            positive_disc, negative_disc = self._extract_discriminative_sets(
                user_history_liked, 
                user_history_disliked or []
            )
        else:
            positive_disc, negative_disc = set(), set()
        
        # Filter entities from paths that are discriminative
        current_positive = self._extract_discriminative_entities(paths_3hop, positive_disc)
        current_negative = self._extract_discriminative_entities(paths_3hop, negative_disc)
        
        # Generate text
        text_parts = []
        
        if len(current_positive) > 0:
            entity_names = []
            for entity_type, entity_id in current_positive:
                if entity_type == TAG:
                    continue
                name = self._get_entity_name(entity_type, entity_id)
                if name:
                    entity_names.append(name)
            
            if len(entity_names) > 0:
                entities_str = ", ".join(entity_names[:15])
                if len(entity_names) > 15:
                    entities_str += f" (and {len(entity_names) - 15} more)"
                text_parts.append(f"This artist shares characteristics with artists the user has previously enjoyed: {entities_str}.")
        
        if len(current_negative) > 0:
            entity_names = []
            for entity_type, entity_id in current_negative:
                if entity_type == TAG:
                    continue
                name = self._get_entity_name(entity_type, entity_id)
                if name:
                    entity_names.append(name)
            
            if len(entity_names) > 0:
                entities_str = ", ".join(entity_names[:15])
                if len(entity_names) > 15:
                    entities_str += f" (and {len(entity_names) - 15} more)"
                text_parts.append(f"This artist shares characteristics with artists the user has previously disliked: {entities_str}.")
        
        text = " ".join(text_parts)
        
        discriminative_info = {
            'positive': positive_disc,
            'negative': negative_disc,
            'current_positive': current_positive,
            'current_negative': current_negative,
        }
        
        return text, discriminative_info
    
    def _extract_pattern(self, path):
        """Extract relation pattern (r1, r2) or (r1, r2, r3) from path."""
        return tuple(step[2] for step in path)
    
    def _extract_discriminative_sets(self, liked_artists, disliked_artists, tau_min=0.3, tau_max=0.7):
        """
        Extract positive and negative discriminative entity sets.
        
        Positive discriminative: frequent in liked, rare in disliked
        Negative discriminative: frequent in disliked, rare in liked
        """
        # Compute entity frequencies in liked artists
        liked_freq = defaultdict(float)
        for artist_id in liked_artists:
            entities = self._extract_descriptive_entities(ARTIST, artist_id)
            for entity in entities:
                liked_freq[entity] += 1.0
        
        # Normalize
        if len(liked_artists) > 0:
            liked_freq = {k: v/len(liked_artists) for k, v in liked_freq.items()}
        
        # Compute entity frequencies in disliked artists
        disliked_freq = defaultdict(float)
        for artist_id in disliked_artists:
            entities = self._extract_descriptive_entities(ARTIST, artist_id)
            for entity in entities:
                disliked_freq[entity] += 1.0
        
        # Normalize
        if len(disliked_artists) > 0:
            disliked_freq = {k: v/len(disliked_artists) for k, v in disliked_freq.items()}
        
        # Build positive discriminative set:
        # - High frequency in liked artists (tau_min <= freq <= tau_max)
        # - Low frequency in disliked artists (freq < tau_min) OR not present
        positive_discriminative = set()
        for entity, freq_liked in liked_freq.items():
            freq_disliked = disliked_freq.get(entity, 0.0)
            
            if tau_min <= freq_liked <= tau_max and freq_disliked < tau_min:
                positive_discriminative.add(entity)
        
        # Build negative discriminative set:
        # - High frequency in disliked artists (tau_min <= freq <= tau_max)
        # - Low frequency in liked artists (freq < tau_min) OR not present
        negative_discriminative = set()
        if len(disliked_artists) > 0:
            for entity, freq_disliked in disliked_freq.items():
                freq_liked = liked_freq.get(entity, 0.0)
                
                if tau_min <= freq_disliked <= tau_max and freq_liked < tau_min:
                    negative_discriminative.add(entity)
        
        return positive_discriminative, negative_discriminative
    
    def _extract_descriptive_entities(self, entity_type, entity_id):
        """
        Extract descriptive entities connected to given entity.
        Descriptive entities: TAG, GENRE
        """
        descriptive_entities = set()
        
        if entity_type not in self.kg.G or entity_id not in self.kg.G[entity_type]:
            return descriptive_entities
        
        # Define which entity types are "descriptive" for Last.fm
        descriptive_types = {TAG, GENRE}
        
        for relation, tail_ids in self.kg.G[entity_type][entity_id].items():
            tail_type = get_tail_type(entity_type, relation)
            
            # Only include descriptive entity types
            if tail_type in descriptive_types:
                for tail_id in tail_ids:
                    if tail_type == TAG:
                        continue
                    descriptive_entities.add((tail_type, tail_id))
        
        return descriptive_entities
    
    def _extract_discriminative_entities(self, paths, discriminative_set):
        """
        Extract entities from paths that are in the discriminative set.
        """
        filtered_entities = set()
        
        for path in paths:
            for step in path:
                # Each step: (entity_type, entity_id, relation, tail_type, tail_id)
                # Check both head and tail of each step
                entity_head = (step[0], step[1])
                entity_tail = (step[3], step[4])
                
                if entity_head in discriminative_set:
                    filtered_entities.add(entity_head)
                if entity_tail in discriminative_set:
                    filtered_entities.add(entity_tail)
        
        return filtered_entities
    
    def _get_entity_name(self, entity_type, entity_id):
        """Get human-readable name for entity."""
        try:
            if entity_type == USER:
                return f"User{entity_id}"
            elif entity_type == ARTIST:
                if hasattr(self.dataset, 'artist') and entity_id < len(self.dataset.artist.vocab):
                    return self.dataset.artist.vocab[entity_id]
            elif entity_type == TAG:
                if hasattr(self.dataset, 'tag') and entity_id < len(self.dataset.tag.vocab):
                    return self.dataset.tag.vocab[entity_id]
            elif entity_type == GENRE:
                if hasattr(self.dataset, 'genre') and entity_id < len(self.dataset.genre.vocab):
                    return self.dataset.genre.vocab[entity_id]
        except Exception as e:
            print(f"Error getting name for {entity_type} {entity_id}: {e}")
            return None
        
        return f"{entity_type}_{entity_id}"
    
    def _format_relation(self, relation, subject='user'):
        """Format relation name for natural language."""
        relation_formats = {
            LISTEN: ("listened to", "are listened to by"),
            TAGGED: ("tagged", "are tagged with"),
            DESCRIBED_AS: ("is described by", "describe"),
            FRIENDS_WITH: ("is friends with", "are friends with"),
            BELONG_TO: ("listens to genres", "belong to"),
        }
        
        if subject == 'user':
            return relation_formats.get(relation, (relation, relation))[0]
        else:  # 'they' or other
            return relation_formats.get(relation, (relation, relation))[1]


class KGPathProcessor:
    """
    Main interface combining extraction and translation for Last.fm.
    
    Supports three-stage multi-agent workflow:
    1. Analyst Agent: process_user_history()
    2. Rec Agent: process_candidate_items()
    """
    
    def __init__(self, kg, dataset, d_max=3, k_top=50):
        self.extractor = KGPathExtractor(kg, d_max, k_top)
        self.translator = KGPathTranslator(kg, dataset)
        self.kg = kg
        self.dataset = dataset
    
    def process_user_history(self, user_id, user_history_liked, user_history_disliked=None):
        """
        FOR ANALYST AGENT: Extract and translate paths for user's historical artists.
        This gives the Analyst Agent a profile of the user's musical preferences.
        
        Args:
            user_id: User ID
            user_history_liked: List of artist IDs user LIKED
            user_history_disliked: List of artist IDs user DISLIKED (optional)
        
        Returns:
            results: Dictionary with:
                - pattern_profile: User's historical pattern frequencies
                - historical_descriptions: List of dicts with artist descriptions
                    [{'artist_id': ..., 'text_2hop': ..., 'text_3hop': ...}, ...]
                - stats: Statistics about the user's history
        """

        
        # Step 1: Extract user's pattern profile
        all_history = user_history_liked + (user_history_disliked or [])
        pattern_profile = self.extractor.extract_user_pattern_profile(user_id, all_history)
        
        # Step 2: Extract and translate paths for each historical artist
        historical_descriptions = []
        
        for artist_id in tqdm(user_history_liked, desc="Processing liked artists"):
            # Extract paths (without pattern scoring to avoid circularity)
            paths_2hop, paths_3hop = self.extractor.extract_paths(
                user_id, 
                artist_id, 
                user_history=None,  # not using history for historical analysis
                use_pattern_scoring=False
            )
            
            # Translate paths
            text_2hop, grouped_2hop = self.translator.translate_2hop(paths_2hop)
            text_3hop, discriminative_info = self.translator.translate_3hop(
                paths_3hop,
                user_history_liked,
                user_history_disliked or []
            )
            
            historical_descriptions.append({
                'artist_id': artist_id,
                'artist_name': self.translator._get_entity_name(ARTIST, artist_id),
                'text_2hop': text_2hop,
                'text_3hop': text_3hop,
                'n_paths_2hop': paths_2hop,
                'n_paths_3hop': paths_3hop,
            })
        
        # Also process disliked artists if available
        if user_history_disliked:
            for artist_id in tqdm(user_history_disliked, desc="Processing disliked artists"):
                paths_2hop, paths_3hop = self.extractor.extract_paths(
                    user_id,
                    artist_id,
                    user_history=None,
                    use_pattern_scoring=False
                )
                
                text_2hop, grouped_2hop = self.translator.translate_2hop(paths_2hop)
                text_3hop, discriminative_info = self.translator.translate_3hop(
                    paths_3hop,
                    user_history_liked,
                    user_history_disliked
                )
                
                historical_descriptions.append({
                    'artist_id': artist_id,
                    'artist_name': self.translator._get_entity_name(ARTIST, artist_id),
                    'text_2hop': text_2hop,
                    'text_3hop': text_3hop,
                    'n_paths_2hop': len(paths_2hop),
                    'n_paths_3hop': len(paths_3hop),
                    'is_disliked': True,
                })
        
        stats = {
            'n_liked': len(user_history_liked),
            'n_disliked': len(user_history_disliked) if user_history_disliked else 0,
            'n_patterns': len(pattern_profile),
            'total_artists_processed': len(historical_descriptions),
        }
        
        results = {
            'user_id': user_id,
            'pattern_profile': pattern_profile,
            'historical_descriptions': historical_descriptions,
            'stats': stats,
        }
        
        return results
    
    def process_candidate_items(self, user_id, candidate_artist_ids, user_history_liked, user_history_disliked=None):
        """
        FOR REC AGENT: Extract and translate paths for candidate artists.
        Paths are scored by the user's historical pattern preferences.
        
        Args:
            user_id: User ID
            candidate_artist_ids: List of candidate artist IDs
            user_history_liked: List of artist IDs user LIKED
            user_history_disliked: List of artist IDs user DISLIKED (optional)
        
        Returns:
            results: Dictionary with:
                - candidate_descriptions: List of dicts with artist descriptions
                    [{'artist_id': ..., 'text_2hop': ..., 'text_3hop': ..., 'score': ...}, ...]
                - stats: Statistics about candidate processing
        """
        
        all_history = user_history_liked + (user_history_disliked or [])
        
        candidate_descriptions = []
        
        for artist_id in tqdm(candidate_artist_ids, desc="Processing candidates"):
            # Extract paths with pattern scoring based on user's history
            paths_2hop, paths_3hop = self.extractor.extract_paths(
                user_id,
                artist_id,
                user_history=all_history,
                use_pattern_scoring=False  # Use historical pattern preferences
            )
            
            # Translate paths
            text_2hop, grouped_2hop = self.translator.translate_2hop(paths_2hop)
            text_3hop, discriminative_info = self.translator.translate_3hop(
                paths_3hop,
                user_history_liked,
                user_history_disliked or []
            )
            
            # Compute a relevance score based on path patterns
            pattern_profile = self.extractor.user_pattern_cache.get(f"{user_id}", {})
            relevance_score = 0.0
            if len(paths_2hop) + len(paths_3hop) > 0:
                for path in paths_2hop + paths_3hop:
                    pattern = self.extractor._extract_relation_pattern(path)
                    relevance_score += pattern_profile.get(pattern, 0.0)
                relevance_score /= (len(paths_2hop) + len(paths_3hop))
            
            candidate_descriptions.append({
                'artist_id': artist_id,
                'artist_name': self.translator._get_entity_name(ARTIST, artist_id),
                'text_2hop': text_2hop,
                'text_3hop': text_3hop,
                'n_paths_2hop': paths_2hop,
                'n_paths_3hop': paths_3hop,
                'relevance_score': relevance_score,
                'discriminative_info': discriminative_info,
            })
        
        # Sort by relevance score
        candidate_descriptions.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        stats = {
            'n_candidates': len(candidate_artist_ids),
            'avg_paths_2hop': np.mean([len(d['n_paths_2hop']) for d in candidate_descriptions]),
            'avg_paths_3hop': np.mean([len(d['n_paths_3hop']) for d in candidate_descriptions]),
            'avg_relevance_score': np.mean([d['relevance_score'] for d in candidate_descriptions]),
        }
        
        results = {
            'user_id': user_id,
            'candidate_descriptions': candidate_descriptions,
            'stats': stats,
        }

        return results


# Helper function to get tail entity type for a given head entity type and relation
def get_tail_type(head_type, relation):
    """
    Map (head_entity_type, relation) -> tail_entity_type
    Based on Last.fm KG schema
    """
    mapping = {
        USER: {
            LISTEN: ARTIST,
            TAGGED: TAG,
            FRIENDS_WITH: USER,
        },
        ARTIST: {
            LISTEN: USER,
            DESCRIBED_AS: TAG,
            BELONG_TO: GENRE,
            IS_FROM: COUNTRY,
            PERFORMS_IN: LANGUAGE,
        },
        TAG: {
            TAGGED: USER,
            DESCRIBED_AS: ARTIST,
        },
        GENRE: {
            BELONG_TO: ARTIST,
        },
        COUNTRY: {
            IS_FROM: ARTIST,
        },
        LANGUAGE: {
            PERFORMS_IN: ARTIST,
        },
    }
    
    return mapping.get(head_type, {}).get(relation, None)
