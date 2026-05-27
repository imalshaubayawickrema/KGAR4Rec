import re
import ast

_TRUNC = 240  # how much to show if we must print any debug

def _trim_to_reason_or_item(text: str) -> str:
    """If the model echoed prompts, start from the first Reason:/Item: label."""
    if not text:
        return ""
    i = text.find("Reason:")
    if i == -1:
        i = text.find("Item:")
    return text[i:] if i != -1 else text

def _dbg(label: str, payload: str, quiet: bool = True):
    """Quiet by default. Flip quiet=False while debugging."""
    if not quiet:
        p = payload.replace("\r", "\n")
        print(f"{label}", (p[:_TRUNC] + ("..." if len(p) > _TRUNC else "")))


def split_analyst_response(response: str, quiet: bool = True):
    if not response:
        _dbg("[split_analyst_response] empty response", "", quiet)
        return None

    text = _trim_to_reason_or_item(response.strip()) + "\n"
    m_summary = re.search(r"^Summary:\s*(.+?)(?=\n[A-Z][a-zA-Z]+:|\n$)", text, flags=re.M | re.S)

    if not m_summary:
        _dbg("[split_analyst_response] cannot split", text, quiet)
        return None

    summary = m_summary.group(1).strip()
    return summary

def split_rec_ranking(response: str, quiet: bool = True):
    """
    Expect:
      Reason: ...
      RankedList: ["artist1", "artist2", ...]   # preferred
    but also tolerate: [artist1, artist2, ...]   # no quotes
    Returns: (reason:str|None, ranking:list[str])
    """
    if not response:
        _dbg("[split_rec_ranking] empty response", "", quiet)
        return None, [], False

    text = _trim_to_reason_or_item(response.strip()) + "\n"

    m_reason = re.search(r"^Reason:\s*(.+?)(?=\nRankedList:)",
                         text, flags=re.M | re.S)
    if m_reason:
        reason_block = m_reason.group(1).strip()
        # Detect per-artist format: multiple "Reason:" lines
        # Collapse them into a single paragraph
        if re.search(r"\n+Reason:", reason_block):
            # Per-artist format — split on one or more newlines before "Reason:"
            parts = re.split(r"\n+Reason:\s*", reason_block)
            reason = " ".join(p.strip() for p in parts if p.strip())
        else:
            reason = reason_block
    else:
        reason = None

    m_rank = re.search(r"^RankedList:\s*(\[.+\])",
                       text, flags=re.M | re.S)
    if not m_rank:
        _dbg("[split_rec_ranking] no RankedList block", text, quiet)
        return reason, [], False

    ranking_raw = m_rank.group(1).strip()

    # 1) Try strict Python literal first
    try:
        parsed = ast.literal_eval(ranking_raw)
        if isinstance(parsed, list):
            ranking = [str(x).strip() for x in parsed]
            return reason, ranking
    except Exception:
        pass

    # 2) Fallbacks for sloppy output (no quotes / newlines)
    #    Extract inside [ ... ]
    inner_m = re.match(r"^\[\s*(.*)\s*\]$", ranking_raw, flags=re.S)
    if not inner_m:
        _dbg("[split_rec_ranking] malformed brackets", ranking_raw, quiet)
        return reason, [], False

    inner = inner_m.group(1).strip()

    # 2a) If there ARE quoted chunks, use them
    quoted = re.findall(r'["“”\']\s*([^"“”\']+?)\s*["“”\']', inner)
    if quoted:
        ranking = [q.strip() for q in quoted if q.strip()]
        return reason, ranking

    # 2b) Otherwise split by commas (best-effort)
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    return reason, parts


def split_user_response(response, rec_items):
    """
    Parse UserAgent evaluations for recommended items.

    Returns:
        evaluations: list of dicts
        num_liked: number of liked items
        overall_decision: bool
    """
    evaluations = []
    if not response or not rec_items:
        return [], 0, False
    top5_items = rec_items[:]

    for i, item in enumerate(top5_items):
        # Match pattern: "Feedback on {item}: ... Decision: yes/no"
        pattern = (
            rf"Feedback on\s+{re.escape(item)}:\s*(.*?)\s*"
            rf"Decision:\s*(like|dislike)"
        )

        match = re.search(
            pattern,
            response,
            flags=re.IGNORECASE | re.DOTALL
        )

        if match:
            feedback = match.group(1).strip()
            decision = match.group(2).strip().lower()
            parsed = True
        else:
            feedback = None
            decision = "unknown"
            parsed = False

        evaluations.append({
            "item": item,
            "position": i + 1,
            "feedback": feedback,
            "decision": decision,
            "parsed": parsed
        })

    num_liked = sum(
        1 for e in evaluations
        if e["decision"] == "like"
    )

    parsed_count = sum(
        1 for e in evaluations
        if e["parsed"]
    )

    overall_decision = (
            parsed_count > 0 and
            (num_liked / parsed_count) > 0.5
    )

    return evaluations, num_liked, overall_decision


def format_hist_kg_descriptions(hist_descriptions, max_items=10, hop_filter='both'):
    """
    Format historical item KG descriptions for Analyst Agent.

    Args:
        hist_descriptions: List of dicts from processor.process_user_history()
        max_items: Maximum number of items to include
        hop_filter: the type of KG paths ['2hop', '3hop', 'both']

    Returns:
        Formatted string for prompt
    """
    if not hist_descriptions or len(hist_descriptions) == 0:
        return "No knowledge graph patterns available for historical items."

    show_2hop = hop_filter in ('2hop', 'both')
    show_3hop = hop_filter in ('3hop', 'both')

    lines = []
    for i, desc in enumerate(hist_descriptions[:max_items]):
        product_name = desc.get('artist_name', f"Item {desc['artist_id']}")
        is_disliked = desc.get('is_disliked', False)
        sentiment = "DISLIKED" if is_disliked else "LIKED"

        has_2hop = show_2hop and desc.get('text_2hop')
        has_3hop = show_3hop and desc.get('text_3hop')

        if has_2hop or has_3hop:
            lines.append(f"\n{i + 1}. {product_name} ({sentiment}):")

        # 2-hop connections
        if has_2hop:
            lines.append(f"   - Direct connections: {desc['text_2hop']}")
        if has_3hop:
            lines.append(f"   - Deeper associations: {desc['text_3hop']}")

    if len(hist_descriptions) > max_items:
        lines.append(f"\n... and {len(hist_descriptions) - max_items} more items")

    return "\n".join(lines)


def format_cand_kg_descriptions(cand_descriptions, max_items=20, hop_filter='both'):
    """
    Format candidate item KG descriptions for Rec Agent.

    Args:
        cand_descriptions: List of dicts from processor.process_candidate_items()
        max_items: Maximum number of candidates to include
        hop_filter: the type of KG paths ['2hop', '3hop', 'both']

    Returns:
        Formatted string for prompt
    """
    if not cand_descriptions or len(cand_descriptions) == 0:
        return "No knowledge graph explanations available for candidates."

    show_2hop = hop_filter in ('2hop', 'both')
    show_3hop = hop_filter in ('3hop', 'both')

    lines = []
    for i, desc in enumerate(cand_descriptions[:max_items]):
        product_name = desc.get('artist_name', f"Item {desc['artist_id']}")
        has_2hop = show_2hop and desc.get('text_2hop')
        has_3hop = show_3hop and desc.get('text_3hop')

        if has_2hop or has_3hop:
            lines.append(f"\n{i + 1}. {product_name}:")
        if has_2hop:
            lines.append(f"   - Direct connections: {desc['text_2hop']}")
        if has_3hop:
            lines.append(f"   - Preference alignment: {desc['text_3hop']}")

    if len(cand_descriptions) > max_items:
        lines.append(f"\n... and {len(cand_descriptions) - max_items} more candidates")

    return "\n".join(lines)


def format_rec_kg_descriptions(cand_descriptions, recommended_product_ids, perspective='you', hop_filter='both'):
    """
    Format KG descriptions for ONLY the recommended items (for User Agent).

    Args:
        cand_descriptions: List of dicts from processor.process_candidate_items()
        recommended_product_ids: List of product IDs that were recommended

    Returns:
        Formatted string for prompt
    """
    if not cand_descriptions or not recommended_product_ids:
        return "No knowledge graph explanations available for recommendations."

    show_2hop = hop_filter in ('2hop', 'both')
    show_3hop = hop_filter in ('3hop', 'both')

    # Create lookup by product_id
    desc_by_id = {desc['artist_id']: desc for desc in cand_descriptions}

    lines = []
    for i, prod_id in enumerate(recommended_product_ids):  # Top 3 only for user agent
        if prod_id not in desc_by_id:
            continue

        desc = desc_by_id[prod_id]
        product_name = desc.get('artist_name', f"Item {prod_id}")
        has_2hop = show_2hop and desc.get('text_2hop')
        has_3hop = show_3hop and desc.get('text_3hop')

        if has_2hop or has_3hop:
            lines.append(f"\n{i + 1}. {product_name}:")
        if has_2hop:
            text_2hop = _convert_perspective(desc['text_2hop'], perspective)
            lines.append(f"   - Direct connections: {text_2hop}")
        if has_3hop:
            text_3hop = _convert_perspective(desc['text_3hop'], perspective)
            lines.append(f"   - Preference alignment: {text_3hop}")

    return "\n".join(lines)

def _convert_perspective(text, perspective='you'):
    """
    Convert text from third-person ('user'/'the user') to first-person ('you') or vice versa.
    
    Args:
        text: Original text
        perspective: 'you' for first-person or 'user' for third-person
        
    Returns:
        Converted text
    """
    if perspective == 'you':
        # Convert third-person to first-person
        text = text.replace('The user ', 'You ')
        text = text.replace('the user ', 'you ')
        text = text.replace('This user ', 'You ')
        text = text.replace('this user ', 'you ')
        text = text.replace(' user ', ' you ')
        
        # Handle verb conjugations
        text = text.replace('You purchased', 'You purchased')
        text = text.replace('You also bought', 'You also bought')
        text = text.replace('You also viewed', 'You also viewed')
        text = text.replace('You mentions', 'You mention')
        text = text.replace('You prefers', 'You prefer')
    return text

def format_pattern_profile_summary(pattern_profile, top_k=5):
    """
    Format user's pattern profile for display (optional, for debugging/analysis).
    
    Args:
        pattern_profile: Dict from processor.process_user_history()['pattern_profile']
        top_k: Number of top patterns to show
        
    Returns:
        Formatted string
    """
    if not pattern_profile or len(pattern_profile) == 0:
        return "No pattern profile available."
    
    lines = ["User's Top Discovery Patterns:"]
    sorted_patterns = sorted(pattern_profile.items(), key=lambda x: x[1], reverse=True)
    
    for i, (pattern, freq) in enumerate(sorted_patterns[:top_k]):
        pattern_str = " → ".join(pattern)
        lines.append(f"{i+1}. {pattern_str}: {freq:.1%}")
    
    return "\n".join(lines)

