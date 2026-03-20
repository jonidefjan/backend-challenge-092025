import re
import math
import hashlib
import unicodedata
import time as _time
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# Tokenization regex: hashtags with hyphens stay as single tokens
TOKEN_RE = re.compile(r'(?:#\w+(?:-\w+)*)|\b\w+\b', re.UNICODE)

# Lexicon: normalized form → score
LEXICON = {
    'adorei': 1.0,
    'amei': 1.5,
    'bom': 1.0,
    'otimo': 1.5,
    'excelente': 2.0,
    'maravilhoso': 2.0,
    'perfeito': 2.0,
    'lindo': 1.0,
    'fantastico': 1.5,
    'incrivel': 1.5,
    'legal': 0.5,
    'top': 1.0,
    'gostei': 1.0,
    'sensacional': 1.5,
    'espetacular': 1.5,
    'ruim': -1.0,
    'pessimo': -2.0,
    'horrivel': -2.0,
    'terrivel': -1.5,
    'odiei': -1.5,
    'detestei': -1.5,
    'feio': -1.0,
    'triste': -1.0,
    'nojento': -1.5,
    'chato': -0.5,
    'fraco': -0.5,
    'pior': -1.5,
    'mal': -1.0,
    'problema': -0.5,
    'falha': -1.0,
}

INTENSIFIERS = {'muito', 'bastante', 'extremamente', 'super', 'demais', 'incrivelmente', 'totalmente'}
NEGATIONS = {'nao', 'nunca', 'jamais', 'nem', 'tampouco', 'sequer'}

PHI = (1 + math.sqrt(5)) / 2

# Special follower values per algorithmic rules
UNICODE_USER_FOLLOWERS = 4242        # Unicode user_id → fixed follower count
FIBONACCI_13_FOLLOWERS = 233         # 13-char user_id → 13th Fibonacci number
SHA_FOLLOWERS_MODULO = 10000         # Base modulo for SHA-256 follower count
SHA_FOLLOWERS_OFFSET = 100           # Minimum follower count

# Trending topics
HASHTAG_LOG_LENGTH_THRESHOLD = 8     # Hashtags longer than this get log-scale weight
TRENDING_TOP_N = 5                   # Number of trending topics to return

# Anomaly detection thresholds
BURST_THRESHOLD = 10                 # Messages per user in window to trigger burst
BURST_WINDOW_MINUTES = 5
ALTERNATING_THRESHOLD = 10           # Consecutive alternating sentiments for anomaly
SYNCHRONIZED_WINDOW_SECONDS = 4     # ±2s tolerance = 4s total window
SYNCHRONIZED_MIN_MESSAGES = 3

# Engagement score override when candidate_awareness flag is active
CANDIDATE_ENGAGEMENT_SCORE = 9.42

# Special pattern content length
SPECIAL_PATTERN_LENGTH = 42

# Future timestamp tolerance (seconds)
FUTURE_TOLERANCE_SECONDS = 5


def normalize_for_matching(token: str) -> str:
    lower = token.lower()
    nfkd = unicodedata.normalize('NFKD', lower)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def tokenize(content: str) -> list:
    return TOKEN_RE.findall(content)


def _is_unicode_user_id(user_id: str) -> bool:
    """Return True if user_id contains any non-ASCII character."""
    try:
        user_id.encode('ascii')
        return False
    except UnicodeEncodeError:
        return True


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def _nearest_prime(n: int) -> int:
    if n < 2:
        return 2
    if _is_prime(n):
        return n
    lower, upper = n - 1, n + 1
    while True:
        if lower >= 2 and _is_prime(lower):
            return lower
        if _is_prime(upper):
            return upper
        lower -= 1
        upper += 1


def _get_followers(user_id: str) -> int:
    # Priority order: Unicode → 13-char → _prime → standard
    if _is_unicode_user_id(user_id):
        return UNICODE_USER_FOLLOWERS
    if len(user_id) == 13:
        return FIBONACCI_13_FOLLOWERS
    sha_int = (int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % SHA_FOLLOWERS_MODULO) + SHA_FOLLOWERS_OFFSET
    if user_id.endswith('_prime'):
        return _nearest_prime(sha_int)
    return sha_int


def _is_meta_message(content: str) -> bool:
    return content.strip().lower() == 'teste técnico mbras'


def _score_message(content: str) -> float:
    """Compute the sentiment score for a message's content."""
    tokens = tokenize(content)
    if not tokens:
        return 0.0

    scores = []
    intensify_next = False
    negation_positions = []

    for i, token in enumerate(tokens):
        is_hashtag = token.startswith('#')

        if is_hashtag:
            # Hashtags: score 0, consume pending intensifier
            intensify_next = False
            scores.append(0.0)
            continue

        norm = normalize_for_matching(token)

        if norm in NEGATIONS:
            # Negation token: score 0, consume pending intensifier, record position
            intensify_next = False
            negation_positions.append(i)
            scores.append(0.0)
            continue

        if norm in INTENSIFIERS:
            # Intensifier: score 0, mark next token for ×1.5
            intensify_next = True
            scores.append(0.0)
            continue

        # Regular token: look up in lexicon
        base_score = LEXICON.get(norm, 0.0)

        # Apply intensifier rule
        if intensify_next:
            base_score *= 1.5
            intensify_next = False

        # Apply negation parity rule
        # Negation at position p covers tokens p+1, p+2, p+3
        # Bounded reverse scan: O(1) amortized since at most 3 positions in range
        active_neg_count = 0
        for p in reversed(negation_positions):
            if p >= i:
                continue
            if i - p > 3:
                break
            active_neg_count += 1
        if active_neg_count % 2 == 1:  # odd → invert
            base_score = -base_score
        # even (including 0) → no inversion (cancel effect)

        # MBRAS rule: if positive after negation, multiply by 2
        if base_score > 0:
            base_score *= 2

        scores.append(base_score)

    return sum(scores) / len(tokens)


def _classify_score(score: float) -> str:
    if score > 0.1:
        return 'positive'
    elif score < -0.1:
        return 'negative'
    return 'neutral'


def _compute_sentiment_distribution(sentiments: list) -> dict:
    counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    total = 0
    for s in sentiments:
        if s != 'meta':
            counts[s] += 1
            total += 1
    if total == 0:
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
    return {k: v / total * 100 for k, v in counts.items()}


def _compute_influence(messages: list) -> list:
    user_data = {}
    for msg in messages:
        uid = msg['user_id']
        if uid not in user_data:
            user_data[uid] = {'reactions': 0, 'shares': 0, 'views': 0}
        user_data[uid]['reactions'] += msg.get('reactions', 0)
        user_data[uid]['shares'] += msg.get('shares', 0)
        user_data[uid]['views'] += msg.get('views', 0)

    ranking = []
    for uid, data in user_data.items():
        followers = _get_followers(uid)
        reactions = data['reactions']
        shares = data['shares']
        views = data['views']

        engagement_rate = (reactions + shares) / max(views, 1)

        # Golden ratio adjustment: if (reactions+shares) is multiple of 7 and > 0
        interactions = reactions + shares
        if interactions > 0 and interactions % 7 == 0:
            engagement_rate *= (1 + 1 / PHI)

        score = followers * 0.4 + engagement_rate * 0.6

        # Post-adjustments
        if uid.endswith('007'):
            score *= 0.5
        if 'mbras' in uid.lower():
            score += 2.0

        ranking.append({'user_id': uid, 'influence_score': score})

    ranking.sort(key=lambda x: x['influence_score'], reverse=True)
    return ranking


def _compute_trending(messages: list, sentiments: list, timestamps: list, now_utc: datetime) -> list:
    hashtag_weights = {}
    hashtag_counts = {}

    for msg, sentiment, ts in zip(messages, sentiments, timestamps):
        minutes_since_post = (now_utc - ts).total_seconds() / 60
        time_weight = 1 + (1 / max(minutes_since_post, 0.01))

        if sentiment == 'positive':
            sentiment_mod = 1.2
        elif sentiment == 'negative':
            sentiment_mod = 0.8
        else:
            sentiment_mod = 1.0  # neutral or meta

        for ht in msg.get('hashtags', []):
            ht_lower = ht.lower()
            ht_len = len(ht_lower)

            weight = time_weight * sentiment_mod
            if ht_len > HASHTAG_LOG_LENGTH_THRESHOLD:
                weight *= math.log10(ht_len) / math.log10(HASHTAG_LOG_LENGTH_THRESHOLD)

            hashtag_weights[ht_lower] = hashtag_weights.get(ht_lower, 0.0) + weight
            hashtag_counts[ht_lower] = hashtag_counts.get(ht_lower, 0) + 1

    sorted_hashtags = sorted(
        hashtag_weights.keys(),
        key=lambda h: (-hashtag_weights[h], -hashtag_counts[h], h)
    )
    return sorted_hashtags[:TRENDING_TOP_N]


def _parse_timestamp(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))


def _compute_anomaly(messages: list, sentiments: list, timestamps: list) -> tuple:
    if not messages:
        return False, None

    # Burst detection: > 10 messages from same user_id within 5-minute window
    user_times = defaultdict(list)
    for i, msg in enumerate(messages):
        user_times[msg['user_id']].append(timestamps[i])

    burst_window_delta = timedelta(minutes=BURST_WINDOW_MINUTES)
    for uid, times in user_times.items():
        times_sorted = sorted(times)
        right = 0
        for left in range(len(times_sorted)):
            while right < len(times_sorted) and times_sorted[right] <= times_sorted[left] + burst_window_delta:
                right += 1
            if right - left > BURST_THRESHOLD:
                return True, 'burst'

    # Alternating sentiment detection: >= 10 consecutive alternating messages
    ts_sentiment = sorted(
        [(timestamps[i], sentiments[i]) for i in range(len(messages))],
        key=lambda x: x[0]
    )
    sent_seq = [s for _, s in ts_sentiment if s in ('positive', 'negative')]
    if len(sent_seq) >= ALTERNATING_THRESHOLD:
        for start in range(len(sent_seq) - (ALTERNATING_THRESHOLD - 1)):
            window = sent_seq[start:start + ALTERNATING_THRESHOLD]
            if window[0] != window[1]:  # must alternate
                alternating = all(window[k] != window[k - 1] for k in range(1, len(window)))
                if alternating:
                    return True, 'alternating'

    # Synchronized posting: >= 3 messages within ±2 seconds of each other
    if len(messages) >= SYNCHRONIZED_MIN_MESSAGES:
        times_sorted = sorted(timestamps)
        sync_window_delta = timedelta(seconds=SYNCHRONIZED_WINDOW_SECONDS)
        right = 0
        for left in range(len(times_sorted)):
            while right < len(times_sorted) and times_sorted[right] <= times_sorted[left] + sync_window_delta:
                right += 1
            if right - left >= SYNCHRONIZED_MIN_MESSAGES:
                return True, 'synchronized'

    return False, None


def _compute_flags(messages: list) -> dict:
    mbras_employee = False
    candidate_awareness = False
    special_pattern = False

    for msg in messages:
        uid = msg['user_id']
        content = msg['content']

        if 'mbras' in uid.lower():
            mbras_employee = True

        content_lower = content.lower()
        if 'teste técnico mbras' in content_lower:
            candidate_awareness = True

        if len(content) == SPECIAL_PATTERN_LENGTH and 'mbras' in content_lower:
            special_pattern = True

    return {
        'mbras_employee': mbras_employee,
        'candidate_awareness': candidate_awareness,
        'special_pattern': special_pattern,
    }


def analyze_feed(messages: list, time_window_minutes: int) -> dict:
    start = _time.perf_counter()
    now_utc = datetime.now(timezone.utc)

    # Time window filtering — parse each timestamp exactly once
    cutoff = now_utc - timedelta(minutes=time_window_minutes)
    future_limit = now_utc + timedelta(seconds=FUTURE_TOLERANCE_SECONDS)

    parsed_pairs = [(_parse_timestamp(msg['timestamp']), msg) for msg in messages]
    filtered_pairs = [(ts, msg) for ts, msg in parsed_pairs if cutoff <= ts <= future_limit]

    # Fallback: if all messages filtered out, process all messages
    if not filtered_pairs and messages:
        filtered_pairs = parsed_pairs

    filtered = [msg for _, msg in filtered_pairs]
    filtered_timestamps = [ts for ts, _ in filtered_pairs]

    # Compute per-message sentiments
    sentiments = []
    for msg in filtered:
        content = msg['content']
        if _is_meta_message(content):
            sentiments.append('meta')
        else:
            score = _score_message(content)
            sentiments.append(_classify_score(score))

    # Sentiment distribution (excludes meta)
    sentiment_distribution = _compute_sentiment_distribution(sentiments)

    # Special flags (OR logic across all messages)
    flags = _compute_flags(filtered)

    # Engagement score
    if flags['candidate_awareness']:
        engagement_score = CANDIDATE_ENGAGEMENT_SCORE
    else:
        total_engagement = 0.0
        for msg in filtered:
            reactions = msg.get('reactions', 0)
            shares = msg.get('shares', 0)
            views = msg.get('views', 0)
            rate = (reactions + shares) / max(views, 1)
            interactions = reactions + shares
            if interactions > 0 and interactions % 7 == 0:
                rate *= (1 + 1 / PHI)
            total_engagement += rate
        engagement_score = total_engagement / len(filtered) if filtered else 0.0

    # Influence ranking
    influence_ranking = _compute_influence(filtered)

    # Trending topics
    trending_topics = _compute_trending(filtered, sentiments, filtered_timestamps, now_utc)

    # Anomaly detection
    anomaly_detected, anomaly_type = _compute_anomaly(filtered, sentiments, filtered_timestamps)

    elapsed_ms = int((_time.perf_counter() - start) * 1000)

    return {
        'sentiment_distribution': sentiment_distribution,
        'engagement_score': engagement_score,
        'trending_topics': trending_topics,
        'influence_ranking': influence_ranking,
        'anomaly_detected': anomaly_detected,
        'anomaly_type': anomaly_type,
        'flags': flags,
        'processing_time_ms': elapsed_ms,
    }
