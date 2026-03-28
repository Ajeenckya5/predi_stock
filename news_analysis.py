"""
News Sentiment & Impact Analysis
================================
Analyzes news headlines for sentiment and impact on buy/sell decisions.
Uses VADER (lightweight, no ML) for sentiment scoring.
"""

from typing import List, Dict, Optional

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False


# Strong keyword sets for sentiment (fallback) and impact
_POSITIVE_KEYWORDS = [
    "surge", "rally", "gain", "rise", "profit", "beat", "growth", "bullish", "buy",
    "soar", "jump", "outperform", "record", "strong", "upgrade", "approval", "breakthrough",
    "deal", "acquisition", "partnership", "dividend", "buyback", "exceed", "upgrade",
]
_NEGATIVE_KEYWORDS = [
    "fall", "drop", "loss", "decline", "cut", "miss", "bearish", "sell", "crash",
    "plunge", "slump", "downgrade", "lawsuit", "investigation", "recall", "default",
    "bankruptcy", "layoff", "restructuring", "weak", "disappoint", "slump",
]
# High-impact keywords (earnings, FDA, M&A, etc.) boost impact score
_HIGH_IMPACT_KEYWORDS = [
    "earnings", "fda", "approval", "acquisition", "merger", "dividend", "buyback",
    "breakout", "upgrade", "downgrade", "lawsuit", "recall", "bankruptcy", "ipo",
    "guidance", "revenue", "profit", "loss", "beat", "miss", "ceo", "cfo",
]


def _fallback_sentiment(text: str) -> float:
    """Simple keyword-based fallback when VADER not installed."""
    if not text:
        return 0.0
    t = text.lower()
    pos = sum(1 for w in _POSITIVE_KEYWORDS if w in t)
    neg = sum(1 for w in _NEGATIVE_KEYWORDS if w in t)
    return max(-1.0, min(1.0, (pos - neg) * 0.25))


def analyze_news_sentiment(news_items: List[Dict]) -> Dict:
    """
    Analyze news items for overall sentiment and impact score.
    Returns:
        sentiment_score: -1 to 1 (negative to positive)
        impact: 0 to 1 (how much weight to give news in decision)
        summary: str
    """
    if not news_items:
        return {"sentiment_score": 0.0, "impact": 0.0, "summary": "No news", "details": []}

    analyzer = SentimentIntensityAnalyzer() if _VADER_AVAILABLE else None
    scores = []
    details = []

    for n in news_items:
        text = (n.get("title") or "") + " " + (n.get("description") or n.get("content") or "")
        text = text.strip()[:500]
        if not text:
            continue
        if analyzer:
            comp = analyzer.polarity_scores(text)
            s = comp["compound"]
        else:
            s = _fallback_sentiment(text)
        scores.append(s)
        details.append({"title": (n.get("title") or "")[:80], "sentiment": round(s, 2)})

    if not scores:
        return {"sentiment_score": 0.0, "impact": 0.0, "summary": "No analyzable news", "details": []}

    avg_sentiment = sum(scores) / len(scores)
    sentiment_score = max(-1.0, min(1.0, avg_sentiment))

    # Impact: articles with high-impact keywords get stronger weight
    n_news = len(scores)
    all_text = " ".join(
        (n.get("title") or "") + " " + (n.get("description") or n.get("content") or "")
        for n in news_items
    ).lower()
    impact_keyword_count = sum(1 for w in _HIGH_IMPACT_KEYWORDS if w in all_text)
    impact = min(1.0, 0.35 + n_news * 0.08 + (max(abs(s) for s in scores) * 0.25) + impact_keyword_count * 0.06)
    # Strong consensus increases impact
    std = (sum((s - avg_sentiment) ** 2 for s in scores) / len(scores)) ** 0.5
    if std < 0.2 and abs(avg_sentiment) > 0.3:
        impact = min(1.0, impact + 0.25)

    if sentiment_score > 0.3:
        summary = f"News bullish ({sentiment_score:.2f}), {n_news} articles"
    elif sentiment_score < -0.3:
        summary = f"News bearish ({sentiment_score:.2f}), {n_news} articles"
    else:
        summary = f"News neutral ({sentiment_score:.2f}), {n_news} articles"

    return {
        "sentiment_score": round(sentiment_score, 3),
        "impact": round(impact, 2),
        "summary": summary,
        "details": details,
        "n_articles": n_news,
    }
