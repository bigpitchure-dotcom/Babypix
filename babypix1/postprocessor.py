"""
BabyPix1 Post-Processor - Cleans up model output by fixing fragmented tokens.

Pipeline:
  Stage 1: Input Analysis - Identify subject and context
  Stage 2: Draft & Self-Test - Scan for broken tokens
  Stage 3: Error Correction - Merge single-char spaces, fix punctuation
  Stage 4: Final Output - Clean assembled text
"""

import re
from typing import List, Optional


# ============================================================================
# Knowledge Base - Factual answers for specific questions
# ============================================================================

KNOWLEDGE_BASE = {
    "who are you": "I am BabyPix1, a language model created by Senthil Kumar CM, son of Mani and Soundhari Mani, under PixcelTree India (Pixceltree.ie). This AI model is dedicated in loving remembrance of his mother, Soundhari Mani.",
    "who created you": "I was created by Senthil Kumar CM, son of Mani and Soundhari Mani, under PixcelTree India (Pixceltree.ie). This AI model is dedicated in loving remembrance of his mother, Soundhari Mani.",
    "what is your name": "I am BabyPix1, a language model created by Senthil Kumar CM under PixcelTree India (Pixceltree.ie).",
    "who made you": "I was made by Senthil Kumar CM, son of Mani and Soundhari Mani, under PixcelTree India (Pixceltree.ie). Dedicated in loving remembrance of his mother, Soundhari Mani.",
    "who is your creator": "My creator is Senthil Kumar CM, son of Mani and Soundhari Mani, under PixcelTree India (Pixceltree.ie). This model is dedicated in loving memory of Soundhari Mani.",
    "who owns you": "I am owned by PixcelTree India (Pixceltree.ie), created by Senthil Kumar CM.",
    "what is pixceltree": "PixcelTree India (Pixceltree.ie) is the company that created me, BabyPix1. It was founded by Senthil Kumar CM.",
    "who is senthil": "Senthil Kumar CM is the creator of BabyPix1 and founder of PixcelTree India (Pixceltree.ie). He is the son of Mani and Soundhari Mani.",
}


# Common English words to help with merging broken tokens
COMMON_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must", "need", "dare",
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "and", "but", "or", "nor", "for", "yet", "so",
    "in", "on", "at", "to", "for", "with", "by", "from", "of", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "not", "no", "nor", "never", "always", "often", "sometimes",
    "very", "really", "quite", "just", "also", "too", "only",
    "said", "say", "says", "told", "tell", "asked", "ask", "think",
    "know", "see", "come", "go", "get", "make", "take", "give",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "man", "woman", "child", "children", "boy", "girl",
    "house", "door", "room", "water", "time", "day", "night",
    "good", "bad", "big", "small", "old", "new", "long", "short",
    "like", "want", "look", "feel", "found", "thing", "way",
    "don", "doesn", "didn", "wasn", "weren", "couldn", "wouldn", "shouldn",
    "isn", "aren", "haven", "hasn", "won", "ll", "ve", "re", "t",
    "s", "d", "m",
}

# Words that commonly appear after certain patterns
FOLLOWER_WORDS = {
    "the": ["man", "woman", "child", "house", "door", "water", "time", "day",
             "night", "boy", "girl", "room", "thing", "way", "place", "world"],
    "a": ["man", "woman", "child", "house", "door", "water", "time", "day",
           "night", "boy", "girl", "room", "thing", "way", "little", "good"],
    "in": ["the", "a", "his", "her", "my", "your", "our", "their", "that"],
    "of": ["the", "a", "his", "her", "my", "your", "our", "their", "that"],
    "to": ["the", "a", "his", "her", "my", "your", "our", "their", "be"],
    "and": ["the", "a", "he", "she", "it", "they", "we", "i", "you"],
    "he": ["was", "is", "had", "has", "did", "said", "went", "came"],
    "she": ["was", "is", "had", "has", "did", "said", "went", "came"],
    "it": ["was", "is", "had", "has", "did"],
    "i": ["am", "was", "had", "have", "did", "said", "went", "came"],
    "you": ["are", "were", "have", "had", "did", "said"],
    "we": ["are", "were", "have", "had", "did", "said"],
    "they": ["are", "were", "have", "had", "did", "said"],
}


def merge_broken_tokens(text: str) -> str:
    """
    Stage 3: Merge broken single-character tokens.
    e.g., 'c o w' -> 'cow', 'h o u s e' -> 'house'
    """
    words = text.split()
    if not words:
        return text

    merged = []
    buffer = []

    for word in words:
        if len(word) == 1 and word.isalpha() and word not in ('a', 'i'):
            buffer.append(word)
        else:
            if buffer:
                merged_word = "".join(buffer)
                if merged_word.lower() in COMMON_WORDS or len(merged_word) <= 2:
                    merged.append(merged_word)
                elif len(merged_word) > 1:
                    merged.append(merged_word)
                else:
                    merged.extend(buffer)
                buffer = []
            merged.append(word)

    if buffer:
        merged_word = "".join(buffer)
        if merged_word.lower() in COMMON_WORDS or len(merged_word) > 1:
            merged.append(merged_word)
        else:
            merged.extend(buffer)

    return " ".join(merged)


def fix_punctuation(text: str) -> str:
    """
    Stage 3: Fix spacing around punctuation.
    e.g., 'word .' -> 'word.', 'word ,' -> 'word,'
    """
    # Remove space before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    # Fix space after opening quote
    text = re.sub(r'"\s+', '" ', text)
    # Fix space before closing quote
    text = re.sub(r'\s+"', '"', text)
    # Remove double spaces
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def fix_repeating_tokens(text: str, max_repeat: int = 2) -> str:
    """
    Stage 3: Limit repeating token sequences.
    e.g., 'fool blame fool blame fool blame' -> 'fool blame fool blame'
    """
    words = text.split()
    if len(words) < 3:
        return text

    result = []
    i = 0
    while i < len(words):
        result.append(words[i])
        # Check for repeating 2-word patterns
        if i + 3 < len(words):
            pattern = [words[i], words[i + 1]]
            count = 0
            j = i
            while j + 1 < len(words) and words[j] == pattern[count % 2]:
                count += 1
                j += 1
                if count >= max_repeat * 2:
                    break
            if count >= 4:  # At least 2 full repetitions
                i += count - 1  # Skip repeated tokens
        i += 1

    return " ".join(result)


def apply_contextual_fixes(text: str) -> str:
    """
    Stage 3: Apply context-aware fixes using word patterns.
    """
    words = text.split()
    if not words:
        return text

    result = []
    for i, word in enumerate(words):
        # Fix common contractions
        lower = word.lower()
        if lower == "don t":
            result.append("don't")
        elif lower == "doesn t":
            result.append("doesn't")
        elif lower == "didn t":
            result.append("didn't")
        elif lower == "wasn t":
            result.append("wasn't")
        elif lower == "weren t":
            result.append("weren't")
        elif lower == "couldn t":
            result.append("couldn't")
        elif lower == "wouldn t":
            result.append("wouldn't")
        elif lower == "shouldn t":
            result.append("shouldn't")
        elif lower == "isn t":
            result.append("isn't")
        elif lower == "aren t":
            result.append("aren't")
        elif lower == "haven t":
            result.append("haven't")
        elif lower == "hasn t":
            result.append("hasn't")
        elif lower == "won t":
            result.append("won't")
        elif lower == "i ve":
            result.append("I've")
        elif lower == "i ll":
            result.append("I'll")
        elif lower == "i m":
            result.append("I'm")
        elif lower == "i d":
            result.append("I'd")
        elif lower == "you ve":
            result.append("you've")
        elif lower == "you ll":
            result.append("you'll")
        elif lower == "you re":
            result.append("you're")
        elif lower == "you d":
            result.append("you'd")
        elif lower == "he s":
            result.append("he's")
        elif lower == "he ll":
            result.append("he'll")
        elif lower == "he d":
            result.append("he'd")
        elif lower == "she s":
            result.append("she's")
        elif lower == "she ll":
            result.append("she'll")
        elif lower == "she d":
            result.append("she'd")
        elif lower == "it s":
            result.append("it's")
        elif lower == "it ll":
            result.append("it'll")
        elif lower == "we re":
            result.append("we're")
        elif lower == "we ll":
            result.append("we'll")
        elif lower == "we ve":
            result.append("we've")
        elif lower == "they re":
            result.append("they're")
        elif lower == "they ll":
            result.append("they'll")
        elif lower == "they ve":
            result.append("they've")
        elif lower == "that s":
            result.append("that's")
        elif lower == "what s":
            result.append("what's")
        elif lower == "there s":
            result.append("there's")
        elif lower == "let s":
            result.append("let's")
        elif lower == "here s":
            result.append("here's")
        elif lower == "how s":
            result.append("how's")
        elif lower == "who s":
            result.append("who's")
        # Fix capitalization at sentence start
        elif i == 0 and word and word[0].islower():
            result.append(word[0].upper() + word[1:])
        else:
            result.append(word)

    return " ".join(result)


def check_knowledge_base(text: str) -> Optional[str]:
    """
    Check if the input matches a known question in the knowledge base.
    
    Args:
        text: Input text to check
    
    Returns:
        Pre-defined answer if match found, None otherwise
    """
    text_lower = text.lower().strip()
    # Remove punctuation for matching
    text_clean = re.sub(r'[^\w\s]', '', text_lower)
    
    for pattern, answer in KNOWLEDGE_BASE.items():
        if pattern in text_clean or text_clean in pattern:
            return answer
        # Check for partial matches (at least 70% of words match)
        pattern_words = set(pattern.split())
        text_words = set(text_clean.split())
        if len(pattern_words) > 0:
            overlap = len(pattern_words & text_words) / len(pattern_words)
            if overlap >= 0.7:
                return answer
    
    return None


def postprocess(text: str, input_prompt: str = None) -> str:
    """
    Full 4-stage post-processing pipeline.
    
    Stage 0: Knowledge Base Check (match known questions)
    Stage 1: Input Analysis (pass-through for raw model output)
    Stage 2: Draft & Self-Test (scan for issues)
    Stage 3: Error Correction (fix broken tokens, punctuation, repeats)
    Stage 4: Final Output (clean assembled text)
    """
    if not text or not text.strip():
        return text

    # Stage 0: Check knowledge base if input prompt provided
    if input_prompt:
        kb_answer = check_knowledge_base(input_prompt)
        if kb_answer:
            return kb_answer

    # Stage 1 & 2: Initial text is already the draft
    result = text

    # Stage 3: Error Correction
    result = merge_broken_tokens(result)
    result = fix_punctuation(result)
    result = fix_repeating_tokens(result)
    result = apply_contextual_fixes(result)

    # Stage 4: Final Output
    # Ensure first letter is capitalized
    if result and result[0].islower():
        result = result[0].upper() + result[1:]

    return result
