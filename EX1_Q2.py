from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Set, Dict

lemmatizer = WordNetLemmatizer()
stop_words: Set[str] = set(stopwords.words('english'))


def preprocess_text(text: str) -> List[str]:
    """
    Tokenizes, lemmatizes, and removes stopwords/short words from the text
    to extract content words for the LESK algorithm.
    """
    tokens = simple_preprocess(text, min_len=1)
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]
    # Filter for non-stopwords and words >= 3 characters
    content_words = [word for word in lemmas if word not in stop_words and len(word) >= 3]
    return content_words



sense = "play"
gloss1 = "a dramatic work intended for performance by actors on a stage"
gloss2 = "deliberate coordinated movement requiring dexterity and skill"

sen11 = "The critics praised the dramatic intensity of the lead actor's play at the festival last night"
sen12 = "Before the curtain rose for the evening's performance, the stage manager checked the lighting cues for the final act of the play"
sen13 = "Every actor hoped to be cast in the controversial new play written by the celebrated playwright"
sen14 = "The large, empty stage was set for the rehearsal of the experimental play that featured minimal scenery"
sen15 = "She found the performance of the new play so moving that she wept through the entire second half"

sen16 = "The grandmaster's every movement was deliberate as he prepared his next play in the end game"
sen17 = "The winning play required coordinated efforts from the entire defensive line, not just one exceptional player"
sen18 = "His incredible dexterity with the racquet allowed him to execute an untouchable drop play right on the sideline"
sen19 = "The young skater demonstrated remarkable skill in her final jump, securing a gold medal after a nearly flawless free play"
sen20 = "The coordinated fast-break play left the opposing team confused and unable to recover their defense"

sentences: List[str] = [sen11, sen12, sen13, sen14, sen15, sen16, sen17, sen18, sen19, sen20]

correct_classification: Dict[str, str] = {
    sen11: "g1", sen12: "g1", sen13: "g1", sen14: "g1", sen15: "g1",
    sen16: "g2", sen17: "g2", sen18: "g2", sen19: "g2", sen20: "g2"
}


def sim_LESK(ambiguous_word: str, g1: str, g2: str, sentences: List[str]) -> Dict[str, str]:
    """
    Implements the simplified LESK algorithm (WSD based on gloss-context overlap).

    Predicts the sense whose dictionary gloss has the largest overlap of content words
    with the ambiguous word's context window.
    """
    classifier: Dict[str, str] = {}
    context_window_size = 5

    # Pre-process glosses once to get sets of content words
    set_g1: Set[str] = set(preprocess_text(g1))
    set_g2: Set[str] = set(preprocess_text(g2))

    for sen in sentences:
        # Tokenize and normalize the sentence
        all_lemmas = [lemmatizer.lemmatize(word.lower().strip('.,!?"\'')) for word in simple_preprocess(sen, min_len=1)]

        try:
            sense_pos = all_lemmas.index(ambiguous_word)
        except ValueError:
            classifier[sen] = "N/A (Word Missing)"
            continue

        start_pos = max(0, sense_pos - context_window_size)
        end_pos = min(sense_pos + context_window_size + 1, len(all_lemmas))
        context_window = all_lemmas[start_pos:end_pos]

        # Remove the ambiguous word itself from the context window
        context_window.remove(ambiguous_word)

        # Extract content words from the context
        content_context: Set[str] = set([
            word for word in context_window
            if word not in stop_words and len(word) >= 3
        ])

        # Calculate overlap (intersection size)
        g1_overlap = len(content_context.intersection(set_g1))
        g2_overlap = len(content_context.intersection(set_g2))

        if g1_overlap < g2_overlap:
            classifier[sen] = "g2"
        elif g2_overlap < g1_overlap:
            classifier[sen] = "g1"
        else:
            classifier[sen] = "tie (g1/g2)"

    return classifier


if __name__ == '__main__':
    LESK_class = sim_LESK(sense, gloss1, gloss2, sentences)

    correct_count = 0
    for sen, predicted_class in LESK_class.items():
        expected_class = correct_classification.get(sen)
        if predicted_class == expected_class:
            correct_count += 1

    total_sentences = len(sentences)

    print(f"--- LESK Algorithm Results (n=5) ---")
    for sen, pred in LESK_class.items():
        expected = correct_classification.get(sen)
        status = "CORRECT" if pred == expected else f"INCORRECT (Expected: {expected})"
        print(f"Prediction: {pred} | {status} | Sentence: {sen[:50]}...")

    print(f"\nThe LESK algorithm correctly classified {correct_count} out of {total_sentences} sentences.")
    print(f"Accuracy: {correct_count / total_sentences:.2f}")