import gensim.downloader
import string
from typing import List, Tuple, Dict, Set
from nltk.corpus import stopwords

stop_words: Set[str] = set(stopwords.words('english'))

# Load the pre-trained Word2Vec model
print("Loading Word2Vec model (word2vec-google-news-300)...")
try:
    m = gensim.downloader.load('word2vec-google-news-300')
except Exception as e:
    print(f"Error loading Word2Vec model: {e}")
    m = None


sense = "play"
gloss1 = "a dramatic work intended for performance by actors on a stage"
gloss2 = "deliberate coordinated movement requiring dexterity and skill"

CONTEXT_SIZE = 5

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

def clean_and_tokenize(text: str, remove_stop_words: bool = False) -> List[str]:
    """
    Cleans text by removing punctuation, converting to lowercase, and optionally
    removing stop words.
    """
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.lower().split()

    if remove_stop_words:
        tokens = [token for token in tokens if token not in stop_words]

    # Return only non-empty tokens
    return [token for token in tokens if token]


def calculate_similarity_sum(model, context: List[str], gloss_words: List[str]) -> Tuple[float, int, int]:
    """
    Computes the double summation of cosine similarities between context words (W)
    and gloss content words (L), restricted to words present in the Word2Vec model.

    Returns: (Total sum, length of valid context W, length of valid gloss L).
    """
    if model is None:
        return 0.0, 0, 0

    # Filter words to ensure they are present in the Word2Vec vocabulary
    valid_context = [w for w in context if w in model.key_to_index]
    valid_gloss = [x for x in gloss_words if x in model.key_to_index]

    if not valid_context or not valid_gloss:
        return 0.0, 0, 0

    total_sim_sum = 0.0

    # Calculate: sum(w_i in W) sum(x in L) cosine_similarity(w_i, x)
    for w_i in valid_context:
        for x in valid_gloss:
            total_sim_sum += model.similarity(w_i, x)

    return total_sim_sum, len(valid_context), len(valid_gloss)


def calculate_sense_similarity(model, sentences: List[str], sense: str, gloss1: str, gloss2: str) -> Dict[str, str]:
    """
    Classifies sentences using the Word2Vec WSD algorithm.

    The sense with the higher normalized similarity score (based on the average
    cosine similarity between context words and gloss words) is selected.
    """
    if model is None:
        return {}

    classification: Dict[str, str] = {}
    sense_lower = sense.lower()

    # Pre-process glosses once
    gloss1_tokens = clean_and_tokenize(gloss1, remove_stop_words=True)
    gloss2_tokens = clean_and_tokenize(gloss2, remove_stop_words=True)

    print(f"\nGloss 1 (L1) content words: {gloss1_tokens}")
    print(f"Gloss 2 (L2) content words: {gloss2_tokens}\n")

    for sen in sentences:
        full_tokens = clean_and_tokenize(sen)

        try:
            ambiguous_index = full_tokens.index(sense_lower)
        except ValueError:
            classification[sen] = "N/A (Word Missing)"
            continue

        # Extract context window of size 2*CONTEXT_SIZE, excluding the ambiguous word
        start_index = max(0, ambiguous_index - CONTEXT_SIZE)
        end_index = min(len(full_tokens), ambiguous_index + CONTEXT_SIZE + 1)

        context_tokens = full_tokens[start_index:ambiguous_index] + full_tokens[ambiguous_index + 1:end_index]

        # Calculate similarity scores (double sum and sizes for normalization)
        sum1, len_W1, len_L1 = calculate_similarity_sum(model, context_tokens, gloss1_tokens)
        sum2, len_W2, len_L2 = calculate_similarity_sum(model, context_tokens, gloss2_tokens)

        # Normalize the sum: Sim = Sum / (|W| * |L|)
        sim_L1 = sum1 / (len_W1 * len_L1) if len_W1 > 0 and len_L1 > 0 else 0.0
        sim_L2 = sum2 / (len_W2 * len_L2) if len_W2 > 0 and len_L2 > 0 else 0.0

        if sim_L1 > sim_L2:
            classification[sen] = f"g1 (Score: {sim_L1:.4f})"
        elif sim_L2 > sim_L1:
            classification[sen] = f"g2 (Score: {sim_L2:.4f})"
        else:
            classification[sen] = "Tie (0.0000)"

    return classification


if __name__ == '__main__':
    if m is not None:
        word2vec_class = calculate_sense_similarity(m, sentences, sense, gloss1, gloss2)

        correct_count = 0
        total_sentences = len(sentences)

        print("--- Word2Vec WSD Classification Results ---")
        for sentence, prediction_str in word2vec_class.items():
            predicted_sense = prediction_str.split()[0]
            expected_sense = correct_classification.get(sentence)
            is_correct = predicted_sense == expected_sense
            status = "CORRECT" if is_correct else f"INCORRECT (Expected: {expected_sense})"

            print(f"[Predicted: {prediction_str} | {status}]")

            if is_correct:
                correct_count += 1

        print("\n--- Summary ---")
        print(f"The Word2Vec WSD algorithm correctly classified {correct_count} out of {total_sentences} sentences.")
        accuracy = (correct_count / total_sentences) * 100 if total_sentences > 0 else 0
        print(f"Accuracy: {accuracy:.2f}%")