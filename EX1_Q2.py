from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
# nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = simple_preprocess(text, min_len=1)
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]
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

sentences = [sen11, sen12, sen13, sen14, sen15, sen16, sen17, sen18, sen19, sen20]

correct_classification = {sen11: "g1", sen12: "g1", sen13: "g1", sen14: "g1", sen15: "g1", sen16: "g2", sen17: "g2", sen18: "g2", sen19: "g2", sen20: "g2"}

def sim_LESK(sense, g1, g2, sentences):
    classifier = {}
    n = 5

    set_g1 = set(preprocess_text(g1))
    set_g2 = set(preprocess_text(g2))

    for sen in sentences:
        all_lemmas = [lemmatizer.lemmatize(word.lower().strip('.,!?"\'')) for word in simple_preprocess(sen, min_len=1)]
        sense_pos = all_lemmas.index(sense)
        context_window = all_lemmas[max(0, sense_pos - n): min(sense_pos + n + 1, len(all_lemmas))]
        context_window.remove(sense)
        content_context = set([word for word in context_window if word not in stop_words and len(word) >= 3])

        g1_overlap = len(content_context.intersection(set_g1))
        g2_overlap = len(content_context.intersection(set_g2))

        if g1_overlap < g2_overlap:
            classifier[sen] = "g2"
        elif g2_overlap < g1_overlap:
            classifier[sen] = "g1"
        else:
            classifier[sen] = "tie (g1/g2)"  # Handle ties

    return classifier



LESK_class = sim_LESK(sense, gloss1, gloss2, sentences)

correct = 0
for sen, clas in LESK_class.items():
    for sen_, clas_ in correct_classification.items():
        if sen == sen_:
            if clas == clas_:
                correct += 1

print("the LESK algorithm correctly classified {} sentences".format(correct))



