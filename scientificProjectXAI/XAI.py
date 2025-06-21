#%%
# Use a pipeline as a high-level helper
from transformers import pipeline

#device = 0 if torch.cuda.is_available() else -1

#print(device)

classifier = pipeline("text-classification", model="Hate-speech-CNERG/bert-base-uncased-hatexplain", device=0)
#, device=device
#original_text = "and this is why I end up with nigger trainee doctors who can not speak properly lack basic knowledge"

#classifier(original_text, return_all_scores=True)
#%%
import json

json_file = './data/dataset.json'

def extract_sentences(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sentences = []
        for key, entry in data.items():
            
            if 'post_tokens' in entry:
                post_tokens = entry['post_tokens']
                sentence = " ".join(post_tokens)
                sentences.append(sentence)
            else:
                print(f"Warning: Entry {key} is missing 'post_tokens' key")

        return sentences

    except FileNotFoundError:
        print(f"Error: File {json_file} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: File {json_file} is not a valid JSON.")
        return []

sentences = extract_sentences(json_file)
#print(len(sentences))
#print(sentences[:5]) 
#%%
import shap
explainer = shap.Explainer(classifier)
shap_values = explainer(sentences[:100])
#%%
import shap
shap.plots.text(shap_values, display=True)
#print(shap_values.values[0])
#print(shap_values.data[0])
#print(shap_values[0])

def shap_to_natural_language(shap_values, top_n=3, class_index=1):
    explanations = []
    for i in range(len(shap_values)):
        sentence = shap_values.data[i]
        sentence_shap_values = shap_values.values[i]
        explanation = f"{i + 1}. Sentence: '{' '.join(sentence)}'\nThe model predicted this because:"
        # 获取特征和值的对
        #word_importance = list(zip(sentence, sentence_shap_values[:, class_index]))
        # 按照SHAP值排序，并选择前top_n个
        #word_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        #top_words = word_importance[:top_n]
        #for word, value in top_words:
        lists = list(zip(sentence, sentence_shap_values[:, class_index]))
        for word, value in lists:
            explanation += f"\n- The word '{word}' has a SHAP value of {value:.4f}"
        explanations.append(explanation)
    return explanations


explanations = shap_to_natural_language(shap_values, sentences)
for explanation in explanations:
    print(explanation)
    print("\n")
#%%
shap.plots.text(shap_values, display=True)