import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from tqdm import tqdm
import numpy as np
import re
from bert_score import score
from args_parser import get_args


def calculate_bleu_scores(file_sentences, df_sentences):
    bleu_scores = []
    for ref, sent in tqdm(zip(df_sentences, file_sentences)):
        try:
            bleu_scores.append(sentence_bleu(
                ref.lower().split(), sent, 
                smoothing_function=SmoothingFunction().method4
            ))
        except:
            print("problem")

    bleu_score = np.array(bleu_scores).mean()
    return bleu_score


def calculate_bert_scores(file_sentences, df_sentences):
    _, _, bert_scores = score(
        file_sentences, df_sentences, lang="en", verbose=True
    )

    avg_bert_score = bert_scores.mean().item()
    return avg_bert_score


def remove_teacher(text):
    return text[10:]


def load_sentences_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    sentences = [line.strip().split(': ')[1] for line in lines]
    return sentences


def read_file_to_dataframe(file_path):
    data = []
    current_index = None
    current_utterance = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Check if the line starts with an index pattern (number followed by a colon)
            if re.match(r'^\d+:', line):
                # If there's an existing utterance, save it before starting a new one
                if current_index is not None:
                    data.append([current_index, ''.join(current_utterance).strip()])
                # Start a new utterance
                current_index, rest_of_line = line.split(':', 1)
                current_utterance = [rest_of_line]
            else:
                # Continue accumulating lines for the current utterance
                current_utterance.append(line)
    
    # Add the last utterance to the data if it exists
    if current_index is not None:
        data.append([current_index, ''.join(current_utterance).strip()])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['Index', 'Utterance'])
    return df


if __name__ == "__main__":
    args = get_args()

    test_data = pd.read_csv(args.teacher_answers)
    test_data = list(test_data["text"].apply(remove_teacher))
    
    df = read_file_to_dataframe(args.generated_sentences)
    generated_sentences = list(df["Utterance"])


    avg_bleu_score = calculate_bleu_scores(generated_sentences, test_data)
    avg_bert_score = calculate_bert_scores(generated_sentences, test_data)


    print(f"Average BLEU score: {avg_bleu_score}")
    print(f"Average BERT score: {avg_bert_score}")