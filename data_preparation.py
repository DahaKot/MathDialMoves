import pandas as pd
from tqdm import tqdm

from args_parser import get_args

import random

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # If using CUDA
    random.seed(seed_value)
    # Ensures that CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1)  # You can choose any seed value

strategies = {"probing": 0, "generic": 0, "focus": 0, "telling": 0}
# the prompt is definetely smth that should be experimented with
PROMPT = """
This is the correct solution of a problem:
{solution}
The following is a fragment of a conversation between a student and a teacher discussing the student solution:
{conversation}
"""

def cut_conversation(data):
    conversations = data["conversation"]

    data['strategies'] = ""
    data['strategies'] = data['strategies'].astype('object')

    # each line is one dialogue
    for i, line in tqdm(enumerate(conversations)):
        variable_line = line
        min_index = 0

        current_strategies = []
        while variable_line.find("(", min_index) > 0:
            # find text in brackets
            # this is the way mathdial denotes strategies we need to extract
            a, b = variable_line.find(
                "(", min_index) + 1, variable_line.find(")", min_index
            )
            strategy = variable_line[a:b]

            # if brackets are used not as a strategy marker, go further
            if strategy not in strategies:
                min_index = b + 1
                continue

            # else we write down which strategies are used in a list    
            strategies[strategy] += 1
            current_strategies.append(strategy)

            # cut out the strategy name from the dialogue 
            variable_line = variable_line[:a - 1] + variable_line[b + 1:]

        # now for each dialogue we have a version without strategies and a
        # separate array of strategies used
        data.at[i, "cut_conversation"] = variable_line
        data.at[i, "strategies"] = current_strategies

    return data, strategies


def generate_data(data, window_size):
    # this is just for statistics
    length_in_replics = []
    length_in_teacher_replics = []

    index = 0
    header = ["index", "text", "label"]
    new_data = []

    for _, problem in tqdm(data.iterrows()):
        # question = problem["question"]
        ground_truth_solution = problem["ground_truth"]
        # incorrect_solution = problem["student_incorrect_solution"]

        cut_replics = problem["cut_conversation"].split("|EOM|")
        replics = problem["conversation"].split("|EOM|")
        length_in_replics.append(len(cut_replics))

        teacher_replics_count = 0
        for j, cut_replic in enumerate(cut_replics):
            if cut_replic[:8] == "Teacher:" and j > 0:
                teacher_replics_count += 1
                a, b = replics[j].find("(") + 1, replics[j].find(")")
                if a >= 0 and b >= 0:
                    label = replics[j][a:b]
                    if label not in strategies:
                        print("ALARM: ", label)
                        print(replics[j])

                    start = max(0, j - window_size) #for shorter windows
                    text = PROMPT.format(solution=ground_truth_solution, conversation="\n".join(cut_replics[start:j]))
                    
                    new_data.append([index, text, label])
                    index += 1

        length_in_teacher_replics.append(teacher_replics_count)

    return pd.DataFrame(new_data, columns=header), length_in_replics, length_in_teacher_replics


if __name__ == "__main__":
    args = get_args()

    data = pd.read_csv(args.data_path)
    data, strategies_stats = cut_conversation(data)

    new_data, length_in_replics, length_in_teacher_replics = generate_data(
        data, args.window_size
    )

    new_data.to_csv(args.save_file, index=False)