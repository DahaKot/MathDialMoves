import argparse

def get_args():
    parser = argparse.ArgumentParser()
    add_args(parser)

    args, _ = parser.parse_known_args()

    return args

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--data_path", type=str, default="./mathdial/data/train.csv"
    )
    parser.add_argument(
        "--save_file", type=str, default="./data/train.csv"
    )
    parser.add_argument(
        "--window_size", type=int, default=3
    )
    parser.add_argument(
        "--train_path", type=str, default="./data/train_window_1.csv"
    )
    parser.add_argument(
        "--test_path", type=str, default="./data/test_window_1.csv"
    )
    parser.add_argument(
        "--teacher_answers", type=str, default="./data/test_replics.csv"
    )
    parser.add_argument(
        "--generated_sentences", type=str, default="predicted_answers_focus_label.txt"
    )
    parser.add_argument(
        "--logging_dir", type=str, default="./logs/default_logs"
    )
    parser.add_argument(
        "--run_name", type=str, default="default_run"
    )
    parser.add_argument(
        "--model_name", type=str, default="roberta-base"
    )
