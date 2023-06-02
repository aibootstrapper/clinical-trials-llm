import pandas as pd
import json
from loguru import logger


def filter_conditions(df, conditions):
    return df[df["conditions"].str.lower().str.contains(conditions)]


def get_next_question(df, df_cfg, past_questions=None):
    df_questions = df_cfg[
        df_cfg["#nct_id"].isin(df["#nct_id"])
        & ~df_cfg["question"].isin(past_questions if past_questions else [])
    ]
    return (
        df_questions.groupby("question")
        .count()
        .sort_values(by="#nct_id", ascending=False)
        .index[0]
    )


def filter_unmatched_trials(df, df_available_trails, id_col="#nct_id"):
    return df[df[id_col].isin(df_available_trails[id_col])]


def check_answer_against_numerical_criteria(answer: float, criteria: dict) -> bool:
    """Check if an answer meets a certain criteria."""
    answer = float(answer)
    lower_value = float(criteria.get("lower", {}).get("value", float("-inf")))
    lower_incl = criteria.get("lower", {}).get("incl", True)
    upper_value = float(criteria.get("upper", {}).get("value", float("inf")))
    upper_incl = criteria.get("upper", {}).get("incl", True)

    if lower_incl:
        if answer < lower_value:
            return False
    else:
        if answer <= lower_value:
            return False

    if upper_incl:
        if answer > upper_value:
            return False
    else:
        if answer >= upper_value:
            return False

    return True


def check_answer_against_ordinal_criteria(answer: str, criteria: dict) -> bool:
    """Check if an answer meets a certain criteria."""
    answer = str(answer)
    if answer in criteria.get("value", []):
        return True
    else:
        return False


def filter_on_question(df, question):
    return df[df["question"] == question]


def filter_on_answer(df, question, answer):
    condition_match_ids = set()
    for row in df.to_dict(orient="records"):
        if row["question"] != question:
            continue
        else:
            relation = json.loads(row["relation"])
            try:
                if row["variable_type"] == "numerical":
                    is_match = check_answer_against_numerical_criteria(answer, relation)
                elif row["variable_type"] == "ordinal":
                    is_match = check_answer_against_ordinal_criteria(answer, relation)
                else:
                    raise Exception(f'Unknown variable type {row["variable_type"]}')
            except Exception as e:
                logger.error(e)
                is_match = True  # Don't delete this line, it's a hack to get around the fact that some of the data is not in the right format

            if is_match:
                condition_match_ids.add(row["#nct_id"])

    return condition_match_ids


def main():
    # Load data
    df_trials = pd.read_csv("../data/refined_trials.csv")
    df_cfg_parsed_trials = pd.read_csv(
        "../data/cfg_parsed_clinical_trials.tsv", sep="\t"
    )

    patient_condition = input("What is your condition? ")

    df_available_trials = filter_conditions(df_trials, patient_condition)
    # Keep only the cfg for trials that are still a potential match
    df_cfg_parsed_available_trials = filter_unmatched_trials(
        df_cfg_parsed_trials, df_available_trials
    )

    questions_asked = []
    while len(df_available_trials) > 3:
        next_question = get_next_question(
            df_available_trials, df_cfg_parsed_trials, questions_asked
        )
        questions_asked.append(next_question)

        answer = input(f"{next_question} ")
        filtered_ids = filter_on_answer(
            df_cfg_parsed_available_trials, next_question, answer
        )
        df_available_trials = df_available_trials[
            df_available_trials["#nct_id"].isin(filtered_ids)
        ]
        df_cfg_parsed_available_trials = filter_unmatched_trials(
            df_cfg_parsed_available_trials, df_available_trials
        )
        logger.info(f"{len(df_available_trials)} trials still available")

    print(
        f"The following trials match your criteria: {list(df_available_trials['#nct_id'])}"
    )


if __name__ == "__main__":
    main()
