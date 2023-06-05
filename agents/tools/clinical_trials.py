from agents.tools.base import ToolInterface
from trial_finder import (
    filter_conditions,
    filter_unmatched_trials,
    get_next_question,
    filter_on_answer,
)


class AskForCondition(ToolInterface):
    name: str = "AskForCondition"
    description: str = "Always use this tool first. If the user has not told you their condition, use this tool to ask them for it."

    def use(self, input: str, agent: "Agent", **kwargs):
        return "What condition would you like to search clinical trials for?"


class NormalResponse(ToolInterface):
    name: str = "NormalResponse"
    description: str = "Only use this tool if the user has told you their condition and no other tool is appropiate. This tool will respond to the user's input."

    def use(self, input: str, agent: "Agent", **kwargs):
        return agent.llm(input)


class ConditionExtractor(ToolInterface):
    name: str = "ConditionExtractor"
    description: str = "Extracts the condition from the user's input. This tool should only be needed if the user has not given you their condition."

    def use(self, input: str, agent: "Agent", **kwargs):
        condition = agent.llm(
            f"Extract and derive the condition name to it's main cancer type. Usually only runs when the condition is set to `pendind`. The output should be the cancer name in lower case and nothing else: {input}"
        )
        print(condition)
        agent.condition = condition
        agent.df_available_trials = filter_conditions(
            agent.df_clinical_trials, agent.condition
        )
        # Keep only the cfg for trials that are still a potential match
        agent.df_cfg_parsed_available_trials = filter_unmatched_trials(
            agent.df_cfg_parsed_trials, agent.df_clinical_trials
        )
        next_question = get_next_question(
            agent.df_available_trials,
            agent.df_cfg_parsed_trials,
            agent.previous_questions,
        )
        agent.previous_questions.append(next_question)
        return agent.llm(
            f"Ask a user with {agent.condition} this question but make it clear what unit the response should be in: {next_question}. Ask for the raw value, without units"
        )


class AskNextQuestion(ToolInterface):
    name: str = "AskNextQuestion"
    description: str = "Used to follow up on a question that has already been asked, this should be used after the user answers a questions asked by the `ConditionExtractor` or `AskNextQuestion`. This tool should only be used if the user has already given you their condition."

    def use(self, input: str, agent: "Agent", **kwargs):
        filtered_ids = filter_on_answer(
            agent.df_cfg_parsed_available_trials, next_question, input
        )
        agent.df_available_trials = agent.df_available_trials[
            agent.df_available_trials["#nct_id"].isin(filtered_ids)
        ]
        agent.df_cfg_parsed_available_trials = filter_unmatched_trials(
            agent.df_cfg_parsed_trials, agent.df_clinical_trials
        )
        next_question = get_next_question(
            agent.df_available_trials,
            agent.df_cfg_parsed_trials,
            agent.previous_questions,
        )
        agent.previous_questions.append(next_question)

        if len(agent.df_available_trials) <= 10:
            return f"Here are the top {len(agent.df_available_trials)} trials that match your criteria:\n\n{list(agent.df_available_trials['#nct_id'])}"

        return agent.llm(
            f"Ask a user with {agent.condition} this question but make it clear what units the response should be in: {next_question}. Show a couple examples of what a good response would look like. Ask for the raw value, without units"
        )
