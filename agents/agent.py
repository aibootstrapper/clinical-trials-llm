import json

import pandas as pd
from pydantic import BaseModel, BaseConfig
from typing import List, Optional, Mapping
from agents.llm import Chat
from agents.tools.base import ToolInterface
from agents.tools.clinical_trials import (
    AskForCondition,
    NormalResponse,
    ConditionExtractor,
    AskNextQuestion,
)


PROMPT_TEMPLATE = """You can use tools and memory to decide what is the most appropiate action to take. Decide what tool you want to use, and then use it by typing the name of the tool. You can use the following tools: 

{tool_descriptions}

Here's your current memory: {memory}

Your goal is to narrow down the list of clinical trials that are relevant to the user. 

Has the user told you their condition? {condition_set}

Use the following format for your response:

{{
    "action": "$TOOL_NAME",
    "input": "$RAW_USER_INPUT"
}}

where `action` is the action to take, exactly one element of [{tool_names}] and `input` is the input to the tool.

Begin!

User input: {input}
"""


class Agent(BaseModel):
    class Config(BaseConfig):
        arbitrary_types_allowed = True

    llm: Chat
    tools: List[ToolInterface]
    prompt_template: str = PROMPT_TEMPLATE
    condition: Optional[str] = None
    df_clinical_trials: Optional[pd.DataFrame] = pd.read_csv(
        "./data/refined_trials.csv"
    )
    df_cfg_parsed_trials: Optional[pd.DataFrame] = pd.read_csv(
        "./data/cfg_parsed_clinical_trials.tsv", sep="\t"
    )
    df_available_trials: Optional[pd.DataFrame] = None
    df_cfg_parsed_available_trials: Optional[pd.DataFrame] = None
    previous_questions: List[str] = []
    memory: List[Mapping] = []

    @property
    def tool_descriptions(self):
        return "\n".join(
            [
                f"{tool.name}: {tool.description}"
                for tool in self.tools
                if tool.name != "AskForCondition"
            ]
        )

    @property
    def tool_names(self):
        return ", ".join(
            [tool.name for tool in self.tools if tool.name != "AskForCondition"]
        )

    @property
    def tool_by_names(self):
        return {tool.name: tool for tool in self.tools}

    def run_input(self, input: str) -> str:
        self.memory.append({"responder": "Human", "message": input})
        if self.condition is None:
            return f'{{"action": "AskForCondition", "input": "{input}"}}'

        else:
            prompt = self.prompt_template.format(
                tool_descriptions=self.tool_descriptions,
                tool_names=self.tool_names,
                input=input,
                condition_set=self.condition is not None
                and self.condition != "pending",
                memory=self.memory,
            )
            response = self.llm(prompt)
            return response

    def run_action(self, action: str, action_input: str) -> str:
        tool = self.tool_by_names[action]
        tool_result = tool.use(action_input, self)
        self.memory.append({"responder": tool.name, "message": tool_result})

        return tool_result

    def run(self, input: str) -> str:
        action = self.run_input(input)
        action_obj = json.loads(action)
        print(action_obj)
        response = self.run_action(action_obj["action"], action_obj["input"])

        return response

    def add_tool(self, tool: ToolInterface):
        self.tools.append(tool)

    def reset(self):
        self.condition = None
        self.previous_questions = []
        self.memory = []
        self.df_available_trials = self.df_clinical_trials.copy()
        self.df_cfg_parsed_available_trials = self.df_cfg_parsed_trials.copy()


if __name__ == "__main__":
    agent = Agent(
        llm=Chat(),
        tools=[
            AskForCondition(),
            NormalResponse(),
            ConditionExtractor(),
            AskNextQuestion(),
        ],
    )

    while True:
        user_input = input("User input: ")
        action = agent.run(user_input)
        action_obj = json.loads(action)
        print(action_obj)
        output = agent.run_action(action_obj["action"], action_obj["input"])
        print(output)
