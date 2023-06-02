from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents.agent import Agent
from agents.llm import Chat
from agents.tools.clinical_trials import (
    AskForCondition,
    NormalResponse,
    ConditionExtractor,
    AskNextQuestion,
)
from trial_finder import (
    filter_conditions,
    filter_unmatched_trials,
    filter_on_answer,
    get_next_question,
)


class Conversation(BaseModel):
    # messages: List[Mapping[str, str]]
    message: str


class PatientCondition(BaseModel):
    condition: str


class PatientAnswer(BaseModel):
    question: str
    answer: str


app = FastAPI()
origins = [
    "http://localhost:5173",  # local dev
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

clinical_trials_agent = Agent(
    llm=Chat(),
    tools=[
        AskForCondition(),
        NormalResponse(),
        ConditionExtractor(),
        AskNextQuestion(),
    ],
)


@app.get("/get_trials")
def get_trials():
    return {"trials": list(clinical_trials_agent.df_available_trials["#nct_id"])}


@app.post("/reset")
def reset():
    clinical_trials_agent.reset()


@app.post("/")
def read_root(conversation: Conversation):
    response = clinical_trials_agent.run(conversation.message)
    print(response)
    return {"message": response}
