from pydantic import BaseModel


class ToolInterface(BaseModel):
    name: str
    description: str

    def use(self, agent: "Agent", **kwargs):
        raise NotImplementedError("This method should be implemented by the subclass.")
