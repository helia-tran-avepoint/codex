from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class Parameter(BaseModel):
    name: str
    type: str


class Variable(BaseModel):
    name: str
    type: str


class MethodInfo(BaseModel):
    name: str = ""
    returnType: str = ""
    description: str = ""
    sourceCode: str = ""
    inheritsFrom: str = ""
    className: str = ""
    namespaceName: str = ""
    parameters: List[Parameter] = Field(default_factory=list)
    variables: List[Variable] = Field(default_factory=list)
    comments: str = ""
    fileLocation: str = ""
    callees: List["MethodInfo"] = Field(default_factory=list)

    @model_validator(mode="before")
    def replace_none(cls, values):
        for field_name, field_value in cls.model_fields.items():
            if values.get(field_name) is None:
                values[field_name] = field_value.default
        return values


MethodInfo.model_rebuild()
