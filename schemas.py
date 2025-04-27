from pydantic import BaseModel
from typing import Dict, List
from typing import Literal

class GrammarBody(BaseModel):
    prompt: str
    
class GFLBody(BaseModel):
    prompt: str 
    pronoun_map: dict[str, Literal['gender_fair', 'female', 'male']] = {
        "Alex": "gender_fair",
        "John": "male",
        "Jane": "female"
    }