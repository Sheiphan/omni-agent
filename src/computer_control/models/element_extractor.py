"""Element extractor model."""
from pydantic import BaseModel, Field


class ElementExtractor(BaseModel):
    """Element extractor for the icon on the desktop."""
    binary_score: str = Field(
        description="Gives the name of the element on the screen of a Desktop."
    ) 