from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict

from dotenv import load_dotenv

import os 
load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")

model= ChatMistralAI(
    api_key=mistral_api_key,
    model_name="mistral-tiny",
    temperature=0.7,
    streaming=True,         
)

class Structure(TypedDict):
    summery:str
    sentiment:str

structured= model.with_structured_output(Structure)

result=structured.invoke("""Red Tape shoes turned out to be better than I expected honestly. I bought them mainly for daily office use and occasional outings, and after using them for a couple of weeks I can say they’re quite comfortable for long hours. The cushioning is soft enough for regular walking, and the shoes feel lightweight compared to some other brands in the same price range.

What I liked most was the design. They look stylish without being too flashy, so they go well with jeans, joggers, and even semi-casual outfits. The finishing and stitching also look decent, and the material doesn’t feel cheap. I’ve received a few compliments from friends as well, especially about how clean and premium they look.

One thing I noticed is that the fit was a little tight initially, especially near the toe area, but after wearing them for 3–4 days they became much more comfortable. The sole grip is okay for normal use, though I wouldn’t recommend them for heavy sports activities or running.

Overall, I feel they’re worth the money if you want comfortable and stylish everyday shoes without spending too much. Good balance of looks, comfort, and price.""")

print(result)