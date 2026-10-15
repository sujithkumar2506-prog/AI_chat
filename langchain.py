from google import genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
load_dotenv()

# llm = ChatGoogleGenerativeAI(
#     model = "gemini-3.8-flash",
#     temperature=0.7
# )

# prompt = "explain me about langchain in simple words?"
# response = llm.invoke(prompt)
# print(" Content\n", response.content)
# print(" Response metadata\n", response.response_metadata)
# print(" Response usage metadata", response.usage_metadata)
# print(" Response ID", response.id)

class modelSchema(BaseModel):
    content: str = Field(description='the main response content of the prompt')
    input_token: int = Field(description='The input token')
    output_token: int = Field(description= 'The output token')
    model: str = Field(description='The gemini AI model that generated the response')
llm = ChatGoogleGenerativeAI(
    model='gemini-3.8-flash',
    temperature=0.4
)
llm = llm.with_structured_output(modelSchema,include_raw=True)
result = response = llm.invoke('Provide me some key tips to self groom and bright face')
parsed = response['parsed']
print(parsed)
print('----------')
raw = result["raw"]
if raw is not None:
    print("Usage:", raw.usage_metadata)
    print("Response metadata:", raw.response_metadata)
    print("Input tokens:", raw.usage_metadata["input_tokens"])
    print("Output tokens:", raw.usage_metadata["output_tokens"])
    print("Total tokens:", raw.usage_metadata["total_tokens"])
    print("Model:", raw.response_metadata.get("model_name"))