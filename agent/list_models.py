from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import time

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_retries=0,
    timeout=120
)

print("Testing Gemini through LangChain...", flush=True)

start = time.time()

try:
    response = llm.invoke("Say hello.")

    print("Response:", response.content)
    print("Execution time:", round(time.time() - start, 2), "seconds")

except Exception as e:
    print("Error:", type(e).__name__)
    print(str(e))