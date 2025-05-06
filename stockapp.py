import os
import dotenv
import yfinance as yf
import gradio as gr
from langchain_groq import ChatGroq
from langchain_core.tools import tool

# Load environment variables
dotenv.load_dotenv()
groq_api_key = os.environ.get("updated_stockapp")

# Initialize Groq model
llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192")

@tool
def company_information(ticker: str) -> dict:
    """Retrieve company info using yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker)
        ticker_info = ticker_obj.get_info()
        return ticker_info
    except Exception as e:
        return {"error": str(e)}

def answer_financial_question(ticker: str, question: str) -> str:
    company_info = company_information(ticker)
    if "error" in company_info:
        return f"Error retrieving data: {company_info['error']}"

    context = f"Here is the information I found about {ticker}:\n{company_info}\n"
    prompt = context + f"\nNow answer this question: {question}"

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Gradio UI
def gradio_interface(ticker, question):
    return answer_financial_question(ticker, question)

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Stock Ticker (e.g., AAPL)"),
        gr.Textbox(label="Your Financial Question")
    ],
    outputs=gr.Textbox(label="Finance Agent Answer"),
    title="Finance Agent",
    description="Ask financial questions about a specific stock ticker. The agent will retrieve the company information and provide an answer.")

if __name__ == "__main__":
    demo.launch()
