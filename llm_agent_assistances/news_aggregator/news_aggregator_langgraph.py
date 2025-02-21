import sys
import requests
from bs4 import BeautifulSoup
from langchain_ollama import ChatOllama
from langgraph.graph import Graph

llm = ChatOllama(model="mistral")

BREAK_LINES = "\n-------------------\n"

def fetch_news(_):
    print("\n 1. Fetching news...\n")

    is_skip_other_steps = True if len(sys.argv) > 1 else False
    url = "https://news.ycombinator.com/"
    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    headlines = []
    for item in soup.find_all("tr", class_="athing"):
        title_tag = item.find("span", class_="titleline").find("a")
        if title_tag:
            headlines.append(title_tag.text)
    
    news = headlines[:5] if headlines else []
    
    print("News:")
    print(news)
    print(BREAK_LINES)

    if is_skip_other_steps:
        return {"skip_analysis": []}

    return {"news": news}

def analyze_news(inputs):
    print("\n 2. Analyzing news...\n")

    news = "\n".join(inputs["news"])
    prompt = f"Which of these news items is the most important? {news}"
    response = llm.invoke(prompt)
    important_news = response.content

    print("Important news:")
    print(important_news)
    print(BREAK_LINES)

    return {"important_news": important_news}

def summarize_news(inputs):
    print("\n 3. Summarizing news...\n")

    important_news = inputs["important_news"]
    prompt = f"Make a short summary of the news: {important_news}"
    response = llm.invoke(prompt)
    summary = response.content

    print("\n Summary:")
    print(summary)
    print(BREAK_LINES)
  
    return {"summary": summary}

def translate_news(inputs):
    print("\n 4. Translating news...\n")

    summary = inputs["summary"]
    prompt = f"Translate this into Ukrainian: {summary}"
    response = llm.invoke(prompt)
    translated_summary = response.content

    return {"translated_summary": translated_summary}

def handle_no_news(inputs):
    print("\n ⚠️ Skipping analysis and other steps.\n")
    return {"important_news": "No news available"}

def router(inputs):
    news = inputs.get("news", [])
    return "analyze" if news else "no_news"


workflow = Graph()

workflow.add_node("fetch", fetch_news)
workflow.add_node("analyze", analyze_news)
workflow.add_node("summarize", summarize_news)
workflow.add_node("translate", translate_news)
workflow.add_node("no_news", handle_no_news)

workflow.add_conditional_edges("fetch", router, {"analyze", "no_news"})
workflow.add_edge("analyze", "summarize")
workflow.add_edge("summarize", "translate")

workflow.set_entry_point("fetch")
workflow.set_finish_point("translate")

news_aggregator = workflow.compile()

result = news_aggregator.invoke({})

if result and result["translated_summary"]:
    print("\n Final response with translation:")
    print(result["translated_summary"])