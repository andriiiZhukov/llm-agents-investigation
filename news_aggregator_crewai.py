import requests
from bs4 import BeautifulSoup
from langchain_ollama import ChatOllama
from crewai import Agent, Task, Crew

llm = ChatOllama(model="mistral")

BREAK_LINES = "\n-------------------\n"

def fetch_news():
    print("\n 1. Fetching news...\n")

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



fetch_agent = Agent(name="Parser", function=fetch_news, description="Parses news")
analyze_agent = Agent(name="Analyst", function=analyze_news, description="Rates the news")
summarize_agent = Agent(name="Editor", function=summarize_news, description="Summarizes the news")
translate_agent = Agent(name="Translator", function=translate_news, description="Translates news")

fetch_task = Task(agent=fetch_agent, description="Collect the latest news")
analyze_task = Task(agent=analyze_agent, description="Analyze the importance of news", dependencies=[fetch_task])
summarize_task = Task(agent=summarize_agent, description="Summarize important news", dependencies=[analyze_task])
translate_task = Task(agent=translate_agent, description="Translate summary news", dependencies=[summarize_task])

news_crew = Crew(tasks=[fetch_task, analyze_task, summarize_task, translate_task])
news_crew.kickoff()

print("\n Final response with translation:")
print(news_crew.results)
