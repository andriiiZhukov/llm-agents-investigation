import requests
from bs4 import BeautifulSoup
from langchain_ollama import ChatOllama
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PyPDF2 import PdfWriter
from langgraph.graph import Graph

llm = ChatOllama(model="phi")

BREAK_LINES = "\n-------------------\n"

def research_agent(query):
    print("\n 1. Fetching sources...\n")

    url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for item in soup.find_all("h3", class_="gs_rt"):
        link = item.a["href"] if item.a else "No link"
        links.append(link)
    
    sources = links[:5]

    print("Found sources:")
    print(sources)
    print(BREAK_LINES)

    return {"sources": sources}

def analyze_sources(inputs):
    print("\n 2. Analyzing sources...\n")

    sources = "\n".join(inputs["sources"])
    response = llm.invoke(f"Analyze sources:\n{sources}")

    analysis = response.content

    print("Analysis:")
    print(analysis)
    print(BREAK_LINES)

    return {"analysis": analysis}

def fact_checker(inputs):
    print("\n 3. Checking facts...\n")

    analysis = inputs["analysis"]
    response = llm.invoke(f"Check the accuracy of the information:\n{analysis}")
    verified_data = response.content

    print("\n Verified data:")
    print(verified_data)
    print(BREAK_LINES)

    return {"verified_data": verified_data}

def report_generator(inputs):
    print("\n 4. Generating report...\n")

    verified_data = inputs["verified_data"]
    response = llm.invoke(f"Create a scientific report on the materials:\n{verified_data}")
    report = response.content

    print("\n Report:")
    print(report)
    print(BREAK_LINES)

    return {"report": report}

def generate_pdf(inputs):
    print("\n 5. Generating PDF...\n")

    pdf_filename = "report.pdf"
    report_text = inputs["report"]

    c = canvas.Canvas(pdf_filename, pagesize=letter)
    _, height = letter

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 50, "Report Summary:")

    text_obj = c.beginText(50, height - 70)
    text_obj.setFont("Helvetica", 10)

    max_width = 500
    lines = []
    for line in report_text.split("\n"):
        words = line.split()
        new_line = ""
        for word in words:
            if c.stringWidth(new_line + word, "Helvetica", 10) < max_width:
                new_line += word + " "
            else:
                lines.append(new_line.strip())
                new_line = word + " "
        lines.append(new_line.strip())

    for line in lines:
        text_obj.textLine(line)

    c.drawText(text_obj)
    c.save()


    # c = canvas.Canvas(pdf_filename, pagesize=letter)
    # _, height = letter

    # c.setFont("Helvetica", 12)
    # c.drawString(50, height - 50, "Report Summary:")
    # text_obj = c.beginText(50, height - 70)
    # text_obj.setFont("Helvetica", 10)

    # for line in report_text.split("\n"):
    #     text_obj.textLine(line)

    # c.drawText(text_obj)
    # c.save()

    # pdf_writer = PdfWriter()
    # with open(pdf_filename, "rb") as f:
    #     pdf_writer.add_page(pdf_writer.add_blank_page(width=612, height=792))

    # with open(pdf_filename, "wb") as f:
    #     pdf_writer.write(f)

    return {"pdf_report": f"{report_text} was written in the {pdf_filename}"}


workflow = Graph()

workflow.add_node("research", research_agent)
workflow.add_node("analyze", analyze_sources)
workflow.add_node("fact_check", fact_checker)
workflow.add_node("report", report_generator)
workflow.add_node("pdf", generate_pdf)

workflow.add_edge("research", "analyze")
workflow.add_edge("analyze", "fact_check")
workflow.add_edge("fact_check", "report")
workflow.add_edge("report", "pdf")

workflow.set_entry_point("research")
workflow.set_finish_point("pdf")

research_assistant = workflow.compile()

query = "Quantum computing"
result = research_assistant.invoke(query)

print("\n Final response:")
print(result["pdf_report"])


