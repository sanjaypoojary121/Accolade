from pypdf import PdfReader
import ollama


reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()


question = input("Ask something: ")


prompt = f"""
Answer the question only using this content:

{text}

Question: {question}
"""


response = ollama.chat(
    model="mistral",
    messages=[
        {"role": "user", "content": prompt}
    ]
)



print("\nAnswer:\n")
print(response["message"]["content"])
