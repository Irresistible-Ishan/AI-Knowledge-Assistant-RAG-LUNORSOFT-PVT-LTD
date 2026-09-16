#v2 code last one didnt work cuz somehow it was parsing pdf as an image ill write a seprate code for pdf using pypdf

import os
from PIL import Image
import pytesseract
from dotenv import load_dotenv
from llama_index.core import Document
# uhhh this is not working for pdf somehow
from llama_index.core import SimpleDirectoryReader
from pypdf import PdfReader

def parseImgOCR(imgpath) -> Document:
    text = pytesseract.image_to_string(Image.open(imgpath))
    return Document(
        text = text ,
        metadata = {
            "file_name" : os.path.basename(imgpath),
            "page_label" : "imgOCR",
            "file_type" : "image",
            "bypass_embedding": True
        })

def parsePDF(pdfpath: str) -> list[Document]:
    documents = []
    n = len(PdfReader(pdfpath).pages)
    bypass = n <= 10
    for i, page in enumerate(PdfReader(pdfpath).pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append(
                Document(
                    text=text.strip(),
                    metadata={
                        "file_name": os.path.basename(pdfpath),
                        "page_label": str(i + 1),
                        "total_pages": n,
                        "file_type": "pdf",
                        "bypass_embedding": bypass
                    }
                )
            )
    return documents

# im putting so much restriction in parameters so we dont get any errors
def loadDocs(file_paths: list[str]) -> list[Document]:
    documents = []
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        if ext in [".png", ".jpg", ".jpeg"]:
            documents.append(parseImgOCR(path))
        elif ext == ".pdf":
            documents.extend(parsePDF(path))
        elif ext in [".txt", ".md"]:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(
                    Document(
                        text = text.strip() ,
                        metadata = {
                            "file_name": os.path.basename(path) ,
                            "page_label": "1",
                            "file_type": "txt" ,
                            "bypass_embedding": len( text.split() ) < 3000
                        }))
        else:
            print(f"skip -unsupported file: {path}")
    return documents


def should_bypass_embedding(docs: list[Document]) -> bool:
    if not docs:
        return False
    return all(doc.metadata.get("bypass_embedding", False) for doc in docs)

# here im just adding the prompt and the OCR text simply - took help of ai for simpler logic here in this class method
def extract_combined_text(docs: list[Document]) -> str:
    return "\n\n".join([f"[{d.metadata.get('file_name', 'Doc')} - Page {d.metadata.get('page_label', '?')}]:\n{d.get_content()}" for d in docs])

if __name__ == "__main__":
    test_files = ["docs/lunorR1instruct.pdf"]
    if os.path.exists(test_files[0]):
        docs = loadDocs(test_files)
        print(f"Successs ! {len(docs)} chunks/pages.")
        print(f"Sample: {docs[0].get_content()[:200]}...")
    else:
        print("wrong path")