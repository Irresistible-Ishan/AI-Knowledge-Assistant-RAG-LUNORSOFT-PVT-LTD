#v2 code last one didnt work cuz somehow it was parsing pdf as an image ill write a seprate code for pdf using pypdf

import os
from PIL import Image
import pytesseract
from llama_index.core import Document
# uhhh this is not working for pdf somehow
from llama_index.core import SimpleDirectoryReader
from pypdf import PdfReader

def parseImgOCR(imgpath) -> Document:
    img = Image.open(imgpath)
    text = pytesseract.image_to_string(image)
    return Document(
        text = text ,
        metadata = {
            "file_name" : os.path.basename(imgpath),
            "page_label" : "imgOCR",
            "file_type" : "image"
        })

def parsePDF(pdfpath : str) -> list[Document]:
    documents = []
    reader = PdfReader(pdfpath)
    for i , page in enumerate(reader.pages) :
        text = page.extract_text()
        if text and text.strip():
            documents.append(Document(
                text=text.strip(),
                metadata={
                    "file_name": os.path.basename(pdfpath) ,
                    "page_label": str(i + 1) ,
                    "file_type": "pdf"
                }))
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
                documents.append(Document(text=text ,metadata={ 
                        "file_name": os.path.basename(path) ,
                            "page_label" : "Text File" ,
                            "file_type": "txt"}))
        else:
            print(f"The given doc is not in support list only .png .jpg .jpeg .pdf .txt .md alowed pls")
    return documents


if __name__ == "__main__":
    test_files = ["docs/lunorR1instruct.pdf"]
    if os.path.exists(test_files[0]):
        docs = loadDocs(test_files)
        print(f"Successs ! {len(docs)} chunks/pages.")
        print(f"Sample: {docs[0].get_content()[:200]}...")
    else:
        print("wrong path")