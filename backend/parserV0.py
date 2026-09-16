import os
from PIL import Image
import pytesseract
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader

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

# im putting so much restriction in parameters so we dont get any errors
def loadDocs(file_paths: list[str]) -> list[Document]:
    documents = []
    for path in file_paths:
        last = os.path.splitext(path)[1].lower()
        if last in [".png", ".jpg", ".jpeg"]:
            documents.append(parseImgOCR(path))
        elif last in [".pdf", ".txt", ".md"]:
            documents.extend(SimpleDirectoryReader(input_files=[path]).load_data())
        else:
            print(f"The given doc is not in support list only .png .jpg .jpeg alowed pls")
    return documents


if __name__ == "__main__":
    test_files = ["docs/lunorR1instruct.pdf"]
    if os.path.exists(test_files[0]):
        docs = loadDocs(test_files)
        print(f"Successs ! {len(docs)} chunks/pages.")
        print(f"Sample: {docs[0].get_content()[:200]}...")
    else:
        print("wrong path")