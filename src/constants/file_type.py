from enum import Enum

class FileType(str,Enum):
    pdf = "pdf"
    docx = "docx"
    image = "image"
    video = "video"