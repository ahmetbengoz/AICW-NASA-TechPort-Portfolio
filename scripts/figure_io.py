"""Strict, atomic figure exports so concurrent readers never see partial PNGs."""
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageFile

def save_png_atomic(figure, path, dpi=600, **kwargs):
    path=Path(path)
    ImageFile.LOAD_TRUNCATED_IMAGES=False
    buffer=BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, facecolor="white",
                   transparent=False, **kwargs)
    buffer.seek(0)
    with Image.open(buffer) as source:
        source.verify()
    buffer.seek(0)
    with Image.open(buffer) as source:
        source.load()
        rgb=Image.new("RGB",source.size,"white")
        if source.mode=="RGBA":
            rgb.paste(source,mask=source.getchannel("A"))
        else:
            rgb.paste(source.convert("RGB"))
    temporary=path.with_suffix(".complete.png")
    rgb.save(temporary, format="PNG", dpi=(dpi,dpi))
    with Image.open(temporary) as check:
        check.verify()
    temporary.replace(path)
