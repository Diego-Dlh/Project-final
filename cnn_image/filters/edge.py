from PIL import Image, ImageFilter

def edge_image(path):
    img = Image.open(path)
    edged = img.filter(ImageFilter.FIND_EDGES)
    edged.save("edged_"+path)
