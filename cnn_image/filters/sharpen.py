from PIL import Image, ImageFilter

def sharpen_image(path):
    img = Image.open(path)
    sharp = img.filter(ImageFilter.SHARPEN)
    sharp.save("sharpened_"+path)
