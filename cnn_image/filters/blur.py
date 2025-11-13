from PIL import Image, ImageFilter

def blur_image(path):
    img = Image.open(path)
    blur = img.filter(ImageFilter.GaussianBlur(radius=2))
    blur.save("blurred_"+path)
