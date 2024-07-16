# Standard library for creating Graphical User Interfaces
from tkinter import *
from tkinter import scrolledtext,messagebox,filedialog
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from PIL import Image

# Load machine learning model and CountVectorizer
model = joblib.load('SVC.pkl')
vectorizer = joblib.load('countVectorizer.pkl')

#Intialize the PorterStemmer
stemmer = PorterStemmer()

# Function to predict spam and ham
def predict_spam_ham():
    text = text_input.get('1.0', 'end-1c')  # Get text from the input field
    if len(text) !=0:
        # Convert to lowercase
        text = text.lower()  
        # Tokenize the text
        text = nltk.word_tokenize(text) 
        # Remove non-alphanumeric characters
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)  
        text = y[:]
        y.clear()
        # Remove stopwords and punctuation
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)  
        text = y[:]
        y.clear()
        # Apply stemming
        for i in text:
            y.append(stemmer.stem(i))  
        final = " ".join(y)

        # Convert text to feature vectors
        input_mail_features = vectorizer.transform([final]).toarray()

        # Making prediction
        prediction = model.predict(input_mail_features)

        # Display prediction result
        if prediction[0] == 1:
            text_display.config(text="Prediction -> Ham")
        else:
            text_display.config(text="Prediction -> Spam")
    
    else:
        messagebox.showerror("Input Error", "Please enter the message.")
    
# Function to convert text to binary
def text_to_binary(text):
    """Converts ASCII text to binary representation."""
    binary = ''.join(format(ord(char), '08b') for char in text)
    return binary

# Function to convert binary to text
def binary_to_text(binary):
    """Converts binary representation back to ASCII text."""
    text = ''.join(chr(int(binary[i:i+8], 2)) for i in range(0, len(binary), 8))
    return text

# Function used in hiding
def hide_text_in_image(image_path, text_to_hide, output_path):
    """Hide text inside an image using LSB steganography."""
    try:
        img = Image.open(image_path).convert("RGB")
        binary_text = text_to_binary(text_to_hide) + '1111111111111110'  # append a delimiter
        text_length = len(binary_text)

        width, height = img.size
        pixels = img.load()
        if text_length > (width * height )/8:
            raise ValueError("Text too long to fit in the image")

        index = 0
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]

                # Modify the least significant bit only if there's still data to store
                if index < text_length:
                    pixels[x, y] = (r & 254 | int(binary_text[index]), g, b)
                    index += 1

        img.save(output_path)
        text_display.config(text="Success : Text hidden successfully in Image")
    except Exception as e:
        text_display.config(text=f"Error hiding text in image: {e}")

# Function to hide text in image
def hide_text():
    """Callback function for hiding text."""
    text_to_hide = text_input.get('1.0', 'end-1c')
    if len(text_to_hide) == 0:
        messagebox.showerror("Input Error", "Please enter the text to encode.")
    else:
        image_path = filedialog.askopenfilename(initialdir="./", title="Select Image File", filetypes=(("PNG files", "*.png"), ("JPG files", "*.jpg"), ("All files", "*.*")))
        if image_path:
            output_path = filedialog.asksaveasfilename(initialdir="./", title="Save Image As", filetypes=(("PNG files", "*.PNG"), ("JPG files", "*.jpg"), ("All files", "*.*")))
            if output_path:
                try:
                    hide_text_in_image(image_path, text_to_hide, output_path)
                except Exception as e:
                    print(f"Error: {e}")

# Function used in retrieving
def reveal_text_in_image(image_path):
    """Extracts hidden text from an image."""
    try:
        img = Image.open(image_path)
        width, height = img.size
        pixels = img.load()

        binary_text = ''
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                binary_text += str(r & 1)

                # Check for the delimiter '1111111111111110'
                if binary_text[-16:] == '1111111111111110':
                    hidden_text = binary_to_text(binary_text[:-16])
                    return hidden_text

        return binary_to_text(binary_text)
    except Exception as e:
        messagebox.showinfo("Info", "The selected image does not contain any hidden text.")
        clear_input()

# Function to retrieve the text from image
def retrieve_text():
    """Callback function for retrieving hidden text."""
    clear_input()
    text_label.config(text="Retrieved text from the image:")
    image_path = filedialog.askopenfilename(initialdir="./", title="Select Image File", filetypes=(("PNG files", "*.png"),))
    if image_path:
        try:
            hidden_text = reveal_text_in_image(image_path)
            if len(hidden_text) != 0:
                text_input.insert('1.0', hidden_text)
        except Exception as e:
            print()

# Function to test whether image contains the message or not
def test():
    clear_input()
    image_path = filedialog.askopenfilename(initialdir="./", title="Select Image File", filetypes=(("PNG files", "*.png"),))

    if image_path:
        try:
            img = Image.open(image_path)
            width, height = img.size
            pixels = img.load()

            binary_text = ''
            found_text=False
            for y in range(height):
                for x in range(width):
                    r, g, b = pixels[x, y]
                    if r & 1 == 1:  # Check LSB of red channel
                        binary_text += '1'
                    else:
                        binary_text += '0'

                    # Check for the delimiter '1111111111111110'
                    if len(binary_text) >= 16 and binary_text[-16:] == '1111111111111110':
                        found_text = True
                        break
                if found_text:
                    break

            if found_text:
                text_display.config(text="The Image contains hidden message")
            else:
                messagebox.showinfo("Info", "The selected image does not contain any hidden text.")
                clear_input()
        except Exception as e:
            messagebox.showinfo("Info", "The selected image does not contain any hidden text.")
            clear_input()
    
# Function to clear
def clear_input():
    # Clear text input
    text_input.delete('1.0', END)
    text_label.config(text="Enter the text to Hide or Message to find Spam/Ham")
    # Clear prediction result
    text_display.config(text="")

root = Tk() # Creating instance of Tk class
root.title("Mini Project")

text_label = Label(root,text="Enter the text to Hide or Message to find Spam/Ham")
text_label.pack(pady=5)

# For text input via gui from user
text_input = scrolledtext.ScrolledText(root,width=40,height=10,wrap=WORD)
text_input.pack(pady=10)

frame = Frame(root)
frame.pack(pady=5)

# Creating a button to hide the text in Image
hide_button = Button(frame, text="Encode Text", command=hide_text)
hide_button.pack(side="left",padx=8)

# Creating a button to retrieve text from Image
retrieve_button = Button(frame, text="Decode Text", command=retrieve_text)
retrieve_button.pack(side="right",padx=8)

# Creating a button to test whether the image contains the text or not
test_button = Button(root, text="Testing", command=test)
test_button.pack(pady=5)

# Creating a button to make predictions whether the message in Image is Spam or Ham
predict_button = Button(root, text="Ham/Spam", command=predict_spam_ham)
predict_button.pack(pady=5)

frame_1 = Frame(root)
frame_1.pack(pady=5)

# Creating a button to clear text input feild and reset everything
clear_button = Button(frame_1, text="Clear", command=clear_input)
clear_button.pack(side="left",padx=8)

# Add a button to close the window
close_button = Button(frame_1, text="Exit", command=root.destroy)
close_button.pack(side="right",padx=8)

# Adding a label to show outputs
text_display = Label(root)
text_display.pack(pady=5)

root.mainloop()# Entering the main event loop.
