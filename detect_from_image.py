import face_recognition
import cv2
import os

# --- Step 1: Load known faces and teach the program how to recognize them ---
print("Loading known faces...")

# Load the image of the known person (Messi)
known_image = face_recognition.load_image_file("Messi1.webp")
# Get the face encoding (a unique mathematical representation of the face)
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Create a list of all known face encodings and their names
known_face_encodings = [known_face_encoding]
known_face_names = ["Lionel Messi"]

# --- Step 2: Load an unknown image to find and identify faces ---
unknown_image_path = "Messi1.webp" # This project includes a second photo of Messi
print(f"Loading unknown image: {unknown_image_path}")

# Check if the file exists
if not os.path.exists(unknown_image_path):
    print(f"Error: Could not find the test image at {unknown_image_path}")
else:
    # Load the unknown image
    unknown_image = face_recognition.load_image_file(unknown_image_path)
    # Convert it to a format OpenCV can use
    unknown_image_cv = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)
    
    # Find all faces and their encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # --- Step 3: Loop through each face found in the unknown image ---
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face matches any of the known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
        name = "Unknown" # Default name if no match is found

        # If a match was found, use the name of the first one
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        # Draw a box around the face
        cv2.rectangle(unknown_image_cv, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # Draw a label with the name below the face
        cv2.rectangle(unknown_image_cv, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(unknown_image_cv, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Save and show the final image
    output_filename = "recognized_faces.jpg"
    print(f"Saving new image with recognized faces to {output_filename}")
    cv2.imwrite(output_filename, unknown_image_cv)
    print("Success!")
