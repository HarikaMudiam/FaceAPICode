import cv2
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

# Set up Azure Cognitive Face client
subscription_key = "0c15a9c0c70c4cf9a037425be108c7c3"
endpoint = "https://facerecogres.cognitiveservices.azure.com/"

face_client = FaceClient(endpoint, CognitiveServicesCredentials(subscription_key))

# Function to create a person group
def create_person_group(face_client, person_group_id, person_group_name):
    try:
        face_client.person_group.create(person_group_id=person_group_id, name=person_group_name)
    except Exception as e:
         print(f"Outer exception message: {str(e)}")
         print(f"Inner exception message: {str(e.__cause__)}")

# Function to add a person to a person group
def add_person(face_client, person_group_id, person_name, image_path):
    person = face_client.person_group_person.create(person_group_id=person_group_id, name=person_name)
    with open(image_path, "rb") as image_file:
        face_client.person_group_person.add_face_from_stream(person_group_id, person.person_id, image_file)
    return person.person_id

# Function to train the person group
def train_person_group(face_client, person_group_id):
    face_client.person_group.train(person_group_id)
    while True:
        training_status = face_client.person_group.get_training_status(person_group_id)
        if training_status.status == 'succeeded':
            break
        elif training_status.status == 'failed':
            raise Exception("Training failed")
        print("Training in progress...")
    
# Function to detect and identify faces in a frame
def identify_faces(face_client, person_group_id, frame):
    ret, buf = cv2.imencode('.jpg', frame)
    image_stream = buf.tobytes()
    face_ids = []
    faces = face_client.face.detect_with_stream(image_stream)
    for face in faces:
        face_ids.append(face.face_id)
    
    if not face_ids:
        return []
    
    results = face_client.face.identify(face_ids, person_group_id)
    return results

# Create a person group
person_group_id = "fab_family"
person_group_name = "FABFamily"
create_person_group(face_client, person_group_id, person_group_name)

# Add a person to the person group
person_name = "Harika"
image_path = "Harika.jpg"
print(image_path)
person_id = add_person(face_client, person_group_id, person_name, image_path)

# Train the person group
train_person_group(face_client, person_group_id)

# Open video file or capture device
video_path = "C:\\Users\\Harika Mudiam\\Downloads\\chatgpt.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Identify faces in the frame
    identified_faces = identify_faces(face_client, person_group_id, frame)
    
    for result in identified_faces:
        for candidate in result.candidates:
            person = face_client.person_group_person.get(person_group_id, candidate.person_id)
            print(f"Identified {person.name} with confidence {candidate.confidence}")

    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
