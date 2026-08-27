import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import easyocr
import cv2


# CONFIG

st.set_page_config(page_title="AI Data Structure Detector", layout="wide")


# LOAD MODELS

@st.cache_resource
def load_models():
    model = YOLO("runs/detect/train9/weights/best.pt")
    reader = easyocr.Reader(['en'])
    return model, reader

model, reader = load_models()


# HELPER FUNCTION

def detect_nodes(img_np):
    results = model(img_np)

    boxes = []
    arrows = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 0:
                boxes.append((x1, y1, x2, y2))
            elif cls == 1:
                arrows += 1

    boxes = sorted(boxes, key=lambda b: b[0])

    values = []

    for (x1, y1, x2, y2) in boxes:
        crop = img_np[y1:y2, x1:x2]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        result = reader.readtext(thresh, detail=0)
        values.append(result[0] if result else "?")

    return values, arrows


# TITLE

st.title("🧠 AI Data Structure → Code Generator")

operation = st.selectbox(
    "⚙️ Operation",
    ["Single Structure", "Merge Two Structures (Image-Based)"]
)


# FIRST IMAGE

uploaded_file = st.file_uploader("Upload First Image", type=["jpg", "png", "jpeg"])

nodes_1 = []
arrows_1 = 0
structure_1 = None

if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    st.subheader("📷 First Image")
    st.image(image, width=400)

    nodes_1, arrows_1 = detect_nodes(img_np)

    # Structure detection (FIRST IMAGE)
    structure_1 = "Linked List" if arrows_1 > 0 else "Array"
    st.success(f"First Image Detected: {structure_1}")

    # Editable nodes
    st.subheader("⚙️ Nodes (First Image)")
    cols = st.columns(len(nodes_1)) if nodes_1 else []

    edited_nodes = []
    for i, val in enumerate(nodes_1):
        with cols[i]:
            new_val = st.text_input(f"Node {i+1}", val, key=f"node1_{i}")
            edited_nodes.append(new_val)

    nodes_1 = edited_nodes


# SECOND IMAGE

nodes_2 = []
arrows_2 = 0
structure_2 = None

if operation == "Merge Two Structures (Image-Based)" and uploaded_file:
    uploaded_file2 = st.file_uploader("Upload Second Image", type=["jpg", "png", "jpeg"])

    if uploaded_file2:
        image2 = Image.open(uploaded_file2)
        img_np2 = np.array(image2)

        st.subheader("📷 Second Image")
        st.image(image2, width=400)

        nodes_2, arrows_2 = detect_nodes(img_np2)

        # Structure detection (SECOND IMAGE)
        structure_2 = "Linked List" if arrows_2 > 0 else "Array"
        st.success(f"Second Image Detected: {structure_2}")

        # Editable nodes
        st.subheader("⚙️ Nodes (Second Image)")
        cols2 = st.columns(len(nodes_2)) if nodes_2 else []

        edited_nodes2 = []
        for i, val in enumerate(nodes_2):
            with cols2[i]:
                new_val = st.text_input(f"Node {i+1}", val, key=f"node2_{i}")
                edited_nodes2.append(new_val)

        nodes_2 = edited_nodes2


# STRUCTURE VALIDATION + WARNING

detected_structure = None

if nodes_1:
    if nodes_2:
        if structure_1 == structure_2:
            detected_structure = structure_1
        else:
            st.error("⚠️ Structure mismatch detected!")
            st.warning(
                f"First Image: {structure_1} | Second Image: {structure_2}\n\n"
                "👉 Please upload a matching structure OR adjust your input."
            )
            st.stop()
    else:
        detected_structure = structure_1


# MERGE LOGIC

result_nodes = nodes_1

if operation == "Merge Two Structures (Image-Based)" and nodes_1 and nodes_2:
    result_nodes = nodes_1 + nodes_2


# LANGUAGE + CODE GENERATION

if result_nodes:
    st.subheader("🌐 Select Language")

    lang = st.radio(
        "Choose language:",
        ["Python", "C"],
        horizontal=True
    )

    st.subheader(" Generated Code")

    nodes = result_nodes

    # PYTHON
    if lang == "Python":

        if detected_structure == "Array":
            code = f"arr = {nodes}"

        else:
            code = "class Node:\n"
            code += "    def __init__(self, data):\n"
            code += "        self.data = data\n"
            code += "        self.next = None\n\n"

            for i, val in enumerate(nodes):
                if i == 0:
                    code += f"head = Node({val})\n"
                else:
                    code += f"node{i} = Node({val})\n"

            for i in range(len(nodes)-1):
                if i == 0:
                    code += "head.next = node1\n"
                else:
                    code += f"node{i}.next = node{i+1}\n"

    # C
    else:

        if detected_structure == "Array":
            values_str = ", ".join(nodes)
            size = len(nodes)

            code = f"""#include <stdio.h>

int main() {{
    int arr[{size}] = {{{values_str}}};

    for(int i = 0; i < {size}; i++) {{
        printf("%d ", arr[i]);
    }}

    return 0;
}}"""

        else:
            code = """#include <stdio.h>
#include <stdlib.h>

struct Node {
    int data;
    struct Node* next;
};

int main() {

"""

            for i in range(len(nodes)):
                code += f"    struct Node* node{i} = (struct Node*)malloc(sizeof(struct Node));\n"

            code += "\n"

            for i, val in enumerate(nodes):
                code += f"    node{i}->data = {val};\n"

            code += "\n"

            for i in range(len(nodes)-1):
                code += f"    node{i}->next = node{i+1};\n"

            code += f"    node{len(nodes)-1}->next = NULL;\n\n"

            code += """    struct Node* head = node0;
    struct Node* temp = head;

    while(temp != NULL) {
        printf("%d -> ", temp->data);
        temp = temp->next;
    }

    printf("NULL\\n");

    return 0;
}"""

    st.code(code, language=lang.lower())

    file_ext = "py" if lang == "Python" else "c"

    st.download_button(
        "📥 Download Code",
        code,
        f"generated_code.{file_ext}"
    )