import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO


model = YOLO("best_yolo_bone.pt")

st.title("Détection de fracture")
st.write("Chargez une image pour effectuer une détection.")

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_cv2 = np.array(image.convert("RGB")) # Convertion PIL to numpy

    # Afficher l'image originale
    st.subheader("Image Originale")
    st.image(image, caption="Image chargée", use_container_width=True)

    if st.button("Lancer la détection"):
        try:
            results = model(image_cv2)
            annotated_image = results[0].plot()

            annotated_pil = Image.fromarray(annotated_image)
            st.subheader("Image avec fracture détectée")
            st.image(annotated_pil, caption="Image avec détection", use_container_width=True)

            # Sauvegarde de l'image avec détection
            output_path = "output.jpg"
            annotated_pil.save(output_path)

            with open(output_path, "rb") as file:
                st.download_button(
                    label="Télécharger l'image annotée",
                    data=file,
                    file_name="detection_result.jpg",
                    mime="image/jpeg"
                )
        except Exception as e:
            st.error(f"Erreur lors de la détection : {e}")
