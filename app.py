import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import sqlite3
import uuid
import base64
from datetime import date, datetime


# ============================================================
# FUNDGRUBE
# Streamlit-App für verlorene und gefundene Gegenstände
#
# Benötigte Dateien im selben GitHub-Repository:
#   app.py
#   keras_model.h5
#   labels.txt
#   requirements.txt
# ============================================================


# ------------------------------------------------------------
# 1. GRUNDEINSTELLUNGEN
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"

# Die Datenbank und Bilder werden während der Laufzeit erzeugt.
# Es werden dafür KEINE zusätzlichen Dateien im GitHub-Repository
# benötigt.
DATABASE_PATH = BASE_DIR / "fundgrube.db"
UPLOAD_DIRECTORY = BASE_DIR / "fundgrube_uploads"

UPLOAD_DIRECTORY.mkdir(exist_ok=True)


st.set_page_config(
    page_title="Fundgrube",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ------------------------------------------------------------
# 2. DESIGN
# ------------------------------------------------------------

st.markdown(
    """
    <style>

    /* ======================================================
       ALLGEMEINES
       ====================================================== */

    .stApp {
        background-color: #F6F2E8;
        color: #20201D;
    }

    .main .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Streamlit Header */
    header[data-testid="stHeader"] {
        background-color: #F6F2E8;
    }


    /* ======================================================
       TITEL
       ====================================================== */

    .fundgrube-title {
        text-align: center;
        font-size: clamp(3rem, 8vw, 6rem);
        font-weight: 900;
        letter-spacing: -0.07em;
        color: #20201D;
        margin-top: 10px;
        margin-bottom: 10px;
    }


    /* ======================================================
       LUPENLOGO
       ====================================================== */

    .magnifier-container {
        width: 200px;
        height: 200px;
        margin: 10px auto 20px auto;
        position: relative;
    }

    .magnifier-glass {
        position: absolute;
        width: 130px;
        height: 130px;
        border: 14px solid #20201D;
        border-radius: 50%;
        left: 10px;
        top: 5px;
    }

    .magnifier-handle {
        position: absolute;
        width: 90px;
        height: 14px;
        background-color: #20201D;
        border-radius: 10px;
        transform: rotate(45deg);
        left: 108px;
        top: 125px;
    }


    /* ======================================================
       SEITENTITEL
       ====================================================== */

    .page-label {
        color: #9F4C28;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-size: 0.75rem;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .page-title {
        font-size: clamp(2.2rem, 5vw, 4rem);
        font-weight: 900;
        letter-spacing: -0.06em;
        margin-top: 0;
        margin-bottom: 5px;
    }

    .page-description {
        color: #716F68;
        font-size: 1rem;
        margin-bottom: 30px;
    }


    /* ======================================================
       KARTEN
       ====================================================== */

    .fund-card {
        background: #FFFDF8;
        border: 1px solid #D8D2C5;
        border-radius: 18px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 10px 25px rgba(39, 33, 24, 0.06);
    }

    .fund-category {
        font-size: 1.3rem;
        font-weight: 800;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .fund-information {
        color: #716F68;
        font-size: 0.9rem;
        line-height: 1.5;
    }


    /* ======================================================
       KLASSIFIZIERUNG
       ====================================================== */

    .classification-box {
        background: #FFFDF8;
        border: 1px solid #D8D2C5;
        border-radius: 18px;
        padding: 25px;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .classification-title {
        color: #9F4C28;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.75rem;
        font-weight: 800;
    }

    .classification-category {
        font-size: 2.2rem;
        font-weight: 900;
        margin-top: 5px;
    }

    .confidence {
        color: #716F68;
        margin-top: 5px;
    }


    /* ======================================================
       HAMBURGER / SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background-color: #FFFDF8;
    }

    section[data-testid="stSidebar"] h2 {
        font-weight: 900;
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {
        border-radius: 12px;
        font-weight: 750;
        min-height: 45px;
    }


    /* ======================================================
       UPLOAD
       ====================================================== */

    [data-testid="stFileUploader"] {
        background-color: #FFFDF8;
        border: 2px dashed #D8D2C5;
        border-radius: 18px;
        padding: 10px;
    }


    /* ======================================================
       MOBILE
       ====================================================== */

    @media (max-width: 700px) {

        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .fundgrube-title {
            font-size: 3.5rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 3. LABELS LADEN
# ------------------------------------------------------------

def load_labels():

    if not LABELS_PATH.exists():
        st.error(
            "Die Datei labels.txt wurde nicht gefunden. "
            "Bitte stelle sicher, dass sie im selben GitHub-Repository "
            "wie app.py liegt."
        )
        st.stop()

    labels = []

    with open(LABELS_PATH, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            # Unterstützt:
            #
            # 0 Hose
            # 1 Schuh
            # 2 T-Shirt
            # 3 Hoodie
            #
            # und auch:
            #
            # Hose
            # Schuh
            # T-Shirt
            # Hoodie

            parts = line.split(maxsplit=1)

            if len(parts) == 2 and parts[0].isdigit():

                labels.append(parts[1].strip())

            else:

                labels.append(line)

    if len(labels) != 4:

        st.error(
            "Die labels.txt muss genau vier Klassen enthalten."
        )

        st.stop()

    return labels


LABELS = load_labels()


# ------------------------------------------------------------
# 4. KI-MODELL LADEN
# ------------------------------------------------------------

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        st.error(
            "Die Datei keras_model.h5 wurde nicht gefunden. "
            "Bitte stelle sicher, dass sie im selben GitHub-Repository "
            "wie app.py liegt."
        )

        st.stop()

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


MODEL = load_model()


# ------------------------------------------------------------
# 5. DATENBANK
# ------------------------------------------------------------

def initialize_database():

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS found_items (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            image_filename TEXT NOT NULL,

            category TEXT NOT NULL,

            confidence REAL NOT NULL,

            location TEXT NOT NULL,

            found_date TEXT NOT NULL,

            created_at TEXT NOT NULL

        )
        """
    )

    connection.commit()

    connection.close()


initialize_database()


# ------------------------------------------------------------
# 6. BILD KLASSIFIZIEREN
# ------------------------------------------------------------

def classify_image(uploaded_file):

    image = Image.open(uploaded_file)

    image = image.convert("RGB")

    # Teachable Machine benötigt 224 x 224.
    image = image.resize((224, 224))

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Standardisierung für das Teachable-Machine-Modell.
    image_array = image_array / 127.5 - 1.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = MODEL.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[predicted_index]
    )

    category = LABELS[predicted_index]

    return category, confidence, predictions


# ------------------------------------------------------------
# 7. BILD SPEICHERN
# ------------------------------------------------------------

def save_image(uploaded_file):

    extension = Path(
        uploaded_file.name
    ).suffix.lower()

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    if extension not in allowed_extensions:
        extension = ".jpg"

    filename = (
        str(uuid.uuid4())
        + extension
    )

    destination = (
        UPLOAD_DIRECTORY
        / filename
    )

    with open(destination, "wb") as file:

        file.write(
            uploaded_file.getbuffer()
        )

    return filename


# ------------------------------------------------------------
# 8. FUNDSTÜCK SPEICHERN
# ------------------------------------------------------------

def save_found_item(
    image_filename,
    category,
    confidence,
    location,
    found_date
):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO found_items
        (
            image_filename,
            category,
            confidence,
            location,
            found_date,
            created_at
        )

        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            image_filename,
            category,
            confidence,
            location,
            found_date.isoformat(),
            datetime.now().isoformat()
        )
    )

    connection.commit()

    connection.close()


# ------------------------------------------------------------
# 9. FUNDSTÜCKE LADEN
# ------------------------------------------------------------

def get_items(search_text=""):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    if search_text.strip():

        search_pattern = (
            "%"
            + search_text.strip()
            + "%"
        )

        cursor.execute(
            """
            SELECT *
            FROM found_items

            WHERE
                category LIKE ?
                OR location LIKE ?

            ORDER BY
                found_date ASC,
                id ASC
            """,
            (
                search_pattern,
                search_pattern
            )
        )

    else:

        cursor.execute(
            """
            SELECT *
            FROM found_items

            ORDER BY
                found_date ASC,
                id ASC
            """
        )

    items = cursor.fetchall()

    connection.close()

    return items


# ------------------------------------------------------------
# 10. ÄLTESTE NEUN FUNDSTÜCKE
# ------------------------------------------------------------

def get_oldest_items():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM found_items

        ORDER BY
            found_date ASC,
            id ASC

        LIMIT 9
        """
    )

    items = cursor.fetchall()

    connection.close()

    return items


# ------------------------------------------------------------
# 11. BILD ALS BASE64 FÜR HTML
# ------------------------------------------------------------

def image_to_base64(image_path):

    try:

        with open(
            image_path,
            "rb"
        ) as image_file:

            encoded = base64.b64encode(
                image_file.read()
            ).decode()

        extension = (
            image_path
            .suffix
            .lower()
            .replace(".", "")
        )

        if extension == "jpg":
            extension = "jpeg"

        return (
            f"data:image/{extension};base64,"
            f"{encoded}"
        )

    except Exception:

        return None


# ------------------------------------------------------------
# 12. HAMBURGER-NAVIGATION
# ------------------------------------------------------------

def navigation():

    with st.sidebar:

        st.markdown(
            "## 🔎 Fundgrube"
        )

        st.markdown(
            "### Navigation"
        )

        page = st.radio(
            "Seite auswählen",
            [
                "Suchen",
                "Fundstück hochladen",
                "Älteste Fundstücke"
            ],
            label_visibility="collapsed"
        )

        st.divider()

        st.caption(
            "Das Menü kann über das "
            "Hamburger-Symbol geöffnet "
            "und geschlossen werden."
        )

    return page


# ------------------------------------------------------------
# 13. SEITE 1 – SUCHEN
# ------------------------------------------------------------

def search_page():

    st.markdown(
        '<div class="fundgrube-title">'
        'Fundgrube'
        '</div>',
        unsafe_allow_html=True
    )

    # Lupengrafik
    st.markdown(
        """
        <div class="magnifier-container">

            <div class="magnifier-glass"></div>

            <div class="magnifier-handle"></div>

        </div>
        """,
        unsafe_allow_html=True
    )

    search_text = st.text_input(
        "Gesuchten Gegenstand eingeben",
        placeholder="Was suchst du?",
        label_visibility="collapsed"
    )

    if search_text:

        st.markdown(
            f"### Suchergebnisse für „{search_text}“"
        )

    else:

        st.markdown(
            """
            <p style="
                text-align:center;
                color:#716F68;
                margin-bottom:30px;
            ">
                Suche nach einem verlorenen Gegenstand.
            </p>
            """,
            unsafe_allow_html=True
        )

    items = get_items(
        search_text
    )

    if not items:

        if search_text:

            st.info(
                "Leider wurde kein passendes "
                "Fundstück gefunden."
            )

        else:

            st.info(
                "Momentan befinden sich noch "
                "keine Fundstücke in der Fundgrube."
            )

        return

    columns = st.columns(3)

    for index, item in enumerate(items):

        with columns[index % 3]:

            image_path = (
                UPLOAD_DIRECTORY
                / item["image_filename"]
            )

            if image_path.exists():

                st.image(
                    str(image_path),
                    use_container_width=True
                )

            st.markdown(
                f"""
                <div class="fund-card">

                    <div class="fund-category">
                        {item["category"]}
                    </div>

                    <div class="fund-information">
                        <strong>Fundort:</strong>
                        {item["location"]}
                    </div>

                    <div class="fund-information">
                        <strong>Funddatum:</strong>
                        {item["found_date"]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ------------------------------------------------------------
# 14. SEITE 2 – UPLOAD
# ------------------------------------------------------------

def upload_page():

    st.markdown(
        '<div class="page-label">Seite 2</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">'
        'Fundstück hochladen'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="page-description">
            Lade ein Foto hoch. Die KI erkennt anschließend
            automatisch, um welche Kategorie es sich handelt.
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Bild aus der Mediathek auswählen",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        help="Erlaubte Bildformate: JPG, JPEG, PNG und WEBP."
    )

    if uploaded_file is None:

        st.markdown(
            """
            <div class="classification-box">

                <div class="classification-title">
                    KI-Kategorien
                </div>

                <p>
                    Die KI kann folgende Gegenstände erkennen:
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        category_columns = st.columns(4)

        for index, label in enumerate(LABELS):

            with category_columns[index]:

                st.markdown(
                    f"""
                    <div class="fund-card"
                         style="text-align:center;">
                        <strong>{label}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        return


    # --------------------------------------------------------
    # BILD VORSCHAU
    # --------------------------------------------------------

    st.markdown(
        "### Hochgeladenes Bild"
    )

    st.image(
        uploaded_file,
        width=350
    )


    # --------------------------------------------------------
    # KI-ANALYSE
    # --------------------------------------------------------

    with st.status(
        "KI analysiert das Bild …",
        expanded=True
    ) as status:

        st.write(
            "Bild wird vorbereitet …"
        )

        category, confidence, predictions = (
            classify_image(uploaded_file)
        )

        st.write(
            "Teachable-Machine-Modell "
            "wertet das Bild aus …"
        )

        st.write(
            "Klassifizierung abgeschlossen."
        )

        status.update(
            label="KI-Klassifizierung abgeschlossen",
            state="complete",
            expanded=False
        )


    # --------------------------------------------------------
    # ERGEBNIS
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="classification-box">

            <div class="classification-title">
                Erkannte Kategorie
            </div>

            <div class="classification-category">
                {category}
            </div>

            <div class="confidence">
                Sicherheit der Klassifizierung:
                {confidence * 100:.1f} %
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # FUNDORT UND DATUM
    # --------------------------------------------------------

    st.markdown(
        "### Fundinformationen"
    )

    location = st.text_input(
        "Wo wurde der Gegenstand gefunden?",
        placeholder=(
            "z. B. Aula, Sporthalle, "
            "2. Obergeschoss"
        )
    )

    found_date = st.date_input(
        "An welchem Tag wurde der Gegenstand gefunden?",
        value=date.today()
    )


    # --------------------------------------------------------
    # SPEICHERN
    # --------------------------------------------------------

    if st.button(
        "Fundstück speichern",
        type="primary",
        use_container_width=True
    ):

        if not location.strip():

            st.warning(
                "Bitte gib zuerst den Fundort ein."
            )

            return

        image_filename = save_image(
            uploaded_file
        )

        save_found_item(
            image_filename=image_filename,
            category=category,
            confidence=confidence,
            location=location.strip(),
            found_date=found_date
        )

        st.success(
            "Das Fundstück wurde erfolgreich "
            "in der Fundgrube gespeichert."
        )

        st.balloons()

        st.info(
            "Du kannst jetzt ein weiteres "
            "Fundstück hochladen."
        )


# ------------------------------------------------------------
# 15. SEITE 3 – ÄLTESTE FUNDSTÜCKE
# ------------------------------------------------------------

def oldest_page():

    st.markdown(
        '<div class="page-label">Seite 3</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">'
        'Älteste Fundstücke'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="page-description">
            Hier werden die neun ältesten Fundstücke angezeigt.
        </div>
        """,
        unsafe_allow_html=True
    )

    items = get_oldest_items()

    if not items:

        st.info(
            "Es wurden noch keine Fundstücke "
            "gespeichert."
        )

        return


    columns = st.columns(3)

    for index, item in enumerate(items):

        with columns[index % 3]:

            image_path = (
                UPLOAD_DIRECTORY
                / item["image_filename"]
            )

            if image_path.exists():

                st.image(
                    str(image_path),
                    use_container_width=True
                )

            st.markdown(
                f"""
                <div class="fund-card">

                    <div class="fund-category">
                        {item["category"]}
                    </div>

                    <div class="fund-information">
                        <strong>Fundort:</strong>
                        {item["location"]}
                    </div>

                    <div class="fund-information">
                        <strong>Funddatum:</strong>
                        {item["found_date"]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ------------------------------------------------------------
# 16. HAUPTPROGRAMM
# ------------------------------------------------------------

def main():

    selected_page = navigation()

    if selected_page == "Suchen":

        search_page()

    elif selected_page == "Fundstück hochladen":

        upload_page()

    elif selected_page == "Älteste Fundstücke":

        oldest_page()


# ------------------------------------------------------------
# APP STARTEN
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
