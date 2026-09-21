import streamlit as st
import tensorflow as tf
import tf_keras
import numpy as np
from PIL import Image
from pathlib import Path
import sqlite3
import uuid
from datetime import date, datetime


# ============================================================
# GRUNDEINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="Fundgrube",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# DATEIPFADE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"

UPLOAD_DIR = BASE_DIR / "fundgrube_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

DATABASE_PATH = BASE_DIR / "fundgrube.db"


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f7f8fc;
    }

    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        color: #222222;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .search-icon {
        text-align: center;
        font-size: 80px;
        margin-bottom: 5px;
    }

    .item-card {
        background: white;
        border-radius: 18px;
        padding: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }

    .item-card img {
        border-radius: 12px;
    }

    .item-title {
        font-size: 20px;
        font-weight: 700;
        margin-top: 10px;
    }

    .item-info {
        color: #666666;
        font-size: 14px;
        margin-top: 5px;
    }

    .classification-box {
        background: white;
        border-radius: 18px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LABELS LADEN
# ============================================================

def load_labels():
    labels = []

    if not LABELS_PATH.exists():
        st.error("Die Datei labels.txt wurde nicht gefunden.")
        return labels

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            # Entfernt z. B. "0 " aus "0 Hose"
            parts = line.split(" ", 1)

            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(line)

    return labels


LABELS = load_labels()


# ============================================================
# MODELL LADEN
# ============================================================

@st.cache_resource
def load_model():
    """
    Lädt das Teachable-Machine-Modell mit tf_keras.

    Das ist wichtig, weil das ältere .h5-Modell von
    Teachable Machine mit der aktuellen Keras-Version
    Probleme beim Laden verursachen kann.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "keras_model.h5 wurde nicht gefunden."
        )

    model = tf_keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


# ============================================================
# DATENBANK
# ============================================================

def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def init_database():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            item_type TEXT NOT NULL,
            location TEXT NOT NULL,
            found_date TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


init_database()


# ============================================================
# BILD VORBEREITEN
# ============================================================

def prepare_image(image):
    """
    Bereitet das Bild für das Teachable-Machine-Modell vor.

    Teachable Machine verwendet hier:
    224 x 224 Pixel
    RGB
    Werte zwischen -1 und 1
    """

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image)

    image_array = image_array.astype(np.float32)

    image_array = image_array / 127.5 - 1.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ============================================================
# KLASSIFIZIERUNG
# ============================================================

def classify_image(image):
    model = load_model()

    prepared_image = prepare_image(image)

    prediction = model.predict(
        prepared_image,
        verbose=0
    )

    prediction = np.asarray(prediction)

    # Falls das Modell mehrere Dimensionen zurückgibt
    prediction = prediction.reshape(-1)

    best_index = int(np.argmax(prediction))

    confidence = float(prediction[best_index])

    if best_index < len(LABELS):
        label = LABELS[best_index]
    else:
        label = f"Klasse {best_index}"

    return label, confidence


# ============================================================
# FUNDSTÜCK SPEICHERN
# ============================================================

def save_item(image, item_type, location, found_date):
    unique_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex[:8]}.jpg"
    )

    image_path = UPLOAD_DIR / unique_name

    image = image.convert("RGB")
    image.save(image_path, format="JPEG", quality=90)

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO items
        (
            filename,
            item_type,
            location,
            found_date,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            unique_name,
            item_type,
            location,
            str(found_date),
            datetime.now().isoformat()
        )
    )

    connection.commit()
    connection.close()


# ============================================================
# FUNDSTÜCKE AUS DATENBANK LADEN
# ============================================================

def get_items(search_term=None, limit=None):
    connection = get_connection()
    cursor = connection.cursor()

    if search_term:
        query = """
            SELECT
                id,
                filename,
                item_type,
                location,
                found_date,
                created_at
            FROM items
            WHERE item_type LIKE ?
            ORDER BY found_date DESC
        """

        cursor.execute(
            query,
            (f"%{search_term}%",)
        )

    else:
        query = """
            SELECT
                id,
                filename,
                item_type,
                location,
                found_date,
                created_at
            FROM items
            ORDER BY found_date ASC
        """

        if limit:
            query += f" LIMIT {int(limit)}"

        cursor.execute(query)

    items = cursor.fetchall()

    connection.close()

    return items


# ============================================================
# BILD ANZEIGEN
# ============================================================

def get_image_path(filename):
    return UPLOAD_DIR / filename


# ============================================================
# NAVIGATION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Suche"


# Streamlit-Menü
with st.sidebar:

    st.markdown("## ☰ Menü")

    if st.button(
        "🔎 Suche",
        use_container_width=True
    ):
        st.session_state.page = "Suche"
        st.rerun()

    if st.button(
        "📤 Fundstück hochladen",
        use_container_width=True
    ):
        st.session_state.page = "Hochladen"
        st.rerun()

    if st.button(
        "🕐 Älteste Fundstücke",
        use_container_width=True
    ):
        st.session_state.page = "Älteste Fundstücke"
        st.rerun()


# ============================================================
# SEITE 1 – SUCHE
# ============================================================

if st.session_state.page == "Suche":

    st.markdown(
        '<div class="main-title">Fundgrube</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="search-icon">🔎</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Finde verlorene Gegenstände wieder'
        '</div>',
        unsafe_allow_html=True
    )

    search = st.text_input(
        "Was suchst du?",
        placeholder="z. B. Hoodie, Schuh, Hose ..."
    )

    if search.strip():

        items = get_items(
            search_term=search.strip()
        )

        st.markdown(
            f"### Suchergebnisse für „{search.strip()}“"
        )

        if not items:

            st.info(
                "Leider wurde kein passendes Fundstück gefunden."
            )

        else:

            columns = st.columns(3)

            for index, item in enumerate(items):

                (
                    item_id,
                    filename,
                    item_type,
                    location,
                    found_date,
                    created_at
                ) = item

                image_path = get_image_path(filename)

                with columns[index % 3]:

                    st.markdown(
                        '<div class="item-card">',
                        unsafe_allow_html=True
                    )

                    if image_path.exists():
                        st.image(
                            str(image_path),
                            use_container_width=True
                        )

                    st.markdown(
                        f'<div class="item-title">'
                        f'{item_type}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f'<div class="item-info">'
                        f'📍 {location}<br>'
                        f'📅 {found_date}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        '</div>',
                        unsafe_allow_html=True
                    )


# ============================================================
# SEITE 2 – HOCHLADEN
# ============================================================

elif st.session_state.page == "Hochladen":

    st.markdown(
        '<div class="main-title">Fundstück hochladen</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Lade ein Foto hoch und die KI erkennt den Gegenstand.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Bild auswählen",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Ausgewähltes Bild",
            use_container_width=True
        )

        st.markdown("### 🤖 KI-Klassifizierung")

        # Sichtbarer Klassifizierungsprozess
        with st.status(
            "KI analysiert das Bild ...",
            expanded=True
        ) as status:

            st.write("📷 Bild wird vorbereitet ...")

            prepared_image = prepare_image(image)

            st.write("🧠 Teachable-Machine-Modell wird geladen ...")

            model = load_model()

            st.write("🔍 Gegenstand wird analysiert ...")

            prediction = model.predict(
                prepared_image,
                verbose=0
            )

            prediction = np.asarray(prediction)
            prediction = prediction.reshape(-1)

            best_index = int(
                np.argmax(prediction)
            )

            confidence = float(
                prediction[best_index]
            )

            if best_index < len(LABELS):
                detected_label = LABELS[best_index]
            else:
                detected_label = f"Klasse {best_index}"

            status.update(
                label="Klassifizierung abgeschlossen",
                state="complete",
                expanded=False
            )

        # Ergebnis
        st.markdown(
            '<div class="classification-box">',
            unsafe_allow_html=True
        )

        st.success(
            f"Erkannt: **{detected_label}**"
        )

        st.write(
            f"Erkennungswahrscheinlichkeit: "
            f"**{confidence * 100:.1f} %**"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        st.divider()

        st.markdown("### 📍 Fundort und Funddatum")

        location = st.text_input(
            "Wo wurde der Gegenstand gefunden?",
            placeholder="z. B. Sporthalle, Schulhof, Raum 204 ..."
        )

        found_date = st.date_input(
            "Wann wurde der Gegenstand gefunden?",
            value=date.today()
        )

        if st.button(
            "💾 Fundstück speichern",
            type="primary",
            use_container_width=True
        ):

            if not location.strip():

                st.warning(
                    "Bitte gib noch den Fundort ein."
                )

            else:

                save_item(
                    image=image,
                    item_type=detected_label,
                    location=location.strip(),
                    found_date=found_date
                )

                st.success(
                    "Das Fundstück wurde erfolgreich gespeichert! 🎉"
                )

                st.balloons()


# ============================================================
# SEITE 3 – ÄLTESTE FUNDSTÜCKE
# ============================================================

elif st.session_state.page == "Älteste Fundstücke":

    st.markdown(
        '<div class="main-title">Älteste Fundstücke</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Hier siehst du die neun ältesten Fundstücke.'
        '</div>',
        unsafe_allow_html=True
    )

    items = get_items(limit=9)

    if not items:

        st.info(
            "Es wurden noch keine Fundstücke gespeichert."
        )

    else:

        columns = st.columns(3)

        for index, item in enumerate(items):

            (
                item_id,
                filename,
                item_type,
                location,
                found_date,
                created_at
            ) = item

            image_path = get_image_path(filename)

            with columns[index % 3]:

                st.markdown(
                    '<div class="item-card">',
                    unsafe_allow_html=True
                )

                if image_path.exists():

                    st.image(
                        str(image_path),
                        use_container_width=True
                    )

                st.markdown(
                    f'<div class="item-title">'
                    f'{item_type}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="item-info">'
                    f'📍 {location}<br>'
                    f'📅 {found_date}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )
