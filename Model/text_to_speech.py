import pyttsx3

def initialize_tts():
    """
    Inicializa el motor de texto a voz con configuraciones predeterminadas.

    Retorna:
    - Un objeto `pyttsx3.Engine` listo para usar.
    """
    engine = pyttsx3.init()

    # Configurar parámetros del motor de texto a voz
    engine.setProperty('rate', 150)  # Velocidad de habla (palabras por minuto)
    engine.setProperty('volume', 0.9)  # Volumen (0.0 a 1.0)
    voices = engine.getProperty('voices')

    # Selecciona una voz en español si está disponible
    for voice in voices:
        if 'es' in voice.languages or 'spanish' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break

    return engine

def speak_text(engine, text):
    """
    Convierte el texto en audio utilizando el motor de texto a voz.

    Parámetros:
    - engine -- Objeto `pyttsx3.Engine` inicializado.
    - text -- Texto que se convertirá a voz.
    """
    if not text:
        return

    engine.say(text)
    engine.runAndWait()
