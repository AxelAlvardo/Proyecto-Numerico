import gradio as gr
import numpy as np
import pandas as pd
import json
import joblib

# Cargar modelo y scaler
theta = np.load("modelo_theta.npy")
scaler = joblib.load("scaler.pkl")

# Cargar columnas originales para one-hot encoding
with open("columns.json") as f:
    columns = json.load(f)

variables_categoricas = {
    "school": ["GP", "MS"],
    "sex": ["F", "M"],
    "address": ["U", "R"],
    "famsize": ["LE3", "GT3"],
    "Pstatus": ["T", "A"],
    "Mjob": ["teacher", "health", "services", "at_home", "other"],
    "Fjob": ["teacher", "health", "services", "at_home", "other"],
    "reason": ["home", "reputation", "course", "other"],
    "guardian": ["mother", "father", "other"],
    "schoolsup": ["yes", "no"],
    "famsup": ["yes", "no"],
    "paid": ["yes", "no"],
    "activities": ["yes", "no"],
    "nursery": ["yes", "no"],
    "higher": ["yes", "no"],
    "internet": ["yes", "no"],
    "romantic": ["yes", "no"],
}

variables_numericas = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2"
]

def predecir_nota(
    age, Medu, Fedu, traveltime, studytime, failures,
    famrel, freetime, goout, Dalc, Walc, health, absences,
    G1, G2,
    school, sex, address, famsize, Pstatus, Mjob, Fjob, reason, guardian,
    schoolsup, famsup, paid, activities, nursery, higher, internet, romantic
):
    datos = {
        "age": [age],
        "Medu": [Medu],
        "Fedu": [Fedu],
        "traveltime": [traveltime],
        "studytime": [studytime],
        "failures": [failures],
        "famrel": [famrel],
        "freetime": [freetime],
        "goout": [goout],
        "Dalc": [Dalc],
        "Walc": [Walc],
        "health": [health],
        "absences": [absences],
        "G1": [G1],
        "G2": [G2],
        "school": [school],
        "sex": [sex],
        "address": [address],
        "famsize": [famsize],
        "Pstatus": [Pstatus],
        "Mjob": [Mjob],
        "Fjob": [Fjob],
        "reason": [reason],
        "guardian": [guardian],
        "schoolsup": [schoolsup],
        "famsup": [famsup],
        "paid": [paid],
        "activities": [activities],
        "nursery": [nursery],
        "higher": [higher],
        "internet": [internet],
        "romantic": [romantic],
    }
    
    df = pd.DataFrame(datos)
    df_encoded = pd.get_dummies(df)
    
    for col in columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    df_encoded = df_encoded[columns]
    
    X_scaled = scaler.transform(df_encoded)
    X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
    y_pred = X_b.dot(theta)
    nota_predicha = y_pred[0,0]
    
    return f"Nota final predicha (G3): {nota_predicha:.2f}"

with gr.Blocks() as demo:
    gr.Markdown("# Predicción nota final estudiante (G3)")

    with gr.Row():
        with gr.Column():
            age = gr.Number(label="Edad (age)", value=18, precision=0)
            Medu = gr.Number(label="Educación madre (Medu)", value=2, precision=0)
            Fedu = gr.Number(label="Educación padre (Fedu)", value=2, precision=0)
            traveltime = gr.Number(label="Tiempo viaje a la escuela (traveltime)", value=1, precision=0)
            studytime = gr.Number(label="Tiempo estudio semanal (studytime)", value=2, precision=0)
            failures = gr.Number(label="Número de fallos (failures)", value=0, precision=0)
            famrel = gr.Number(label="Relación familiar (famrel)", value=3, precision=0)
            freetime = gr.Number(label="Tiempo libre después escuela (freetime)", value=3, precision=0)
            goout = gr.Number(label="Salir con amigos (goout)", value=3, precision=0)
            Dalc = gr.Number(label="Consumo alcohol día laborable (Dalc)", value=1, precision=0)
            Walc = gr.Number(label="Consumo alcohol fin semana (Walc)", value=1, precision=0)
            health = gr.Number(label="Estado salud (health)", value=3, precision=0)
            absences = gr.Number(label="Faltas a clase (absences)", value=4, precision=0)
            G1 = gr.Number(label="Nota primer trimestre (G1)", value=10, precision=0)
            G2 = gr.Number(label="Nota segundo trimestre (G2)", value=10, precision=0)

        with gr.Column():
            school = gr.Dropdown(label="Escuela (school)", choices=variables_categoricas["school"], value="GP")
            sex = gr.Dropdown(label="Sexo (sex)", choices=variables_categoricas["sex"], value="F")
            address = gr.Dropdown(label="Dirección (address)", choices=variables_categoricas["address"], value="U")
            famsize = gr.Dropdown(label="Tamaño familia (famsize)", choices=variables_categoricas["famsize"], value="GT3")
            Pstatus = gr.Dropdown(label="Estado padres (Pstatus)", choices=variables_categoricas["Pstatus"], value="T")
            Mjob = gr.Dropdown(label="Trabajo madre (Mjob)", choices=variables_categoricas["Mjob"], value="other")
            Fjob = gr.Dropdown(label="Trabajo padre (Fjob)", choices=variables_categoricas["Fjob"], value="other")
            reason = gr.Dropdown(label="Razón elección escuela (reason)", choices=variables_categoricas["reason"], value="course")
            guardian = gr.Dropdown(label="Tutor (guardian)", choices=variables_categoricas["guardian"], value="mother")
            schoolsup = gr.Dropdown(label="Apoyo escolar (schoolsup)", choices=variables_categoricas["schoolsup"], value="no")
            famsup = gr.Dropdown(label="Apoyo familiar (famsup)", choices=variables_categoricas["famsup"], value="no")
            paid = gr.Dropdown(label="Clases pagadas (paid)", choices=variables_categoricas["paid"], value="no")
            activities = gr.Dropdown(label="Actividades extra (activities)", choices=variables_categoricas["activities"], value="no")
            nursery = gr.Dropdown(label="Guardería (nursery)", choices=variables_categoricas["nursery"], value="yes")
            higher = gr.Dropdown(label="Desea educación superior (higher)", choices=variables_categoricas["higher"], value="yes")
            internet = gr.Dropdown(label="Internet en casa (internet)", choices=variables_categoricas["internet"], value="yes")
            romantic = gr.Dropdown(label="Relación romántica (romantic)", choices=variables_categoricas["romantic"], value="no")

    btn = gr.Button("Predecir Nota Final")
    output = gr.Textbox(label="Resultado")

    btn.click(
        predecir_nota,
        inputs=[
            age, Medu, Fedu, traveltime, studytime, failures,
            famrel, freetime, goout, Dalc, Walc, health, absences,
            G1, G2,
            school, sex, address, famsize, Pstatus, Mjob, Fjob, reason, guardian,
            schoolsup, famsup, paid, activities, nursery, higher, internet, romantic
        ],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
