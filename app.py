import gradio as gr
import numpy as np
import pandas as pd
import json
import joblib
import pickle

# -----------------------------
# MODELO LINEAL MULTIVARIADO
# -----------------------------

theta = np.load("modelo_theta.npy")
scaler = joblib.load("scaler.pkl")

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
    nota_predicha = y_pred[0, 0]

    return f"Nota final predicha (G3): {nota_predicha:.2f}"



# Carga archivos logisticos (ya debes tener en tu carpeta)
beta_logistico = np.load("logistic_beta.npy")
with open("logistic_scaler.pkl", "rb") as f:
    scaler_logistico = pickle.load(f)
with open("logistic_columns.json") as f:
    columns_logistico = json.load(f)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predecir_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    # Armar dataframe o array con las columnas en orden
    X_dict = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age,
    }

    # Crear dataframe 1 fila con columnas esperadas (sin 'intercept' porque beta la incluye)
    X_df = pd.DataFrame([X_dict])

    # Escalar
    X_scaled = scaler_logistico.transform(X_df)

    # Añadir intercepto (columna de 1s)
    X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]  # shape (1, n+1)

    # Calcular z = X_b · beta
    z = np.dot(X_b, beta_logistico)  # beta_logistico es np.array de shape (n+1,)

    prob = sigmoid(z)[0].item()
    clase = 1 if prob >= 0.5 else 0
    return f"Probabilidad de diabetes: {prob:.3f} → Clase predicha: {clase}"




# -----------------------------
# FORMULARIO LOGÍSTICO (PRUEBA)
# -----------------------------
def predecir_logistico_dummy(valor):
    # Esto es solo una prueba temporal
    if valor > 0.5:
        return "Clase predicha: B"
    else:
        return "Clase predicha: A"

# -----------------------------
# INTERFAZ CON SELECCIÓN
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## Selecciona el tipo de regresión")
    selector_modelo = gr.Radio(["Regresión Lineal Multivariada", "Regresión Logística Multiclase"], value="Regresión Lineal Multivariada", label="Tipo de modelo")

    # Contenedor LINEAL MULTIVARIADA
    with gr.Column(visible=True) as form_lineal:
        gr.Markdown("### Predicción nota final estudiante (G3)")
        with gr.Row():
            with gr.Column():
                age = gr.Number(label="Edad (age)", value=18)
                Medu = gr.Number(label="Educación madre (Medu)", value=2)
                Fedu = gr.Number(label="Educación padre (Fedu)", value=2)
                traveltime = gr.Number(label="Tiempo viaje a la escuela (traveltime)", value=1)
                studytime = gr.Number(label="Tiempo estudio semanal (studytime)", value=2)
                failures = gr.Number(label="Número de fallos (failures)", value=0)
                famrel = gr.Number(label="Relación familiar (famrel)", value=3)
                freetime = gr.Number(label="Tiempo libre después escuela (freetime)", value=3)
                goout = gr.Number(label="Salir con amigos (goout)", value=3)
                Dalc = gr.Number(label="Consumo alcohol día laborable (Dalc)", value=1)
                Walc = gr.Number(label="Consumo alcohol fin semana (Walc)", value=1)
                health = gr.Number(label="Estado salud (health)", value=3)
                absences = gr.Number(label="Faltas a clase (absences)", value=4)
                G1 = gr.Number(label="Nota primer trimestre (G1)", value=10)
                G2 = gr.Number(label="Nota segundo trimestre (G2)", value=10)

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

        btn_lineal = gr.Button("Predecir Nota Final")
        output_lineal = gr.Textbox(label="Resultado")
        btn_lineal.click(
            predecir_nota,
            inputs=[
                age, Medu, Fedu, traveltime, studytime, failures,
                famrel, freetime, goout, Dalc, Walc, health, absences,
                G1, G2,
                school, sex, address, famsize, Pstatus, Mjob, Fjob, reason, guardian,
                schoolsup, famsup, paid, activities, nursery, higher, internet, romantic
            ],
            outputs=output_lineal
        )

    # Contenedor LOGÍSTICA 
    with gr.Column(visible=False) as form_logistico:
        gr.Markdown("### Predicción de Diabetes - Regresión Logística")
        Pregnancies = gr.Number(label="Número de embarazos (Pregnancies)", value=1)
        Glucose = gr.Number(label="Glucosa en sangre (Glucose)", value=120)
        BloodPressure = gr.Number(label="Presión arterial (BloodPressure)", value=70)
        SkinThickness = gr.Number(label="Grosor piel (SkinThickness)", value=20)
        Insulin = gr.Number(label="Insulina (Insulin)", value=79)
        BMI = gr.Number(label="Índice masa corporal (BMI)", value=25.0)
        DiabetesPedigreeFunction = gr.Number(label="Función pedigree diabetes", value=0.5)
        Age = gr.Number(label="Edad", value=33)

        btn_logistico = gr.Button("Predecir Diabetes")
        output_logistico = gr.Textbox(label="Resultado")
        btn_logistico.click(
            predecir_diabetes,
            inputs=[
                Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                BMI, DiabetesPedigreeFunction, Age
            ],
            outputs=output_logistico
        )


    def cambiar_formulario(opcion):
        return (
            gr.update(visible=opcion == "Regresión Lineal Multivariada"),
            gr.update(visible=opcion == "Regresión Logística Multiclase")
        )

    selector_modelo.change(cambiar_formulario, inputs=[selector_modelo], outputs=[form_lineal, form_logistico])

if __name__ == "__main__":
    demo.launch()