import gradio as gr
import pandas as pd
import pickle

# Load model from file
model_filename = "apartments_random_forest_regressor.pkl"
with open(model_filename, mode="rb") as f:
    model = pickle.load(f)
# Load dataset
df = pd.read_csv("apartments_data_enriched_with_new_features.csv")
features = ['rooms', 'area', 'pop', 'pop_dens', 'frg_pct', 'emp', 'tax_income', 'room_per_m2', 'luxurious', 'temporary', 'furnished', 'nature', 'area_cat_ecoded', 'zurich_city']

def predict(rooms, area, pop, pop_dens, frg_pct, emp, tax_income, luxurious, temporary, furnished, nature, area_cat_ecoded, zurich_city):
    input_data = pd.DataFrame([[rooms, area, pop, pop_dens, frg_pct, emp, tax_income, 0, luxurious, temporary, furnished, nature, area_cat_ecoded, zurich_city]],
                              columns=features)
    input_data['room_per_m2'] = round(input_data['area'] / input_data['rooms'], 2)
    prediction = model.predict(input_data)[0]
    return round(float(prediction), 2)

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Rooms"),
        gr.Number(label="Area"),
        gr.Number(label="Population"),
        gr.Number(label="Population Density"),
        gr.Number(label="Foreign Percentage"),
        gr.Number(label="Employment"),
        gr.Number(label="Tax Income"),
        gr.Checkbox(label="Luxurious"),
        gr.Checkbox(label="Temporary"),
        gr.Checkbox(label="Nature"),
        gr.Checkbox(label="Furnished"),
        gr.Number(label="Area Category Encoded"),
        gr.Checkbox(label="Zurich City")
    ],
    outputs="text",
    examples=[
        [3, 100, 50000, 5000, 50, 0.8, 80000, 1, 0, 1, 0, 1, 1],
        [4, 120, 60000, 6000, 60, 0.9, 90000, 0, 1, 0, 1, 2, 1],
        [2, 80, 40000, 4000, 40, 0.7, 70000, 0, 0, 0, 0, 0, 0]
    ],
    title="Apartment Price Prediction",
    description="Enter the data of the apartment to predict the its price."
)

demo.launch()