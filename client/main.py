import gradio as gr
import json
import numpy as np
import requests
from constants import sql_statements


URL = 'http://127.0.0.1:8000'


def fetch_simple_result(data, function, column, column2, privacy, epsilon, delta):
    if not privacy:
        epsilon = 1e8
        delta = 0
    print('Column is', column, column2, 'for dset', data)
    res = requests.get(URL + '/common', params={'dataset': "health" if data == "Health Data" else "pums" if data == 'Pums Data' else 'fire','function': function,
                       'column': column if data == 'Health Data' else column2, 'epsilon': epsilon, 'delta' : delta, 'lower': None, 'upper': None})
    print(res.content)

    if res.status_code != 200:
        return res.json()['detail']
    return res.json()['result']

def fetch_sql_query(sql, privacy = True, epsilon = 0.1, delta = 1e-5):
    print('Running SQL query', sql)
    res = requests.get(URL + '/query', params={'q': sql, 'privacy': privacy, 'epsilon': epsilon, 'delta': delta})
    print('SQL Run Complete',res.content)

    if res.status_code != 200:
        return res.json()['detail']
    print('SQL Result', res.json()['output'])
    return res.json()['output']

def fetch_columns():
    # return {
    #     'Health Data': requests.get(URL + '/get_columns', params={'dataset': "health"}).json(),
    #     'PUMS Data': requests.get(URL + '/get_columns', params={'dataset': "pums"}).json(),
    # }
    return {
        'Health Data': ['Age Bracket', 'Num Cases'],
        'Fire Data': ['Box', 'Unit sequence in call dispatch', 'Number of Alarms', 'Final Priority', 'Analysis Neighborhoods']
    }
columns = fetch_columns()

with gr.Blocks() as demo:
    with gr.Column():
        gr.HTML('''
            <h1>
                Differential Privacy Demo
            </h1>
            <p style='text-align: center;'>
                Query multiple databases with differential privacy! Try it now 😉
            </p>
        ''')

        with gr.Tab(label="Basic"):
            data_dropdown = gr.Dropdown(choices=[
                                        "Health Data", 'Fire Data'], value="Health Data", label="Select a Dataset", interactive=True)

            function_dropdown = gr.Radio([
                "BoundedMean",
                "BoundedStandardDeviation",
                "BoundedSum",
                "BoundedVariance",
                "Count",
                "Max",
                "Min",
                "Median",
                "Percentile",
            ], label="Select a Function", value="Count", interactive=True)

            column_dropdown_health = gr.Dropdown(choices=['Age Bracket', 'Num Cases'], value = 'Age Bracket', interactive=True, visible = True)
            column_dropdown_fire = gr.Dropdown(choices=['Box', 'Unit sequence in call dispatch', 'Number of Alarms', 'Final Priority', 'Analysis Neighborhoods'], value = 'Final Priority', interactive=True, visible = False)

            privacy_checkbox = gr.Checkbox(
                label="Enable Privacy", value=True, interactive=True)

            epsilon_slider = gr.Slider(
                minimum=0, maximum=10, value=0.1, label="Epsilon", step=0.02, interactive=True)
            delta_slider = gr.Slider(
                minimum=0, maximum=0.1, value=0.01, label="Delta", step=0.001, interactive=True)
            data_dropdown.change(lambda x: gr.update(visible = x == 'Health Data', value = 'Age Bracket'), inputs = data_dropdown, outputs = column_dropdown_health)
            data_dropdown.change(lambda x: gr.update(visible = x == 'Fire Data', value = 'Final Priority'), inputs = data_dropdown, outputs = column_dropdown_fire)
            
            button = gr.Button(label="Query")
            button.click(lambda data, function, column, column2, privacy, epsilon, delta: fetch_simple_result(data, function, column, column2, privacy, epsilon, delta), inputs=[
                         data_dropdown, function_dropdown, column_dropdown_health, column_dropdown_fire, privacy_checkbox, epsilon_slider, delta_slider], outputs=[gr.outputs.Textbox(label="Result")])


        # '''
        with gr.Tab(label = "SQL"):
            sql_textbox = gr.Textbox(label="SQL Query", value=np.random.choice(sql_statements), lines=5)
            with gr.Row():
                sql_button = gr.Button(label="Query", show_label=True)
                random_button = gr.Button(label="Try Random", show_label=True)
            
            privacy_checkbox = gr.Checkbox(
                label="Enable Privacy", value=True, interactive=True)

            epsilon_slider = gr.Slider(
                minimum=0, maximum=5, value=0.1, label="Epsilon", step=0.1, interactive=True)
            delta_slider = gr.Slider(
                minimum=0, maximum=0.2, value=0.01, label="Delta", step=0.001, interactive=True)

            # markdown_text = gr.Markdown("```sql\nSELECT * FROM table LIMIT 10\n```", interactive=True)
            output_textbox = gr.Textbox(label="Result", lines=5, interactive=True)
            sql_button.click(lambda sql, privacy, epsilon, delta: fetch_sql_query(sql, privacy, epsilon, delta), inputs=[sql_textbox, privacy_checkbox, epsilon_slider, delta_slider], outputs=[output_textbox])

            random_button.click(lambda val:  gr.update(value = np.random.choice(sql_statements)), inputs=[random_button], outputs=[sql_textbox])


        # '''
if __name__ == '__main__':
    demo.launch(share=True)
