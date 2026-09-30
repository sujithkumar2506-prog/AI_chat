from google import genai
from dotenv import load_dotenv

load_dotenv()

print("App starting...")
client = genai.Client()
permanent_instruction = """You are a helpful AI Engineering mentor.
Explain technical concepts simply,
with a real-world analogy and a practical example."""
prev_interaction = None
while True:
    text_input = input("Please enter some text (or type 'exit' to quit): ")
    if text_input.lower() == "exit":
        print("Exiting the app. Goodbye!")
        break
    try:
        if prev_interaction is None:
            interaction = client.interactions.create(
                model="gemini-3.8-flash",
                input=permanent_instruction + "\n" + text_input
            )
            prev_interaction = interaction
            print(interaction.output_text)
        else:
            interaction = client.interactions.create(
                model="gemini-3.8-flash",
                input=permanent_instruction + "\n" + text_input,

                previous_interaction_id =prev_interaction.id
            )
            prev_interaction = interaction
            print(interaction.output_text)
    except Exception as e:
        print(f"An error occurred: {e}")