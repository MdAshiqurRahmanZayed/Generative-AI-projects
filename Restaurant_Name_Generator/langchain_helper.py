from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import getpass
import os
from langchain_google_genai import GoogleGenerativeAI

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")
import os

llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.6)


def generate_restaurant_name_and_items(cuisine):
    prompt_template_name = PromptTemplate(
        input_variables=["cuisine"],
        template="I want to open a restaurant for {cuisine} food. Suggest a only one fancy name for this.",
    )

    name_chain = LLMChain(
        llm=llm, prompt=prompt_template_name, output_key="restaurant_name"
    )

    prompt_template_items = PromptTemplate(
        input_variables=["restaurant_name"],
        template="""Suggest some menu items for {restaurant_name}. Return it as a comma separated string""",
    )

    food_items_chain = LLMChain(
        llm=llm, prompt=prompt_template_items, output_key="menu_items"
    )

    chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=["cuisine"],
        output_variables=["restaurant_name", "menu_items"],
    )

    response = chain({"cuisine": cuisine})

    return response


if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Italian"))
