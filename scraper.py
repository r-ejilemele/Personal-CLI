import requests
from bs4 import BeautifulSoup
from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()


def scrape_cnn():
    cnn = requests.get("https://www.cnn.com/")
    html = BeautifulSoup(cnn.text, "lxml")
    pages = [
        link.get("href") for link in html.findAll("a", class_="header__nav-item-link")
    ]
    pages = [
        i
        for i in pages
        if i is not None
        and "underscored" not in i
        and "style" not in i
        and "travel" not in i
        and "weather" not in i
        and "opinion" not in i
    ]

    possible_class_names = {
        "container__item-media-wrapper container_lead-plus-headlines-with-images__item-media-wrapper card--media-large",
        "container__link container__link--type-article container_lead-plus-headlines-with-images__link container_lead-plus-headlines-with-images__left container_lead-plus-headlines-with-images__light",
        "container__link container__link--type-article container_lead-plus-headlines__link",
        "container__link container__link--type-article container_lead-plus-headlines__link container_lead-plus-headlines__left container_lead-plus-headlines__light",
    }
    articles = dict()
    for page in pages:
        open_html = BeautifulSoup(requests.get(page).text, "lxml")

        links = []

        for option in possible_class_names:
            links.extend(
                open_html.findAll(
                    "a",
                    class_=option,
                )
            )
        for link in links:
            text = link.find("span", class_="container__headline-text")
            if text is not None:
                articles[text.text] = "https://www.cnn.com" + link.get("href")
                break
    return articles


def scraper():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    supabase: Client = create_client(url, key)
    email = os.getenv("SUPABASE_USER_EMAIL")
    password = os.getenv("SUPABASE_USER_PASSWORD")
    auth_user = supabase.auth.sign_in_with_password({"email":email, "password":password})
    # access_token = auth_user['access_token']
    # refresh_token = auth_user['refresh_token']
    response = (
        supabase.table("articles").insert({"id": 3, "article_heading": "Denmark"}).execute()
    )
    print("Added successfully")
    return
    # print(response.text)
    # print(url)
    # print(key)
    # cnn = scrape_cnn()


if __name__ == "__main__":
    scraper()
