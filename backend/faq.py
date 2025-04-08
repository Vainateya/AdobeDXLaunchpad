import requests
from bs4 import BeautifulSoup

def fetch_and_parse_faq(url="https://certification.adobe.com/support/faq") -> list[tuple[str, str]]:
    """
    parses the general faq page and returns a list of tuples in the form (question, answer)
    """
    response = requests.get(url)
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    faq_list = []

    accordion_items = soup.find_all("div", class_="accordion-item")
    for item in accordion_items:
        question_elem = item.find("button", class_="accordion-button")
        answer_elem = item.find("div", class_="accordion-body")
        if question_elem and answer_elem:
            question_text = question_elem.get_text(strip=True)
            answer_text = answer_elem.get_text(" ", strip=True)
            faq_list.append((question_text, answer_text))
    return faq_list