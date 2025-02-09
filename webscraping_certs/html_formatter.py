from bs4 import BeautifulSoup
from typing import List
import re

from pprint import pprint

# Load the raw HTML from a file
with open("test_html.html", "r", encoding="utf-8") as file:
    html_content = file.read()

soup = BeautifulSoup(html_content, "html.parser")

def _rem_big_spaces(text):
    text = text.replace("\n", "")
    return re.sub(r'\s+', ' ', text.strip()).strip()


def get_jobs(soup: BeautifulSoup) -> List[str]:
    job_section = soup.find("strong", string="Job Titles")
    if job_section:
        job_list = job_section.find_next("ul")  # Find the next <ul> containing job titles
        if job_list:
            job_titles = [li.text.strip() for li in job_list.find_all("li")]
            return job_titles
    return []

def get_min_exp_rec(soup: BeautifulSoup) -> List[str]:
    experience_section = soup.find("strong", string="Minimum experience recommended")
    experience_container = experience_section.find_parent("h6").find_next_sibling("div")
    accordion_body = experience_container.find(class_="accordion-body")
    experience_list = []
    
    for i, element in enumerate(accordion_body.contents):
        print(i, element)
        if element.name == "ul":
            experience_list.extend([_rem_big_spaces(li.text) for li in element.find_all("li")])
        elif element.name and _rem_big_spaces(element.text):
            experience_list.append(_rem_big_spaces(element.text))
    print(experience_list)
    return experience_list

def get_exam_sections(soup: BeautifulSoup) -> dict[str, List[str]]:
    exam_section = soup.find("strong", string="Exam objectives and scope")

    if exam_section:
        exam_container = exam_section.find_parent("h6").find_next_sibling("div")
        exam_dict = {}
        
        if exam_container:
            for element in exam_container.find_all("strong"):
                section_title = _rem_big_spaces(element.text)
                ul = element.find_next("ul")
                if ul:
                    exam_dict[section_title] = [_rem_big_spaces(li.text) for li in ul.find_all("li")]
        
        return exam_dict
    else:
        return {}

def get_exam_sections(soup: BeautifulSoup) -> dict[str, List[str]]:
    exam_section = soup.find("strong", string="Study Resources")

    if exam_section:
        exam_container = exam_section.find_parent("h6").find_next_sibling("div")
        exam_dict = {}
        
        if exam_container:
            for element in exam_container.find_all("strong"):
                section_title = _rem_big_spaces(element.text)
                ul = element.find_next("ul")
                if ul:
                    exam_dict[section_title] = [_rem_big_spaces(li.text) for li in ul.find_all("li")]
        
        return exam_dict
    else:
        return {}

# Either "Study Resources" or "Exap Prep Guide"
def get_study_sections(soup: BeautifulSoup) -> dict[str, List[tuple[str, str]]]:
    study_section = soup.find("strong", string="Study resources ")
    if not study_section:
        study_section = soup.find("strong", string="Study resources")

    if study_section:
        print("has")
        study_container = study_section.find_parent("h6").find_next_sibling("div")
        study_dict = {}
        
        if study_container:
            for element in study_container.find_all("strong"):
                section_title = re.sub(r'\s+', ' ', element.text.strip())
                ul = element.find_next("ul")
                if ul:
                    study_dict[section_title] = [
                        (re.sub(r'\s+', ' ', li.text.strip()), li.find("a")["href"] if li.find("a") else None)
                        for li in ul.find_all("li")
                    ]
        
        return study_dict
    else:
        return {}

def get_exam_detail(soup: BeautifulSoup, detail: str):
    key_element = soup.find("p", string=re.compile(f"^{detail}"))
    if key_element:
        value_element = key_element.find_next("p")
        print(value_element)
        if value_element:
           return value_element.text.strip()

    return ""

def get_all_exam_details(soup: BeautifulSoup):
    # Extract exam details
    exam_details_keys = [
        "EXAM ID:", "LEVEL:", "COST:", "LANGUAGE(S):", "DELIVERY:", "PASSING SCORE:", "TIME LIMIT:"
    ]
    exam_details = {}

    for key in exam_details_keys:
        exam_details[key] = get_exam_detail(soup, key)

    return exam_details


out = get_all_exam_details(soup)
pprint(out)
