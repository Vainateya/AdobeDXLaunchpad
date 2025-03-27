from typing import NamedTuple, List
import requests
from bs4 import BeautifulSoup
import numpy as np
import re
import os
import pickle
import json

class Module(NamedTuple):
    title: str
    description: str

def mod_to_dict(mod: Module):
    return {mod.title: mod.description}

def mod_to_text(mod: Module):
    return mod.title + ": " + mod.description

def save_sources_pickle(sources, filename):
    with open(filename, 'wb') as f:
        pickle.dump(sources, f)

def load_sources_pickle(filename):
    with open(filename, 'rb') as f:
        sources = pickle.load(f)
    return sources

class Source:
    def __init__(self, category: str, level: str, job_role: str, display: str, all_text: str, link: str = ''):
        self.category = category
        self.level = level
        self.job_role = job_role
        self.display = display
        self.all_text = all_text
        self.link = link
        self.course_levels = ['Foundations', 'Professional', 'Expert']
        self.certificate_levels = ['Professional', 'Expert', 'Master']
        self.job_roles = ['Developer', 'Business Practitioner', 'Architect', 'All']
        self.type = "source"

    def __str__(self):
        return self.display

    def get_embedding(self, model):
        pass
    
    def is_prereq_to(self, other_source):
        pass
        
    def is_same_role_as(self, other_source):
        if self.job_role == 'All' or other_source.job_role == 'All':
            return True
        else:
            return self.job_role == other_source.job_role
        
    def has_same(self, other_source, *features):
        if type(self) != type(other_source):
            return False
        for feature in features:
            if feature == 'category' and self.category != other_source.category:
                return False
            elif feature == 'level' and self.level != other_source.level:
                return False
            elif feature == 'job_role' and not self.is_same_role_as(other_source):
                return False
        return True
        
    def _clean(self, s):
        return re.sub(r'[\s\n]+', ' ', s).strip()
    
    def to_dict(self):
        return {
            "type": self.type,
            "category": self.category,
            "level": self.level,
            "job_role": self.job_role,
            "display": self.display,
            "link": self.link
            # "all_text": self.all_text
        }

    def to_text(self):
        return "\n".join(f"{key}: {value}" for key, value in self.to_dict().items())

class Course(Source):
    def __init__(self, link):
        r = requests.get(link)
        soup = BeautifulSoup(r.content, 'html.parser')

        div = soup.find('div', class_='table-responsive mb-0 mb-xl-4')
        rows = div.find('tbody').find('tr').find_all('td')

        application = self._clean(rows[0].text)
        level = self._clean(rows[1].text)
        job_role = self._clean(rows[2].text)
        course_number = self._clean(rows[3].text)
        points = int(self._clean(rows[4].text))
        time = self._clean(rows[5].text)
        num_modules = int(self._clean(rows[6].text))

        if job_role != 'All':
            job_role = job_role[:-1]

        display = soup.find('h1', class_='text-white').text

        course_objectives_under = soup.find('h4', string='Course objectives').find_next_sibling()
        modules_above = soup.find('h4', string='Course modules').find_previous_sibling()

        course_objectives = self._clean(course_objectives_under.text)
        ul = self._clean(modules_above.text)
        if ul != course_objectives:
            course_objectives += ' ' + ul

        div = soup.find('div', id='course-module-accordion-control')
        ms = div.find_all('div', class_='accordion-item')
        modules = []

        for m in ms:
            header = m.find('strong', class_='text-decoration-underline')
            title = self._clean(header.text)

            body = m.find('div', class_='accordion-body')
            desc = self._clean(body.text)

            modules.append(Module(title=title, description=desc))

        super().__init__(application, level, job_role, display, soup.text, link)
        self.course_number = course_number
        self.points = points
        self.time = time
        self.num_modules = num_modules
        self.objectives = course_objectives
        self.modules = modules
        self.type = "course"

    def get_embedding(self, model):
        relevant_text = [self.objectives] + [m.description for m in self.modules]
        embs = model.embed_documents(relevant_text)
        objectives_embedding, modules_embeddings = embs[0], embs[1:]
        return np.array([objectives_embedding, np.mean(modules_embeddings, axis=0)])
    
    def is_prereq_to(self, other_source):
        if self.category != other_source.category:
            return False
        if type(other_source) == Certificate:
            return self.certificate_levels.index(other_source.level) >= self.course_levels.index(self.level)
        else:
            return self.course_levels.index(other_source.level) == self.course_levels.index(self.level) + 1
    
    def to_dict(self):
        """Convert the Course object to a dictionary."""
        course_dict = super().to_dict()
        course_dict.update({
            "course_number": self.course_number,
            "points": self.points,
            "time": self.time,
            "num_modules": self.num_modules,
            "objectives": self.objectives,
            "modules": '\n'.join([mod_to_text(mod) for mod in self.modules])
        })

        return course_dict

class Certificate(Source):
    def __init__(self, file_name):
        with open(file_name, "r", encoding="utf-8") as file:
            html_content = file.read()
        html = os.path.basename(file_name)
        self.soup = BeautifulSoup(html_content, "html.parser")
        certificate = html[:-5].split()
        
        level = certificate[-1]
        certificate.pop()

        if 'Business Practitioner' in html:
            job_role = 'Business Practitioner'
            certificate = certificate[:-2]
        else:
            job_role = certificate[-1]
            certificate.pop()

        name = ' '.join(certificate)

        display = self._clean(self.soup.find('h1', class_='text-white').text)
        
        try:
            base_tag = self.soup.find('base', href=True)
            if base_tag:
                href = base_tag['href'].strip('/')
                if href.startswith("certification/"):
                    self.link = f"https://certification.adobe.com/{href}"
                else:
                    self.link = f"https://certification.adobe.com/certification/{href}"
            else:
                self.link = None
        except Exception:
            self.link = None

        
        super().__init__(name, level, job_role, display, self.soup.text, link=self.link)

        self.prereq = ', '.join(self._get_min_exp_rec())
        self.type = "certificate"
        self.details = self._get_all_exam_details()

        study_materials = self._get_exam_sections()
        study_materials_parsed = []

        for section in study_materials:
            study_materials_parsed.append(section + ': ')
            details = ''
            for detail in study_materials[section]:
                details += detail + ', '
            study_materials_parsed[-1] += details
        self.study_materials = study_materials_parsed

    def get_embedding(self, model):
        relevant_text = [self.prereq] + [i for i in self.study_materials]
        embs = model.embed_documents(relevant_text)
        min_exp, exam_objectives = embs[0], embs[1:]
        return np.array([min_exp, np.mean(exam_objectives, axis=0)])
    
    def is_prereq_to(self, other_source):
        if self.category != other_source.category:
            return False
        if type(other_source) == Course:
            return self.course_levels.index(other_source.level) >= self.certificate_levels.index(self.level) + 1
        else:
            return self.certificate_levels.index(other_source.level) == self.certificate_levels.index(self.level) + 1

    def _rem_big_spaces(self, text):
        text = text.replace("\n", "")
        return re.sub(r'\s+', ' ', text.strip()).strip()

    def _get_jobs(self) -> List[str]:
        job_section = self.soup.find("strong", string="Job Titles")
        if job_section:
            job_list = job_section.find_next("ul")  # Find the next <ul> containing job titles
            if job_list:
                job_titles = [li.text.strip() for li in job_list.find_all("li")]
                return job_titles
        return []

    def _get_min_exp_rec(self) -> List[str]:
        experience_section = self.soup.find("strong", string="Minimum experience recommended")
        experience_container = experience_section.find_parent("h6").find_next_sibling("div")
        accordion_body = experience_container.find(class_="accordion-body")
        experience_list = []
        
        for i, element in enumerate(accordion_body.contents):
            if element.name == "ul":
                experience_list.extend([self._rem_big_spaces(li.text) for li in element.find_all("li")])
            elif element.name and self._rem_big_spaces(element.text):
                experience_list.append(self._rem_big_spaces(element.text))
        return experience_list

    def _get_exam_sections(self) -> dict[str, List[str]]:
        exam_section = self.soup.find("strong", string="Exam objectives and scope")
        if exam_section:
            exam_container = exam_section.find_parent("h6").find_next_sibling("div")
            exam_dict = {}
            
            if exam_container:
                for element in exam_container.find_all("strong"):
                    section_title = self._rem_big_spaces(element.text)
                    ul = element.find_next("ul")
                    if ul:
                        exam_dict[section_title] = [self._rem_big_spaces(li.text) for li in ul.find_all("li")]
            
            return exam_dict
        else:
            return {}

    # Either "Study Resources" or "Exap Prep Guide"
    def _get_study_sections(self) -> dict[str, List[tuple[str, str]]]:
        study_section = self.soup.find("strong", string="Study resources ")
        if not study_section:
            study_section = self.soup.find("strong", string="Study resources")

        if study_section:
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
        
    def _get_exam_detail(self, detail: str):
        key_element = self.soup.find("p", string=re.compile(f"^{detail}"))
        if key_element:
            value_element = key_element.find_next("p")
            if value_element:
                return value_element.text.strip()

        return ""

    def _get_all_exam_details(self):
        exam_details_keys = [
            "EXAM ID:", "LEVEL:", "COST:", "LANGUAGE(S):", "DELIVERY:", "PASSING SCORE:", "TIME LIMIT:"
        ]
        exam_details = {}

        for key in exam_details_keys:
            exam_details[key] = self._get_exam_detail(key)

        return exam_details
    
    def to_dict(self):
        """Convert the Certificate object to a dictionary."""
        certificate_dict = super().to_dict()
        certificate_dict.update({
            "prerequisites": self.prereq,
            "study_materials": '\n'.join(self.study_materials)
        })
        return certificate_dict