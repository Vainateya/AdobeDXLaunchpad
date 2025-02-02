import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

from AdobeDXLaunchpad.courses.utils import *

class Scraper:

    def _clean(self, s):
        return re.sub(r'\s+', ' ', s).strip()

    def get_course(self, link, parsed_desc=False):
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

        course_objectives = self._clean(soup.find('h4', string='Course objectives').find_next_sibling().text)

        div = soup.find('div', id='course-module-accordion-control')
        ms = div.find_all('div', class_='accordion-item')
        modules = []

        for m in ms:
            header = m.find('strong', class_='text-decoration-underline')
            title = self._clean(header.text)

            body = m.find('div', class_='accordion-body')
            if parsed_desc:
                desc = self._clean(body.find('p').text)
                desc_items = m.find_all('li')
                if len(desc_items) != 0:
                    desc += ' '
                    for item in desc_items:
                        desc += self._clean(item.text) + ', '
                    desc = desc[:-2]
            else:
                desc = self._clean(body.text)

            modules.append(Module(title=title, description=desc))

        return Course(application, level, job_role, course_number, points, time, num_modules, course_objectives, modules)

scraper = Scraper()

course_numbers = [
    1043, 1045, 1046, 1047, 1048, 1049, 1050, 1054, 1055, 1056, 
    1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 
    1067, 1068, 1069, 1221
]

courses = []

for n in tqdm(course_numbers):
    courses.append(scraper.get_course(f'https://certification.adobe.com/courses/{n}'))

print(len(courses))