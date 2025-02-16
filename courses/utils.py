from typing import NamedTuple, List
        
class Module(NamedTuple):
    title: str
    description: str

class Course(NamedTuple):
    application: str
    level: str
    job_role: str
    course_number: str
    points: int
    time: str
    num_modules: int
    objectives: str
    modules: List[Module]
    all_text: str