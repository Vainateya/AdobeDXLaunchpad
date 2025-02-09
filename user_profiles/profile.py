import json
import os
from typing import List, Dict, Optional

class UserProfileManager:
    """
    Manages user profiles stored as JSON files, allowing creation, retrieval, 
    and modification of user data. I'm trying to making this scalable so that we 
    can later adapt for cloud storage solutions.
    """
    
    PROFILE_DIR = "user_profiles"  # Directory to store user profile JSON files

    DEFAULT_PROFILE = {
        "role": None,
        "industry": None,
        "years_experience": None,
        "education": None,
        "field_of_study": None,
        "certifications": [],
        "learning_goals": [],
        "career_goals": [],
        "desired_certifications": [],
        "interest_areas": [],
        "current_skills": [],
        "desired_skills": [],
        "time_commitment": None,
        "budget": None,
        "recommended_courses": [],
        "completed_courses": [],
        "feedback": {},
        "profile_complete": False
    }
    
    def __init__(self, user_id: int):
        """
        Initializes a user profile, either loading an existing one or creating a new default profile.
        """
        self.user_id = user_id
        self.file_path = os.path.join(self.PROFILE_DIR, f"{user_id}.json")
        
        if not os.path.exists(self.PROFILE_DIR):
            os.makedirs(self.PROFILE_DIR)
        
        if os.path.exists(self.file_path):
            self.profile = self._load_profile()
        else:
            self.profile = self._create_new_profile()
    
    def _create_new_profile(self) -> Dict:
        """Creates a new default user profile and saves it as a JSON file."""
        with open(self.file_path, "w") as f:
            json.dump(self.DEFAULT_PROFILE, f, indent=4)
        return self.DEFAULT_PROFILE.copy()
    
    def _load_profile(self) -> Dict:
        """Loads an existing user profile from JSON file."""
        with open(self.file_path, "r") as f:
            return json.load(f)
    
    def update_profile(self, **kwargs) -> None:
        """Updates the user profile with new information and saves the changes."""
        for key, value in kwargs.items():
            if key in self.profile:
                self.profile[key] = value
        
        self.profile["profile_completeness"] = self._calculate_completeness()
        self._save_profile()
    
    def _calculate_completeness(self) -> str:
        """Determines the completeness status of the profile."""
        #TODO
    
    def _save_profile(self) -> None:
        """Saves the updated user profile to the JSON file."""
        with open(self.file_path, "w") as f:
            json.dump(self.profile, f, indent=4)
    
    def get_profile(self) -> Dict:
        """Returns the current state of the user profile."""
        return self.profile
    
    def delete_profile(self) -> None:
        """Deletes the user's profile JSON file permanently."""
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            self.profile = None
    
# Example Usage:
if __name__ == "__main__":
    user = UserProfileManager("user123")
    user.update_profile(role="UX Designer", desired_skills=["Adobe XD", "Motion Design"])
    print(user.get_profile())