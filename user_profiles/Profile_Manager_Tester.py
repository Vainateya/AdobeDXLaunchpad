from User_Profile_Manager import UserProfileManager

def main():
    user = UserProfileManager("user123")
    user.update_profile(role="UX Designer", desired_skills=["Adobe DX", "Motion Design"])
    print(user.get_profile())


if __name__ == "__main__":
    main()