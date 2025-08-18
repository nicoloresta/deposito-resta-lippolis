from school import StudentManagers, Student

if __name__ == "__main__":
    choice = None
    sm = StudentManagers()

    while choice != "exit":
        choice = input("Enter a command (add/show/find/change_db/create_db): ")
        if choice == "add":
            name = input("Enter student's name: ")
            surname = input("Enter student's surname: ")
            student = Student(name, surname)
            try:
                sm.add_student(student)
            except ValueError as e:
                print(e)
        elif choice == "show":
            try:
                sm.show_students()
            except ValueError as e:
                print(e)
        elif choice == "change_db":
            new_db = input("Enter the new database name: ")
            try:
                sm.set_current_db(new_db)
            except ValueError as e:
                print(e)
        elif choice == "create_db":
            new_db = input("Enter the new database name: ")
            try:
                sm.create_db(new_db)
            except ValueError as e:
                print(e)
        elif choice == "find":
            name = input("Enter student's name to find: ")
            surname = input("Enter student's surname to find: ")

            student = sm.find_student(name, surname)
            if student:
                print(f"Found student: {student}")
            else:
                print("Student not found.")

        else:
            print("Unknown command.")