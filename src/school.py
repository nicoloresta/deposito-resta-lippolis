
class Student:
    def __init__(self, name: str, surname: str):
        self.name__ = name
        self.surname__ = surname

    def __eq__(self, other) -> bool:
        if not isinstance(other, Student):
            return False
        return self.name__ == other.name__ and self.surname__ == other.surname__

    def __repr__(self):
        return f"Name: {self.name__}, Surname: {self.surname__}"

class StudentManagers:
    def __init__(self):
        # dict[str, list[Student]]
        self.dbs = {}
        self.current_db = None

    def create_db(self, db: str) -> None:
        if db in self.dbs:
            raise ValueError(f"Database {db} already exists.")
        self.dbs[db] = []

    def set_current_db(self, db: str) -> None:
        if db not in self.dbs:
            raise ValueError(f"Database {db} does not exist.")
        self.current_db = db
        

    # Methods to implement
    def find_student(self, name: str, surname: str) -> Student:
        """
            Finds a student in a specific database
            with partial and case-insensitive matching.

            Parameters:
            ------------
            name: str 
                The name of the student to find.
            surname: str
                The surname of the student to find.
            
            Returns:
            ------------
            Student or None
                Returns the first matching student or None if not found.
            
            Raises:
            ------------
            Exception
                If no current database is set.

            TypeError:
                If the name or surname is not a string.
            
            Examples:
            ------------
            >>> find_student("Fra", "Ros")
            Name: Francesco, Surname: Rossi if found.
            None if not found.
        """

        if self.current_db is None:
            raise Exception("No current database set.")
        
        if not isinstance(name, str) or not isinstance(surname, str):
            raise TypeError("Name and surname must be strings.")
        
        for student in self.dbs[self.current_db]:
            if (name.lower() in student.name__.lower() and
                surname.lower() in student.surname__.lower()):
                return student
        return None
