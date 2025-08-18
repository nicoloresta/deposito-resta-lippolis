
class Student:
    def __init__(self, name: str, surname: str):
        self.name__ = name
        self.surname__ = surname

    def __eq__(self, other) -> bool:
        if not isinstance(other, Student):
            return False
        return self.name__ == other.name__ and self.surname__ == other.surname__

    def __lt__(self, other) -> bool:
        if not isinstance(other, Student):
            return NotImplemented
        return (self.surname__, self.name__) < (other.surname__, other.name__)

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
    
    def add_student(self, student: Student) -> None:
        if self.current_db is None:
            raise ValueError("No current database set.")
        if student in self.dbs[self.current_db]:
            raise ValueError(f"Student {student} already exists in database {self.current_db}.")
        self.dbs[self.current_db].append(student)
    
    def show_students(self) -> None:
        if self.current_db is None:
            raise ValueError("No current database set.")
        students = sorted(self.dbs[self.current_db])
        for student in students:
            print(student)
