
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
